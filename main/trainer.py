'''
 # @ Author: Yichao Cai
 # @ Create Time: 2024-01-18 10:58:27
 # @ Description: Trainer Class
 '''
import logging
import random
import numpy as np
import torch
import clip
import os.path as osp
from tqdm import tqdm
from accelerate import Accelerator
import torch.utils.data as tud
from utils.loss import VICRegLoss, InfoNCELoss
from torch.utils.tensorboard import SummaryWriter
from utils.misc import save_checkpoint, save_config_file, clip_encode_noised_text, EarlyStopping
from utils.data_utils import ImageTextDataset, InfiniteIterator
from main.networks import DisentangledNetwork

class ClipHCLTrainer:
    def __init__(self, args) -> None:
        self.args = args
        self.device = self.args.device
        self.accelerator = Accelerator()
        
        # ''' Set CLIP model'''
        self.clip, _ = clip.load(self.args.clip_name, self.device)
        for param in self.clip.parameters():
            param.requires_grad = False     # Stop gradient for CLIP
        self.input_size = self.clip.visual.input_resolution     # CLIP's input size
        self.clip_dim = self.clip.text_projection.shape[1]      # CLIP's representation dimension

        self.clip.eval()
        self.clip = self.accelerator.prepare(self.clip)
        self.dtype = self.clip.dtype
        self.simple_tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        
        # ''' Set dataset with full prompts and synthetic images'''
        if self.args.class_names:
            self.dataset = ImageTextDataset(classes=self.args.class_names, prompt_path=self.args.prompts_path, 
                                        img_path=self.args.images_path, input_size=self.input_size,
                                        augment_args=self.args.augmentations, prompts_aug_params=self.args.prompt_aug_params)
            self.data_iter = InfiniteIterator(self.accelerator.prepare(tud.DataLoader(dataset=self.dataset, batch_size=self.args.batch_size, 
                                                                  pin_memory=True, shuffle=self.args.shuffle, drop_last=False, num_workers=16)))
        self.last_ckpt_path = None
        
        # ''' Set the network'''
        self.nets2train = self.args.which_network
        self.network = DisentangledNetwork(in_dim=self.clip_dim, latent_dim=self.args.latent_dim, out_dim=self.args.out_dim, 
                                      activation=self.args.activation, drop_rate = self.args.drop_rate_psi_n_beta,
                                      which_network=self.nets2train, repeat=self.args.repeat, scale=self.args.scale)

        self.network = self.network.to(self.device)
        if self.args.load_weights is not None:
            for subnet in self.args.load_weights:
                ckpt = torch.load(self.args.weights_path, map_location=self.device)
                for module in ckpt.keys():
                    if subnet in module:
                        self.network.state_dict()[module] == ckpt[module]
        
        self.network.train()
        self.network = self.accelerator.prepare(self.network)
        
        # ''' Set optimizer, loss criterion'''
        self.optimizer = args.optimizer(self.network.parameters(), args.learning_rate, weight_decay=args.weight_decay)
        self.optimizer = self.accelerator.prepare(self.optimizer)
        if self.args.early_stop:
            self.earlystopping = EarlyStopping(patience=self.args.early_stop["patience"], delta=self.args.early_stop["delta"],
                                            start_save=self.args.check_frequency)
        if 'psi' in self.nets2train:
            self.loss_image_modal = InfoNCELoss(self.args.tau_img)
        if 'beta' in self.nets2train:
            self.loss_text_modal =  InfoNCELoss(self.args.tau_txt)
        self.loss_align_content = InfoNCELoss(tau=args.tau_cnt)
        
        # ''' Set logger'''
        self.writer = SummaryWriter()
        logging.basicConfig(filename=osp.join(self.writer.log_dir, "training.log"), level=logging.DEBUG)
        self.log_freq = self.args.log_frequency
        self.check_freq = self.args.check_frequency

    def __encode_noised_text(self, texts, n_ctx=4, std=0.02):
        if texts is None:
            return None
    
        if n_ctx < 1:
            return self.__encode_txt(texts)
        
        nois_pos = random.sample(["front", "middle", "end"], k=1)[0]
        noised_text = clip_encode_noised_text(self.clip, texts, n_ctx=n_ctx, position=nois_pos, std=std)
        return noised_text
    
    def __encode_txt(self, prompts):
        if prompts is None:
            return None
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        txt = self.clip.encode_text(tokenized_prompts).type(torch.float32).to(self.device)
        return txt
    
    def __encode_img(self, imgs):
        if imgs is None:
            return None
        imgs = self.clip.encode_image(imgs).type(torch.float32).to(self.device)
        return imgs
        
    def train(self):
        save_config_file(self.writer.log_dir, self.args) # save config file of this traning 
        
        total_losses = []
        losses_img = []
        losses_txt = []
        losses_cross = []
        
        for step in tqdm(range(1, int(self.args.total_steps)+1)):
            # laod a batch
            if 'psi' in self.nets2train:
                c, t, t_, x, x_ = next(self.data_iter)
            else:
                c, t, t_ = next(self.data_iter)
                x, x_ = None, None
            
            self.optimizer.zero_grad()  
            
            # forward CLIP
            with torch.no_grad():
                z_c = self.__encode_txt(c)
                z_t = self.__encode_txt(t)
                z_t_ = self.__encode_noised_text(t_, self.args.noise_len, self.args.std)
                z_x = self.__encode_img(x)
                z_x_ = self.__encode_img(x_)
            
            # forward loss 
            loss_img = 0.0
            loss_txt = 0.0
            loss_cross = 0.0
            if self.nets2train is None:
                checkpoint_name = osp.join(self.writer.log_dir, f"checkpoint_ident.pth")
                save_checkpoint(self.network.state_dict(), is_best=False, filename=checkpoint_name)
                break
            
            if 'psi' in self.nets2train:
                z_x = self.network(z_x)
                z_x_ = self.network(z_x_)
                loss_img = self.loss_image_modal(z_x, z_x_)
                total_loss = loss_img
            if 'beta' in self.nets2train:
                z_t = self.network(z_t)
                z_t_ = self.network(z_t_)
                z_c = self.network(z_c)
                
                loss_txt_1 = self.loss_text_modal(z_t, z_t_)
                loss_txt_2 = self.loss_align_content(z_t_, z_c) 
                total_loss = loss_txt_1 + self.args.scale_coef * loss_txt_2
            
            # backward loss & optimizing
            total_loss.backward()
            self.optimizer.step()
            
            # tensorboard logging
            total_losses.append(total_loss.item())
            losses_img.append(loss_img.item() if isinstance(loss_img, torch.Tensor) else loss_img)
            losses_txt.append(loss_txt.item() if isinstance(loss_txt, torch.Tensor) else loss_txt)
            losses_cross.append(loss_cross.item() if isinstance(loss_cross, torch.Tensor) else loss_cross)
            if step % self.log_freq == 0:
                n_log = step // self.log_freq
                if n_log == 0:
                    continue
                self.writer.add_scalar("total", np.mean(total_losses[(n_log-1)*self.log_freq+1: n_log*self.log_freq+1]), global_step=step)
                self.writer.add_scalar("images", np.mean(losses_img[(n_log-1)*self.log_freq+1: n_log*self.log_freq+1]), global_step=step)
                self.writer.add_scalar("text", np.mean(losses_txt[(n_log-1)*self.log_freq+1: n_log*self.log_freq+1]), global_step=step)
                self.writer.add_scalar("cross-modal", np.mean(losses_cross[(n_log-1)*self.log_freq+1: n_log*self.log_freq+1]), global_step=step)
            
            # check & save checkpoint
            if step % self.check_freq == 0:
                n_check = step // self.check_freq
                if n_check == 0:
                    continue
                local_mean_loss = np.mean(total_losses[(n_check-1)*self.check_freq+1: n_check*self.check_freq+1])
                logging.debug(f"Steps: {step}\tLoss: {local_mean_loss}.")
                
                checkpoint_name = osp.join(self.writer.log_dir, f"checkpoint_{step}.pth")
                if self.args.early_stop:
                    if self.earlystopping(local_mean_loss, self.network, step, checkpoint_name):
                        self.last_ckpt_path = checkpoint_name
                    if self.earlystopping.early_stop:
                        logging.info(f"Early stopped at step: {step}.")
                        break
                else:
                    save_checkpoint(self.network.state_dict(), is_best=False, filename=checkpoint_name)
                    self.last_ckpt_path = checkpoint_name
        logging.info("Training completed.")
        logging.info(f"Model checkpoints has been saved at {self.writer.log_dir}.")