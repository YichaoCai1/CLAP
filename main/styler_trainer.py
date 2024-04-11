'''
 # @ Author: Yichao Cai
 # @ Create Time: 2024-01-18 10:58:27
 # @ Description: Trainer Class
 '''
import logging
import numpy as np
import torch
import clip
import os.path as osp
from tqdm import tqdm
from accelerate import Accelerator
import torch.utils.data as tud
from utils.loss import InfoNCELoss
from torch.utils.tensorboard import SummaryWriter
from utils.misc import save_checkpoint, save_config_file, EarlyStopping
from utils.data_utils import InfiniteIterator, PromptStylerCLDataset
from main.networks import DisentangledNetwork


class ClipHCLTrainer:
    def __init__(self, args) -> None:
        self.args = args
        self.device = self.args.device
        self.accelerator = Accelerator()
        
        # ''' set CLIP model'''
        self.clip, _ = clip.load(self.args.clip_name, self.device)
        for param in self.clip.parameters():
            param.requires_grad = False     # stop gradient for CLIP
        self.input_size = self.clip.visual.input_resolution     # CLIP's input size
        self.clip_dim = self.clip.text_projection.shape[1]      # CLIP's representation dimension

        self.clip.eval()
        self.clip = self.accelerator.prepare(self.clip)
        self.dtype = self.clip.dtype
        self.simple_tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        
        # ''' set dataset with full prompts and synthetic images'''
        if self.args.class_names:

            self.dataset = PromptStylerCLDataset(clip_name=self.args.clip_name, class_names=self.args.class_names,
                                                 root_path=self.args.prompts_path, K=80)
            self.data_iter = InfiniteIterator(self.accelerator.prepare(tud.DataLoader(dataset=self.dataset, batch_size=self.args.batch_size, 
                                                                  pin_memory=True, shuffle=self.args.shuffle, drop_last=False, num_workers=16)))

        # ''' set the network'''
        self.network = DisentangledNetwork(in_dim=self.clip_dim, latent_dim=self.args.latent_dim, out_dim=self.args.out_dim, 
                                      activation=self.args.activation, drop_rate = self.args.drop_rate_psi_n_beta,
                                      which_network=["beta"])

        self.network = self.network.to(self.device)        
        self.network.train()
        self.network = self.accelerator.prepare(self.network)
        
        # ''' set optimizer, loss criterion'''
        self.optimizer = args.optimizer(self.network.parameters(), args.learning_rate, weight_decay=args.weight_decay)
        self.optimizer = self.accelerator.prepare(self.optimizer)
        if self.args.early_stop:
            self.earlystopping = EarlyStopping(patience=self.args.early_stop["patience"], delta=self.args.early_stop["delta"],
                                            start_save=self.args.check_frequency)
        self.loss_text_modal =  InfoNCELoss(self.args.tau_txt)
        self.loss_align_content = InfoNCELoss(tau=args.tau_cnt)
        
        # ''' set logger'''
        self.writer = SummaryWriter()
        logging.basicConfig(filename=osp.join(self.writer.log_dir, "training.log"), level=logging.DEBUG)
        self.log_freq = self.args.log_frequency
        self.check_freq = self.args.check_frequency

    def __encode_txt(self, prompts):
        if prompts is None:
            return None
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        txt = self.clip.encode_text(tokenized_prompts).type(torch.float32).to(self.device)
        return txt

    def train(self):
        save_config_file(self.writer.log_dir, self.args) # save config file of this traning 
        
        total_losses = []
        
        for step in tqdm(range(1, int(self.args.total_steps)+1)):
            # laod a batch
            c, t, t_ = next(self.data_iter)            
            # forward CLIP
            z_c = self.__encode_txt(c)
            if isinstance(self.dataset, PromptStylerCLDataset):
                z_t = t.type(torch.float32).to(self.device)
                z_t_ = t_.type(torch.float32).to(self.device)
            else:
                z_t = self.__encode_txt(t)
                z_t_ = self.__encode_txt(t_)
            
            self.optimizer.zero_grad()  
            # forward loss             
            z_t = self.network(z_t)
            z_t_ = self.network(z_t_)
            
            total_loss = self.loss_text_modal(z_t, z_t_) + self.loss_align_content(z_t, z_c)
            
            # backward loss & optimizing
            total_loss.backward()
            self.optimizer.step()
            
            # tensorboard logging
            total_losses.append(total_loss.item())
            if step % self.log_freq == 0:
                n_log = step // self.log_freq
                if n_log == 0:
                    continue
                self.writer.add_scalar("total", np.mean(total_losses[(n_log-1)*self.log_freq+1: n_log*self.log_freq+1]), global_step=step)
            
            # check & save checkpoint
            if step % self.check_freq == 0:
                n_check = step // self.check_freq
                if n_check == 0:
                    continue
                local_mean_loss = np.mean(total_losses[(n_check-1)*self.check_freq+1: n_check*self.check_freq+1])
                logging.debug(f"Steps: {step}\tLoss: {local_mean_loss}.")
                
                checkpoint_name = osp.join(self.writer.log_dir, f"checkpoint_{step}.pth")
                if self.args.early_stop:
                    self.earlystopping(local_mean_loss, self.network, step, checkpoint_name)
                    if self.earlystopping.early_stop:
                        logging.info(f"Early stopped at step: {step}.")
                        break
                else:
                    save_checkpoint(self.network.state_dict(), is_best=False, filename=checkpoint_name)
        logging.info("Training completed.")
        logging.info(f"Model checkpoints has been saved at {self.writer.log_dir}.")
        