'''
 # @ Author: Yichao Cai
 # @ Create Time: 2024-02-01 11:57:10
 # @ Description:
'''
import os.path as osp
import torch
import clip
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from utils.loss import InfoNCELossBasic, SequentialOrthogonalLoss
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from utils.misc import save_checkpoint, save_config_file, check_mkdir


class PromptStyler(nn.Module):
    """ Code adapated from: https://github.com/KaiyangZhou/CoOp/blob/main/trainers/coop.py#L247
    """
    def __init__(self, class_names, clip_model, device, K=80, n_ctx=4) -> None:
        super().__init__()
        self.n_cls = len(class_names)
        self.n_styles = K
        
        self.dtype = clip_model.dtype
        style_dim = clip_model.ln_final.weight.shape[0]
        self.device = device

        # # generate content representations for onece
        # self.contents = self.__encode_text(class_names)
        
        # initialize K style vectors
        style_vectors = torch.empty(K, n_ctx, style_dim, dtype=self.dtype).to(self.device)  # [K, 4, 512]
        nn.init.normal_(style_vectors, std=0.02)
        self.style_words = nn.Parameter(style_vectors)
        
        # template prompts 
        style_prompt = "a " + " ".join(['x']*n_ctx) + " style of a"
        style_content_prompts = [f"{style_prompt} {name}" for name in class_names]
        self.tokenized_styles = torch.cat([clip.tokenize(style_prompt) for _ in range(self.n_styles)]).to(self.device)
        self.tokenized_style_contents = torch.cat([clip.tokenize(p) for p in style_content_prompts]).to(self.device)
        with torch.no_grad():
            style_embedding = clip_model.token_embedding(self.tokenized_styles).type(self.dtype).to(self.device)   # [K, WD, 512]
            style_con_embedding = clip_model.token_embedding(self.tokenized_style_contents).type(self.dtype).to(self.device)   # [n_cls, WD, 512]
        self.register_buffer("_style_prefix", style_embedding[:, :2, :])   # [SOS]a
        self.register_buffer("_style_suffix", style_embedding[:, 2+n_ctx:, :])   # style of a[EOS]
        self.register_buffer("_style_con_prefix", style_con_embedding[:, :2, :]) # [SOS]a
        self.register_buffer("_style_con_suffix", style_con_embedding[:, 2+n_ctx:, :]) # style of a [class][EOS]
    
    def construct_style_embedding(self, id):
        """ return style_embeddings from 0 to id(included)"""
        style_embeddings = torch.cat(
            [
                self._style_prefix[:id+1, :, :],
                self.style_words[:id+1, :, :],
                self._style_suffix[:id+1, :, :]
            ],
            dim=1
        )
        return style_embeddings     # [id, *, 512]
    
    def construct_style_content_embedding(self, id):
        id_style = self.style_words[id, :, :].expand(self.n_cls, -1, -1)    # [n_cls, ]
        """ return id-th styled-content"""
        style_content_embeddings = torch.cat(
            [
                self._style_con_prefix,
                id_style,
                self._style_con_suffix
            ],
            dim=1
        )
        return style_content_embeddings # [n_cls, *, 512]

    def forward(self, id):
        styles = self.construct_style_embedding(id)
        styled_contents = self.construct_style_content_embedding(id)
        return styles, styled_contents
    

class PromptStyleTrainer:
    def __init__(self, args) -> None:
        self.args = args
        self.device = self.args.device
        self.accelerator = Accelerator()
        
        self.clip_model, _ = clip.load(self.args.clip_name, self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False     # stop gradient for CLIP
        self.clip_dim = self.clip_model.text_projection.shape[1] 
        self.clip_model.eval()
        self.clip = self.accelerator.prepare(self.clip_model)
        self.dtype = self.clip_model.dtype
        
        # encode content feature for onece
        self.n_styles = self.args.n_styles
        self.classes = sorted(self.args.class_names)
        self.content_features = self.__encode_text(self.classes)
        
        self.prompt_styler = PromptStyler(self.classes, self.clip_model, self.device, self.n_styles,
                                          n_ctx=self.args.n_ctx)
        self.prompt_styler.train()
        self.prompt_styler = self.accelerator.prepare(self.prompt_styler)
        self.tokenized_stl = self.prompt_styler.tokenized_styles
        self.tokenized_stl_cnt = self.prompt_styler.tokenized_style_contents
        
        self.content_consistency_loss = InfoNCELossBasic(tau=self.args.tau)
        self.styles_contrast_loss = SequentialOrthogonalLoss()
        self.training_steps = self.args.iterations  # iterations for training each style word
        
        self.writer = SummaryWriter()
    
    def __encode_text(self, prompts):
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        txt_features = self.clip_model.encode_text(tokenized_prompts).type(torch.float32).to(self.device)
        return txt_features
    
    def __encode_text_embedding(self, embeddings, tokenized_prompts):
        x = embeddings + \
            self.clip_model.positional_embedding.type(self.dtype).to(self.device)
        x = x.permute(1, 0, 2) # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2) # LND -> NLD
        x = self.clip_model.ln_final(x)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection 
        return x.type(torch.float32)
    
    def train(self):
        save_config_file(self.writer.log_dir, self.args)    # save config file
        
        for id in range(self.n_styles):
            print(f"Training the {id}-th style word...")
            # Only train k-th style word
            self.prompt_styler.style_words.requires_grad_ = False
            self.prompt_styler.style_words[id].requires_grad_ = True
            optimizer = self.args.optimizer(self.prompt_styler.parameters(), lr=self.args.learning_rate,
                                            momentum=self.args.momentum)
            optimizer = self.accelerator.prepare(optimizer)
            
            for iter in tqdm(range(self.training_steps)):
                style_embed, style_content_embed = self.prompt_styler(id)
                optimizer.zero_grad()
                style_features = self.__encode_text_embedding(style_embed, self.tokenized_stl[:id+1])
                style_content_features = self.__encode_text_embedding(style_content_embed, self.tokenized_stl_cnt)
                
                # forward loss
                loss_style = self.styles_contrast_loss(style_features)
                loss_content = self.content_consistency_loss(style_content_features, self.content_features)
                loss_total = loss_content + loss_style
                
                # backward & optmizing
                loss_total.backward()
                optimizer.step()
                
                self.writer.add_scalar(f"total_{id}-th", loss_total, global_step=iter)
                self.writer.add_scalar(f"style_{id}-th", loss_style, global_step=iter)
                self.writer.add_scalar(f"content_{id}-th", loss_content, global_step=iter)
            
            # save text features
            _, style_content = self.prompt_styler(id)
            with torch.no_grad():
                features2save = self.__encode_text_embedding(style_content, self.tokenized_stl_cnt)
                features2save = features2save.to("cpu").numpy()
            for c, name in enumerate(self.classes):
                save_dir = check_mkdir(osp.join(self.writer.log_dir, name))
                np.save(osp.join(save_dir, f"stye_{id}"), features2save[c])
            
        save_checkpoint(self.prompt_styler.state_dict(), is_best=False, filename=osp.join(self.writer.log_dir, "stylewords.pth"))
        print("Training completed.")
        