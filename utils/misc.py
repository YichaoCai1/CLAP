# -*- coding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2023/10/23 13:20:11
@Author  :   YichaoCai 
@Version :   1.0
@Desc    :   None
'''

import os
import os.path as osp
import yaml
import clip
import torch
import shutil
import random
import numpy as np
               

def load_property(yaml_path):
    with open(yaml_path) as rf:
        return list(yaml.safe_load(rf))

def check_mkdir(path_dir):
    if not osp.exists(path_dir):
        os.mkdir(path_dir)
        print(f"\'{path_dir}\'-- directory made.")
    return path_dir

def set_manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


def load_property(yaml_path):
    with open(yaml_path) as rf:
        return sorted(list(yaml.safe_load(rf)))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def correct(output, target):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return correct[0].sum(0, keepdim=True)
    

class Args:
    def __init__(self, config_file) -> None:
        self.__dict__.update(self.fetch_configs(config_file))

    def fetch_configs(self, config_file):
        def eval_dict(dict_in):
            for k in dict_in:
                if isinstance(dict_in[k], str):
                    try:
                        if "path" in k or "root" in k or \
                            "clip_name" in k:
                            continue
                        dict_in[k] = eval(dict_in[k])
                    except NameError:
                        continue
                elif isinstance(dict_in[k], dict):
                    eval_dict(dict_in[k])
            return dict_in

        with open(config_file, 'r', encoding="utf-8") as rf:
            return eval_dict(yaml.safe_load(rf))

    def set_device(self, device):
        self.device = device
        
    def set_property(self, key, val):
        self.__dict__[key] = val

    def __repr__(self) -> str:
        return f"{self.__dict__}"
    
    def __str__(self) -> str:
        return f"{self.__dict__}"
    

class EarlyStopping:
    """ Early stopping if the loss doesn't improve after a given patience."""
    def __init__(self, patience=10, delta=0, start_save=0) -> None:
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.delta = delta
        self.start_save = start_save
    
    def __call__(self, loss, model, step_id, save_path):
        score = -loss
        if self.best_score is None:
            self.best_score = score
            if step_id >= self.start_save:
                save_checkpoint(model.state_dict(), is_best=False, filename=save_path)
                return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.loss_min = loss
            if step_id >= self.start_save:
                save_checkpoint(model.state_dict(), is_best=False, filename=save_path)
                return True
        return False

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
 
    return starts_from_zero / value_range


def normalize_features(*args):
    features = []
    for feature in args:
        if feature is not None:
            feature = feature / feature.norm(dim=-1, keepdim=True)
        features.append(feature)
    return features


def clip_encode_noised_text(clip_model, texts, n_ctx=4, position="front", std=0.02):
    """
    Add a random Gaussian (0, std) noise  to a text;
    Position "front" means the noise is add before the orginal text.
        Code adapted from: https://github.com/KaiyangZhou/CoOp/blob/main/trainers/coop.py
    """
    if clip_model is None or texts is None:
        return None
    
    n_cls = len(texts)
    ctx_dim = clip_model.ln_final.weight.shape[0]
    simple_tokenizer = clip.simple_tokenizer.SimpleTokenizer()
    dtype = clip_model.dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).to(device)
    torch.nn.init.normal_(ctx_vectors, std=0.02)
    ctx_vectors = ctx_vectors.unsqueeze(0).expand(n_cls, -1, -1)
    
    texts_len = [len(simple_tokenizer.encode(name)) for name in texts]
    prompts = [" ".join(["X"] * n_ctx) + " " + name for name in texts]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
    with torch.no_grad():
        embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(device)
    token_prefix = embedding[:, :1, :]  # SOS
    token_suffix = embedding[:, 1+n_ctx:, :]    # CLS, EOS
    
    # concatenating Gaussian noise
    if position == "front": # before class
        prompts_embeddings = torch.cat([token_prefix, ctx_vectors, token_suffix], \
            dim=1).type(dtype).to(device)
    elif position == "middle":
        prompts_embeddings = []
        half_len_cxt = n_ctx // 2
        for i in range(n_cls):
            t_len = texts_len[i]
            prefix = token_prefix[i:i+1, :, :]
            token_cls = token_suffix[i:i+1, :t_len, :]
            suffix = token_suffix[i:i+1, t_len:, :]
            ctx_half1 = ctx_vectors[i:i+1, :half_len_cxt, :]
            ctx_half2 = ctx_vectors[i:i+1, half_len_cxt:, :]
            embed = torch.cat([prefix, ctx_half1, token_cls, ctx_half2, suffix], dim=1)
            prompts_embeddings.append(embed)
        prompts_embeddings = torch.cat(prompts_embeddings, dim=0).type(dtype).to(device)
    else:
        prompts_embeddings = []
        for i in range(n_cls):
            t_len = texts_len[i]
            prefix = token_prefix[i:i+1, :, :]
            token_cls = token_suffix[i:i+1, :t_len, :]
            suffix = token_suffix[i:i+1, t_len:, :]
            ctx = ctx_vectors[i:i+1, :, :]
            embed = torch.cat([prefix, token_cls, ctx, suffix], dim=1)
            prompts_embeddings.append(embed)
        prompts_embeddings = torch.cat(prompts_embeddings, dim=0).type(dtype).to(device)
    
    # Inference noised text embeddings with clip model
    with torch.no_grad():
        x = prompts_embeddings + clip_model.positional_embedding.type(dtype).to(device)
        x = x.permute(1, 0, 2) # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2) # LND -> NLD
        x = clip_model.ln_final(x)
        
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ clip_model.text_projection 
    return x.type(torch.float32)
    
   