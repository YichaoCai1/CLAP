'''
 # @ Author: Yichao Cai
 # @ Create Time: 2024-01-19 20:10:23
 # @ Description:
 '''

import os
import numpy as np
import argparse
import torch
import clip
import os.path as osp
import torch.utils.data as tud
from main.networks import DisentangledNetwork
from main.evaluate import eval_zero_shot
from utils.misc import Args, set_manual_seed, load_property
from utils.data_utils import MultiEnvDataset


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


parser = argparse.ArgumentParser()
parser.add_argument("config", help="The path of config file for evaluation.")
args = parser.parse_args()

# loading configs
configs = Args(args.config)
configs.set_device("cuda" if torch.cuda.is_available() else "cpu")
# print(configs)

# mannual seed
if configs.manual_seed:
    set_manual_seed(configs.manual_seed)

classes = {}
# classes["PACS"] = load_property(r"data/classes/PACS.yaml")
# classes["VLCS"] = load_property(r"data/classes/VLCS.yaml")
# classes["OfficeHome"] = load_property(r"data/classes/OfficeHome.yaml")
# classes["DomainNet"] = load_property(r"data/classes/DomainNet.yaml")
class_names_path = r"data/classes/"
for class_file in os.listdir(class_names_path):
    classes[class_file[:-5]] = load_property(osp.join(class_names_path, class_file))

clip_model, preprocess = clip.load(configs.clip_name, device=configs.device)
input_size = clip_model.visual.input_resolution     # CLIP's input size
clip_dim = clip_model.text_projection.shape[1]      # CLIP's representation dimension

network = None
if not configs.eval_clip:
    network = DisentangledNetwork(in_dim = clip_dim, latent_dim = configs.latent_dim, out_dim=configs.out_dim,
                                activation = configs.activation, which_network=configs.which_network,
                                repeat=configs.repeat, scale=configs.scale)
    network = network.to(configs.device)

wf = open(osp.join(configs.ckpt_path, "eval_results_zeroshot.txt"), 'w')

ckpt_list = ["clip.pth"]
if not configs.eval_clip:
    ckpt_list = os.listdir(configs.ckpt_path)

# Evaluations
for ckpt in ckpt_list:
    if not ckpt.endswith(".pth"):
        continue
    wf.write("\n\n--------------------------------------------------------\n")
    wf.write(f"\nEvaluation results of {ckpt}:\n")
    print(f"\nEvaluation results of {ckpt}:\n")
    
    if not configs.eval_clip:
        state_dict = torch.load(osp.join(configs.ckpt_path, ckpt), map_location=configs.device)
        network.load_state_dict(state_dict=state_dict, strict=True)
        network.eval()
    
    for dataset in configs.eval_sets:
        # train a linear classifier with 1-shot (class name)
        cls_nums = len(classes[dataset])
        
        advers = ["None", "FGSM", "PGD-20", "CW-20"]
        for env in configs.eval_sets[dataset]:
            evalset = MultiEnvDataset(osp.join("data/datasets", dataset),
                                        test_env=env, transform=preprocess)
            loader = tud.DataLoader(evalset, batch_size=32, shuffle=True, num_workers=24)
            
            print(f"{env}:\n")
            wf.write(f"\n{env}:\n")
            
            for adver in advers:
                acc_zeros, _ = eval_zero_shot(clip_model, network, loader, evalset.prompts,
                                        device=configs.device, adver=adver, cls_nums=cls_nums,
                                        eval_clip=configs.eval_clip)
                
                wf.write(f"\t{adver} -- {acc_zeros},\t")
                print(f"\t{adver} -- {acc_zeros},\t\t{adver}\t")     