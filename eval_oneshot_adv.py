'''
 # @ Author: Yichao Cai
 # @ Create Time: 2024-02-20 16:46:01
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
from main.evaluate import eval_linear_prob
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
classes["PACS"] = load_property(r"data/classes/PACS.yaml")
classes["VLCS"] = load_property(r"data/classes/VLCS.yaml")
classes["OfficeHome"] = load_property(r"data/classes/OfficeHome.yaml")
classes["DomainNet"] = load_property(r"data/classes/DomainNet.yaml")

clip_model, preprocess = clip.load(configs.clip_name, device=configs.device)
input_size = clip_model.visual.input_resolution     # CLIP's input size
clip_dim = clip_model.text_projection.shape[1]      # CLIP's representation dimension

network = None
if not configs.eval_clip:
    network = DisentangledNetwork(in_dim = clip_dim, latent_dim = configs.latent_dim, out_dim=configs.out_dim,
                                activation = configs.activation, which_network=configs.which_network,
                                repeat=configs.repeat, scale=configs.scale)
    network = network.to(configs.device)

wf = open(osp.join(configs.ckpt_path, "eval_results_oneshot_adv.txt"), 'w')
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
        wf.write(f"\n{dataset}:\n")
        print(f"\n{dataset}:\n")
        for env in configs.eval_sets[dataset]:
            wf.write(f"\n**********:\n")
            # train a linear classifier with 1-shot (class name)
            cls_nums = len(classes[dataset])
            classifier = torch.nn.Linear(configs.out_dim, cls_nums).to(configs.device)
            
            # load few-shot classifier
            state = torch.load(osp.join(configs.ckpt_path, dataset, env, f"1-shots", ckpt[:-4], "best.pth"))
            classifier.load_state_dict(state, strict=True)
            
            
            evalset = MultiEnvDataset(osp.join("data/datasets", dataset),
                                        test_env=env, transform=preprocess)
            loader = tud.DataLoader(evalset, batch_size=32, shuffle=True, num_workers=24)
            
            print(f"{env}:\n")
            wf.write(f"\n{env}:\n")
            
            advers = ["FGSM", "PGD-20", "CW-20"]
            for adver in advers:
                acc_ones = eval_linear_prob(clip_model, network, classifier, loader, 
                                        device=configs.device, adver=adver, cls_nums=cls_nums,
                                        eval_clip=configs.eval_clip)

                wf.write(f"\t{adver} -- {acc_ones:.2f},\t")
                print(f"\t{adver} -- {acc_ones:.2f}\t")     
