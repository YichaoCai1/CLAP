'''
 # @ Author: Yichao Cai
 # @ Create Time: 2024-02-07 13:40:23
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
classes["PACS"] = load_property(r"data/classes/PACS.yaml")
classes["VLCS"] = load_property(r"data/classes/VLCS.yaml")
classes["OfficeHome"] = load_property(r"data/classes/OfficeHome.yaml")
classes["DomainNet"] = load_property(r"data/classes/DomainNet.yaml")

clip_model, preprocess = clip.load(configs.clip_name, device=configs.device)
clip_dim = clip_model.text_projection.shape[1]      # CLIP's representation dimension

def one_eval_get_zspc(scale):
    global configs
    global clip_model
    global preprocess
    global clip_dim
    
    network = DisentangledNetwork(in_dim = clip_dim, latent_dim = configs.latent_dim, out_dim=configs.out_dim,
                            activation = configs.activation, which_network=configs.which_network,
                            repeat=configs.repeat, scale=scale)
    network = network.to(configs.device)
    
    ckpt_name = None
    for file_name in os.listdir(configs.ckpt_path):
        if file_name.endswith(".pth"):
            ckpt_name = file_name
            break
    dataset = "PACS"
    state_dict = torch.load(osp.join(configs.ckpt_path, ckpt_name), map_location=configs.device)
    network.load_state_dict(state_dict=state_dict, strict=True)
    network.eval()
    cls_nums = len(classes[dataset])

    mean_acc_c = []
    mean_acc_pc = []
    mean_acc_cp = []
    for env in configs.eval_sets[dataset]:
        evalset = MultiEnvDataset(osp.join("data/datasets", dataset),
                                    test_env=env, transform=preprocess)
        loader = tud.DataLoader(evalset, batch_size=128, shuffle=True, num_workers=24)
        
        _, acc_zs = eval_zero_shot(clip_model, network, None, loader, evalset.prompts,
                                device=configs.device, adver=None, cls_nums=cls_nums,
                                eval_clip=False)
        mean_acc_c.append(acc_zs[0])
        mean_acc_cp.append(acc_zs[1])
        mean_acc_pc.append(acc_zs[2])
    return np.mean(mean_acc_c), np.mean(mean_acc_cp), np.mean(mean_acc_pc)


def binary_search(scale_left, scale_right, next_infer="all", acc_optimals=None):
    if next_infer == "all":
        acc_left_c, acc_left_cp, acc_left_pc = one_eval_get_zspc(scale_left)
        acc_right_c, acc_right_cp, acc_right_pc = one_eval_get_zspc(scale_right)
    elif next_infer == "left":
        acc_left_c, acc_left_cp, acc_left_pc = one_eval_get_zspc(scale_left)
        acc_right_c, acc_right_cp, acc_right_pc = acc_optimals
    else:
        acc_left_c, acc_left_cp, acc_left_pc = acc_optimals
        acc_right_c, acc_right_cp, acc_right_pc = one_eval_get_zspc(scale_right)
    
    # comparison target
    acc_left = acc_left_pc 
    acc_right = acc_right_pc 
    
    if acc_left < acc_right:
        scale_optimal = scale_right
        acc_optimals = (acc_right_c, acc_right_cp, acc_right_pc)
        scale_left = 0.5 * (scale_right + scale_left)  
        next_infer = "left"
    else:
        scale_optimal = scale_left
        acc_optimals = (acc_left_c, acc_left_cp, acc_left_pc)
        scale_right = 0.5 * (scale_right + scale_left)  
        next_infer = "right"
    print(f"\n scale: {scale_optimal:.3f}, acc_c: {acc_optimals[0]:.2f}, acc_cp: {acc_optimals[1]:.2f}, acc_pc: {acc_optimals[2]:.2f}")
    print(f"next scale left: {scale_left:.3f}, next scale right: {scale_right:.3f}\n")
    
    return scale_left, scale_right, next_infer, acc_optimals


print(osp.join(configs.ckpt_path))
scale_list = [np.power(10, -1.5), np.power(10, -1.), np.power(10, -0.5), np.power(10, 0.), np.power(10, 1.), np.power(10, 1.5), np.power(10, 2.)]
acc_list = []
for scale in scale_list:
    acc_c, acc_cp, acc_pc = one_eval_get_zspc(scale)
    print(f"\n============\n scale: {scale:.4f}, acc_c: {acc_c:.2f}, acc_cp: {acc_cp:.2f}, acc_pc: {acc_pc:.2f}")
    acc_list.append(acc_pc)
index = np.argmax(acc_list)
print(index)


# scale_left = 0.1 
# scale_right = 0.3162
# next_infer = "all"
# acc_optimals = None
# for step in range(10):
#     scale_left, scale_right, next_infer, acc_optimals = binary_search(scale_left, scale_right, next_infer, acc_optimals)