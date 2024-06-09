import random
import os
import shutil
import os.path as osp
from utils.misc import check_mkdir, set_manual_seed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', type=str, help="The directory name of your dataset.")
parser.add_argument('--seed', type=int, default=2024, help="Set manual random noise.")
args = parser.parse_args()

# Conserdered few-shots [1, 4, 8, 16, 32], 
# Since OfficeHome does not contain many instance for each class
set_manual_seed(args.seed)


shots = [1, 4, 8, 16, 32]
fewshot_path = check_mkdir("data/fewshot_datasets/")
fewshot_paths = []
for shot in shots:    
    fewshot_paths.append(check_mkdir(osp.join(fewshot_path, f"{shot}-shots")))


# datasets = ["PACS", "VLCS", "OfficeHome", "DomainNet"]  #  evaluate datasets
# for dset in datasets:
dset = args.dataset_name
dset_paths = []
for fs_path in fewshot_paths: 
    dset_paths.append(check_mkdir(osp.join(fs_path, dset)))
src_path = osp.join('data/datasets', dset)
envs = os.listdir(src_path)
for env in envs:
    src_env_path = osp.join(src_path, env)
    tar_env_paths = []
    for dset_p in dset_paths:
        tar_env_paths.append(check_mkdir(osp.join(dset_p, env)))
    for name in os.listdir(src_env_path):
        src_img_path = osp.join(src_env_path, name)
        src_list = os.listdir(src_img_path)
        
        for i, shot in enumerate(shots):
            tar_dir = check_mkdir(osp.join(tar_env_paths[i], name))
            if shot < len(src_list):
                tar_list = random.sample(src_list, k=shot)  # random sample k=shot images
            else:
                tar_list = src_list
            
            for img in tar_list:
                src_img = osp.join(src_img_path, img)
                tar_img = osp.join(tar_dir, img)
                shutil.copy(src_img, tar_img)
