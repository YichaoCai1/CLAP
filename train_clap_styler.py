'''
 # @ Author: Yichao Cai
 # @ Create Time: 2024-01-19 13:24:13
 # @ Description: Training script
 '''
import argparse
import torch
from main.styler_trainer import ClipHCLTrainer
from utils.misc import Args, set_manual_seed, load_property

parser = argparse.ArgumentParser()
parser.add_argument("config", help="The path of config file for training.")
args = parser.parse_args()


# loading configs
configs = Args(args.config)
configs.set_device("cuda" if torch.cuda.is_available() else "cpu")
print(configs.device)
classes = []
for dset in configs.datasets:
    classes.extend(load_property(f"data/classes/{dset}.yaml"))
configs.set_property("class_names", sorted(list(set(classes))))


# set manual seed
if configs.manual_seed:
    set_manual_seed(configs.manual_seed)
print(configs)


# training the disentangled network
trainer = ClipHCLTrainer(configs)
trainer.train()         
