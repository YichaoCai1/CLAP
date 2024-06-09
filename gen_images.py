import os
import argparse
import random
import numpy as np
import torch
import yaml
import os.path as osp
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", help="The dataset name for image synthesizing.")
args = parser.parse_args()

def fix_random_seed(m_seed=2024):
    random.seed(m_seed)
    np.random.seed(m_seed)
    torch.manual_seed(m_seed)

def load_property(yaml_path):
    with open(yaml_path) as rf:
        return list(yaml.safe_load(rf))

def check_mkdir(path_dir):
    if not osp.exists(path_dir):
        os.mkdir(path_dir)
    print(f"\'{path_dir}\'-- directory made.")
    return path_dir


# Fix seed
fix_random_seed(m_seed=2024)
class_names = load_property(f"data/classes/{args.dataset_name}.yaml")

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# image shape & number per forward run
height = 512
width = 512
num_images = 1

# Sythesizing images
prompts_path = f"data/prompts_template/{args.dataset_name}"
root_path = check_mkdir("data/synthesis_images")     
random_bg = " in a random environment"
for name in class_names:
    save_path = check_mkdir(osp.join(root_path, name))
    load_path = osp.join(prompts_path, name+'.txt')
    print(f"Synthesizing images for {name}...")
    with open(load_path, 'r') as rf:
        for line in tqdm(rf.readlines()):
            prompt = line.rstrip('\n')
            img_path = check_mkdir(osp.join(save_path, prompt))
            if osp.exists(img_path):
                img_count = 0
                for id in range(1, num_images+1):
                    if osp.exists(osp.join(img_path, prompt+f"_{id+1}.png")):
                        img_count += 1
                if img_count == num_images:
                    continue
            
            # generation
            images = pipe(prompt + random_bg, height=height, width=width, 
                          num_images_per_prompt=num_images).images        
            for id, image in enumerate(images):
                image.save(osp.join(img_path, prompt+f"_{id+1}.png"))

