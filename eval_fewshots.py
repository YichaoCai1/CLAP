'''
 # @ Author: Yichao Cai
 # @ Create Time: 2024-01-28 23:40:39
 # @ Description:
 '''

import os
import argparse
import torch
import clip
import os.path as osp
import torch.utils.data as tud
from main.networks import DisentangledNetwork
from main.evaluate import training_classifier_with_img, eval_linear_prob
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
ckpt_list = ["clip.pth"]
if not configs.eval_clip:
    network = DisentangledNetwork(in_dim = clip_dim, latent_dim = configs.latent_dim, out_dim=configs.out_dim,
                                activation = configs.activation, which_network=configs.which_network,
                                repeat=configs.repeat, scale=configs.scale)
    network = network.to(configs.device)
    ckpt_list = os.listdir(configs.ckpt_path)
    
wf = open(osp.join(configs.ckpt_path, "eval_results_fewshot.txt"), 'w')

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
        # train a linear classifier with few-shots text-samples
        cls_nums = len(classes[dataset])
        wf.write("-------------------------\n")
        wf.write(f"{dataset}:\n")
        print(f"{dataset}:\n")
        
        for env in configs.eval_sets[dataset]:
            wf.write(f"\n**********:\n")
            print(f"{env}:\n")
            wf.write(f"\n{env}:\n")
            for shot in [1, 4, 8, 16, 32]:
                classifier = torch.nn.Linear(configs.out_dim, cls_nums).to(configs.device)
                if not osp.exists(osp.join(configs.ckpt_path, dataset, env, f"{shot}-shots", ckpt[:-4], "best.pth")):
                    if not osp.exists(osp.join(configs.ckpt_path, dataset)):
                        os.mkdir(osp.join(configs.ckpt_path, dataset))
                    if not osp.exists(osp.join(configs.ckpt_path, dataset, env)):
                        os.mkdir(osp.join(configs.ckpt_path, dataset, env))
                    if not osp.exists(osp.join(configs.ckpt_path, dataset, env, f"{shot}-shots")):
                        os.mkdir(osp.join(configs.ckpt_path, dataset, env, f"{shot}-shots"))
                    
                    print("Training a linear classifier using few-shot images for linear probing...")
                    linDataset = MultiEnvDataset(osp.join(f"data/fewshot_datasets/{shot}-shots", dataset), test_env=env, transform=preprocess)
                    dataloader = tud.DataLoader(linDataset, batch_size=32, shuffle=True, num_workers=24, drop_last=False)
                    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.005)
                    loss_func = torch.nn.CrossEntropyLoss().to(configs.device)            
                    training_classifier_with_img(clip_model=clip_model, network=network, classifier=classifier,
                                        save_path=osp.join(configs.ckpt_path, dataset, env, f"{shot}-shots", ckpt[:-4]),
                                        dataloader=dataloader, optimizer=optimizer,
                                        loss_func=loss_func, device=configs.device, epochs=1000,
                                        eval_clip=configs.eval_clip)
            
                # load few-shot classifier
                state = torch.load(osp.join(configs.ckpt_path, dataset, env, f"{shot}-shots", ckpt[:-4], "best.pth"))
                classifier.load_state_dict(state, strict=True)


                print(f"{shot}-shots:  ")
                wf.write(f"{shot}-shots:  ")
                evalset = MultiEnvDataset(osp.join("data/datasets", dataset),
                                            test_env=env, transform=preprocess)
                loader = tud.DataLoader(evalset, batch_size=16, shuffle=True, num_workers=16)
                
                acc_ones = eval_linear_prob(clip_model, network, classifier, loader, 
                                        device=configs.device, adver=None, cls_nums=cls_nums,\
                                            eval_clip=configs.eval_clip)
                wf.write(f"\t{acc_ones},\t")
                print(f"\t{acc_ones},\t")
