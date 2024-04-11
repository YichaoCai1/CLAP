import torch
import clip
import numpy as np
import random
from tqdm import tqdm
from sklearn.manifold import TSNE
import torch.utils.data as tud
from main.networks import DisentangledNetwork
from utils.data_utils import MultiEnvDataset
from utils.misc import scale_to_01_range
from matplotlib import pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

seed = 2023
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


dset = "PACS"
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']


def build_model(ckpt_path, latent_dim, which_net, alpha=1):
    global device
    disentangler = DisentangledNetwork(in_dim = 512, latent_dim = latent_dim, out_dim=512, activation = torch.nn.SiLU, which_network=which_net, scale=alpha)
    disentangler = disentangler.to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    disentangler.load_state_dict(state_dict=state_dict, strict=True)
    disentangler.eval()
    return disentangler


def plot_tsne(features, class_names, save_path):
    global colors
    tsne = TSNE(n_components=2, perplexity=30).fit_transform(features)
    tx = scale_to_01_range(tsne[:, 0])
    ty = scale_to_01_range(tsne[:, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for lb, clr in enumerate(colors[:len(class_names)]):
        indices = [i for i, l in enumerate(labels) if l == lb]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        ax.scatter(current_tx, current_ty, c=clr, label=class_names[lb], alpha=0.6)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig(save_path)
    fig.clear()
    


# loading models
clip_model, preprocess = clip.load('ViT-B/16', device=device)

clap_path = "runs/Results_CLAP_ViTB/PACS/checkpoint_7200.pth" if dset == "PACS" else None
if clap_path is None:
    raise NotImplementedError

imgaug_path = "runs/Results_ImgAug_ViTB/PACS_ImgAug/checkpoint_7200.pth"

alpha = 0.208 
latent_dim = 256 
network_clap = build_model(clap_path, latent_dim, which_net=["beta"], alpha=alpha)
network_imaug = build_model(imgaug_path, latent_dim, which_net=["psi"], alpha=alpha)

# load dataset
envs = ["art_painting", "cartoon", "photo", "sketch"]
print(f"{dset}:\n")
for env in envs:
    print(f"{env}:\n")
    clip_features = np.empty((0, 512))
    clap_features = np.empty((0, 512))
    imaug_features = np.empty((0, 512))
    labels = np.empty(0)
    envdset = MultiEnvDataset('data/datasets/'+dset+'/', test_env=env, 
                            transform=preprocess)
    class_names = envdset.sorted_classes
    dloader = tud.DataLoader(envdset, batch_size=32, shuffle=False, drop_last=False)

    # inference
    for x, y in tqdm(dloader):       
        x = x.to(device)
        labels = np.concatenate((labels, y.cpu().detach().numpy()))

        x = clip_model.encode_image(x).to(torch.float32)
        f_clip = x.cpu().detach().numpy()
        clip_features = np.concatenate((clip_features, f_clip))

        f_clap = network_clap(x).cpu().detach().numpy()
        clap_features = np.concatenate((clap_features, f_clap))
        
        f_imaug = network_imaug(x).cpu().detach().numpy()
        imaug_features = np.concatenate((imaug_features, f_imaug))

    # plot t-sne
    plot_tsne(clip_features, class_names, f"visuals/tsne-CLIP-{dset}-{env}.pdf")
    plot_tsne(clap_features, class_names, f"visuals/tsne-CLAP-{dset}-{env}.pdf")
    plot_tsne(imaug_features, class_names, f"visuals/tsne-IMAUG-{dset}-{env}.pdf")
