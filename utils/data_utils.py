'''
 # @ Author: Yichao Cai
 # @ Create Time: 2024-01-16 14:22:36
 # @ Description: data utilizing
 '''
import os
import torch
import copy
import random
import numpy as np
from PIL import Image
import os.path as osp
import torch.utils.data as tud
import torchvision.transforms as tfs
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from typing import Iterable
from utils.eda import eda


class ImageTextDataset(tud.Dataset):
    ''' Dataset return: 
            Text of a class name, text prompts of the class, 
            & synthetic images corresponding to the prompt
    '''
    def __init__(self, classes, prompt_path, img_path, input_size, 
                 augment_args, prompts_aug_params,
                 visual=False) -> None:  
        super().__init__()
        self.classes = sorted(classes)
        self.num_classes = len(self.classes)
        self._prompt_path = prompt_path
        self._img_path = img_path
        self._input_size = input_size
        self._augment_args = augment_args
        self._visual=visual

        # Prompt styles
        self._object_size = [" large ", " normal sized ", " small "]
        self._object_color = [" red", " blue ", " green ", " yellow ", " purple ",\
            " orange ", " black ", " white ", " brown ", " multicolored "]
        self._art_style = [" realistic ", " impressionistic "]
        self._image_type = ["painting ", " sketch ", " cartoon ", " clipart ", " photograph ",
                            " infograph ", " mosaic art ", " sculpture "]
        self._prompts_aug_params = prompts_aug_params
        
        # Ban bad synthetic data with specific words in the prompt
        self._ban_list = {} # {" mosaic art ", " sculpture "} #  {" multicolored ", " mosaic art "}
        
        if self._img_path is not None:
            self.__set_transform()
        self._txt_samples = self.__set_samples_list() # [n*c1, n*c2, ..., n*cn]
        self._volume = len(self._txt_samples)
        self.samples_per_class = self._volume//self.num_classes
        print(f"Prompt samples per class: {self.samples_per_class}.")
        
        self._class_indexed_txts = []
        for cls in range(self.num_classes):
            cls_prompts =  copy.deepcopy(self._txt_samples[cls*(self._volume // self.num_classes): \
                (cls+1)*(self._volume//self.num_classes)])  # for safe shuffle
            random.shuffle(cls_prompts)
            self._class_indexed_txts.append(cls_prompts)  # random shuffle
    
    def __set_samples_list(self):
        text_samples = []   # {("text prompt", "class name")}
        for class_name in self.classes:
            prompt_file = osp.join(self._prompt_path, class_name+".txt")
            with open(prompt_file, 'r') as rf:
                cnt = 0
                for prompt in rf.readlines():
                    if self.__ban_sample(prompt):
                        continue
                    prompt = prompt.rstrip('\n')
                    if len(prompt) > 0:
                        cnt += 1
                        text_samples.append((class_name, prompt))
                rf.close()
        return text_samples      
    
    def __ban_sample(self, prompt):
        for ban_word in self._ban_list:
            if prompt.find(ban_word) != -1:
                return True
        return False
    
    def __random_transform_prompt(self, prompt):    
        replace = self._prompts_aug_params["replace"] 
        if torch.rand(1) < self._prompts_aug_params["object_size"]:
            prompt = self.__replace_word(prompt, self._object_size) if replace \
                else self.__drop_word(prompt, self._object_size)
                
        if torch.rand(1) < self._prompts_aug_params["object_color"]:
            prompt = self.__replace_word(prompt, self._object_color) if replace \
                else self.__drop_word(prompt, self._object_color)
        
        if torch.rand(1) < self._prompts_aug_params["art_style"]:
            prompt = self.__replace_word(prompt, self._art_style) if replace \
                else self.__drop_word(prompt, self._art_style)
                    
        if torch.rand(1) < self._prompts_aug_params["img_type"]:
            prompt = self.__replace_word(prompt, self._image_type) if replace \
                else self.__drop_word(prompt, self._image_type)
                
        if torch.rand(1) < self._prompts_aug_params["reverse"]:
            prompt = self.__reverse_prompt(prompt)
            
        return prompt.replace("  ", " ")
    
    @staticmethod
    def __drop_word(prompt, wd_list):
        for wd in wd_list:
            if wd in prompt:
                prompt = prompt.replace(wd, " ")
                break
        return prompt
    
    @staticmethod
    def __replace_word(prompt, wd_list):
        tmp_list = wd_list.copy()
        tmp_list.append(" ")
        for wd in wd_list:
            if wd in prompt:
                tmp_list.remove(wd)
                prompt = prompt.replace(wd, random.sample(tmp_list, k=1)[0])
                break
        return prompt
    
    @staticmethod
    def __reverse_prompt(prompt):
        of_pos = prompt.find(" of ")
        if of_pos == -1:
            return prompt
        img_info = prompt[:of_pos]
        object_info = prompt[of_pos+4:]
        return object_info + " in " + img_info + " style"
    
    def __eda_augment_prompt(self, prompt):
        augmented_prompt = eda(prompt)[0]
        return augmented_prompt
    
    def __set_transform(self):
        if self._img_path is None:
            return None
        
        def _convert_image_to_rgb(image):
            return image.convert("RGB")
        
        trans_aug = [v2.ColorJitter(brightness=.5, hue=.3),   # random color distortion
                    RandomMask(p=self._augment_args['mask'][0], ratio=self._augment_args['mask'][1], \
                        k1=self._augment_args['mask'][2], k2=self._augment_args['mask'][3]),
                    v2.RandomPerspective(distortion_scale=0.5, p=self._augment_args["perspective"]),     
                    tfs.RandomResizedCrop(scale=self._augment_args["crop"], ratio=(1., 1.), size=self._input_size, 
                                          antialias=True, interpolation=tfs.InterpolationMode.BICUBIC),
                    tfs.RandomHorizontalFlip(p=self._augment_args["hflip"]),
                    tfs.CenterCrop(self._input_size),
                    _convert_image_to_rgb,
                    tfs.ToTensor()]
        trans = [tfs.Resize(self._input_size, interpolation=tfs.InterpolationMode.BICUBIC),
                 tfs.CenterCrop(self._input_size),
                 _convert_image_to_rgb,
                 tfs.ToTensor()]
        
        if self._visual == False:
            trans_aug.extend([tfs.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
            trans.extend([tfs.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        self._transforms = tfs.Compose(trans)
        self._transforms_aug = tfs.Compose(trans_aug)
    
    def __len__(self) -> int:
        return self._volume
    
    def __getitem__(self, index):
        # if self._unique_class: 
        class_name, prompt = self._class_indexed_txts[index%self.num_classes][index//self.num_classes]
        # else:
        #     class_name, prompt = self._txt_samples[index]
        
        if self._prompts_aug_params["eda"]:
            prompt_aug = self.__eda_augment_prompt(prompt)
        else:
            prompt_aug = self.__random_transform_prompt(prompt)
            
        if self._img_path is None:
            return class_name, prompt, prompt_aug
        
        sync_imgs_path = osp.join(self._img_path, class_name, prompt)
        img_name = random.sample(os.listdir(sync_imgs_path), k=1)[0]
        img = Image.open(osp.join(sync_imgs_path, img_name))
        img_aug = self._transforms_aug(img)        
        img = self._transforms(img)
        
        return class_name, prompt, prompt_aug, img, img_aug


class MultiEnvDataset(tud.Dataset):
    def __init__(self, data_root, test_env, transform) -> None:
        super().__init__()
        self.dataset = ImageFolder(osp.join(data_root, test_env), transform=transform)
        self.classes = [name.replace("_", " ").lower() for name in self.dataset.classes]
        self.sorted_classes = sorted(self.classes)
        self.prompts = {'C':[], 'PC':[], 'CP':[]}
        self.prompts['C'] = self.sorted_classes
        self.prompts['CP'] = [f"a {name} in a photo" for name in self.sorted_classes]
        self.prompts['PC'] = [f"a photo of a {name}" for name in self.sorted_classes]
        # print(self.prompts['C'])
        # print(self.prompts['CP'])
        # print(self.prompts['PC'])
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        y = self.sorted_classes.index(self.classes[y])      # some systems do not load filefold in order
        return x, y


class PromptStylerLinearDataset(tud.Dataset):
    def __init__(self, root_path, class_names, device, K=80) -> None:
        super().__init__()
        self.class_names = sorted(class_names)
        self.n_cls = len(class_names)
        self._volume = self.n_cls * K
        self.device = device
        self.root_path = root_path
        self.K = K
        self.features_cache = {cls_name:{} for cls_name in self.class_names}
    
    def __len__(self):
        return self._volume
    
    def __getitem__(self, index):   # [n1k1,...,n1kK, ..., nNk1, nNkK]
        label = index//self.K
        name = self.class_names[label]
        style_id = index%self.K
        text = f"stye_{style_id}.npy"
        if not text in self.features_cache:
            feature = np.load(osp.join(self.root_path, name, text))
            self.features_cache[name][text] = torch.from_numpy(feature).to(torch.float32)
        feature = self.features_cache[name][text]
        return feature, label
    

class PromptStylerCLDataset(tud.Dataset):
    def __init__(self, clip_name, class_names, root_path, K=80) -> None:
        super().__init__()
        self.class_names = sorted(class_names)
        self.n_cls = len(class_names)
        self._volume = self.n_cls * K
        self.path = osp.join(root_path, clip_name.replace('/', '_'))
        self.K = K
        print("Cacheing features...")
        self.features_cache = self.__cache_features()
        print("Finished cache~")
        
    def __cache_features(self):
        cache = []
        for name in self.class_names:
            class_cache = []
            for stl_id in range(self.K):
                np_arr = np.load(osp.join(self.path, name, f"stye_{stl_id}.npy"))
                feature = torch.from_numpy(np_arr).to(torch.float32)
                class_cache.append(feature)
            cache.append(class_cache)
        return cache
        
    def __len__(self):
        return self.n_cls
    
    def __getitem__(self, index): 
        name = self.class_names[index]
        style_ids = random.sample(range(0, self.K), k=2)
        feature1 = self.features_cache[index][style_ids[0]]
        feature2 = self.features_cache[index][style_ids[1]]
        return name, feature1, feature2


class InfiniteIterator:
    ''' Infinitely repeat the iterable.
    '''
    def __init__(self, iterable: Iterable):
        self._iterable = iterable
        self.iterator = iter(self._iterable)

    def __iter__(self):
        return self

    def __next__(self):
        for _ in range(2):
            try:
                return next(self.iterator)
            except StopIteration:
                # reset iterator
                del self.iterator
                self.iterator = iter(self._iterable)


class RandomMask(torch.nn.Module):
    def __init__(self, p=0.5, ratio=0.3, k1=8, k2=16, val=0) -> None:
        ''' ratio: max ratio of masked blocks, int(max_ratio*k_size^2)
            p: probability of random mask
            k1,k2: lower and higher bound of block numbers, random[k1, k2)^2
            val: filling value of the masked blocks
        '''
        super().__init__()
        self.ratio= ratio
        self.p = p
        self.k1 = k1
        self.k2 = k2
        self.val = val
    
    def forward(self, image):
        if torch.rand(1) >= self.p:
            return image
        
        k_num = int(np.random.uniform(low=self.k1, high=self.k2))
        indices = list(range(k_num*k_num))
        im = np.array(image)
        H, W = im.shape[:2]
        kh = H // k_num
        kw = W // k_num
        for id in random.sample(indices, random.randint(0, int(self.ratio * len(indices)))):
            i = id // k_num
            j = id % k_num
            im[i*kh:(i+1)*kh, j*kw:(j+1)*kw, :] = self.val
        return Image.fromarray(im.astype('uint8')).convert('RGB')


class FewShotNameDataset(tud.Dataset):
    ''' Dataset used for linear probing.
    '''
    def __init__(self, class_names, shots=1) -> None:
        " 1, 8, 24, 48, 480 shots"
        super().__init__()
        self.indexed_names = sorted(class_names)
        self.atts = {
            "color": ["red ", "blue ", "green ", "yellow ", "purple ",
                            "orange ", "black ", "white ", "brown ", "multicolored "],
            "size": ["large ", "normal sized ", "small "],
            "type": ["painting ", "sketch ", "cartoon ", "clipart ", "photograph ",
                            "infograph ", "mosaic art ", "sculpture "],
            "style": ["realistic ", "impressionistic "]
        }
        
        self.samples = copy.deepcopy(self.indexed_names)
        self._volume = len(self.indexed_names)
        self.labels = list(range(self._volume))
        
        
        if shots >= 8:    # 8-shot [Type][CLASS]
            new_smaples = []
            new_labels = []
            for id, sample in enumerate(self.samples):
                for word in self.atts["type"]:
                    new_smaples.append("a " + word + "of " + sample)
                    new_labels.append(self.labels[id])
            self.samples = new_smaples
            self.labels = new_labels
        
        if shots >= 24:   # 24-shot [Type][Size][CLASS]
            new_smaples = []
            new_labels = []
            for id, sample in enumerate(self.samples):
                for word in self.atts["size"]:
                    spaces = self.__index_spaces(sample)
                    new_sample = self.__insert_word(sample, spaces[2], " a " + word)
                    new_sample = new_sample.replace("  ", " ")
                    new_smaples.append(new_sample)
                    new_labels.append(self.labels[id])
            self.samples = new_smaples
            self.labels = new_labels
        
        if shots >= 48:    # 16-shot [Style][Type][Size][CLASS]
            new_smaples = []
            new_labels = []
            for id, sample in enumerate(self.samples):
                for word in self.atts["style"]:
                    new_smaples.append("a " + word + sample)
                    new_labels.append(self.labels[id])
            self.samples = new_smaples
            self.labels = new_labels
            
        if shots >= 480:   # 48-shot [Style][Type][Size][Color][CLASS]
            new_smaples = []
            new_labels = []
            for id, sample in enumerate(self.samples):
                for word in self.atts["color"]:
                    spaces = self.__index_spaces(sample)
                    if "normal sized" in sample:
                        new_sample = self.__insert_word(sample, spaces[7]+1, word)
                    else:
                        new_sample = self.__insert_word(sample, spaces[6]+1, word)
                    new_sample = new_sample.replace("  ", " ")
                    new_smaples.append(new_sample)
                    new_labels.append(self.labels[id])
            self.samples = new_smaples
            self.labels = new_labels
        self._volume = len(self.samples)

    @staticmethod
    def __index_spaces(prompt):
        spaces = []
        for i in range(len(prompt)):
            if prompt[i] == " ":
                spaces.append(i)
        return spaces    
    
    @staticmethod
    def __insert_word(prompt, index, word):
        prompt_list = list(prompt)
        prompt_list.insert(index, word)
        return ''.join(prompt_list)
    
    def __len__(self) -> int:
        return self._volume
    
    def __getitem__(self, index):
        return self.samples[index], self.labels[index]



if __name__ == "__main__":
    classes = ["bird", "dog", "person"]
    shots = [1, 8, 24, 48, 480]
    for shot in shots:
        dataset = FewShotNameDataset(class_names=classes, shots=shot)
        dataloader = tud.DataLoader(dataset, batch_size=2, shuffle=True)
        print(f"{shot}-shot:")
        for text, label in dataloader:
            print(text)
            print(label)
            print("=======================================\n")
            break
    
          
    