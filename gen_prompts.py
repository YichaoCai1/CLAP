import os
import yaml
import itertools
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', type=str, help="The directory name of your dataset.")
args = parser.parse_args()

os.chdir('..')
def check_mkdir(path_dir):
    if not osp.exists(path_dir):
        os.mkdir(path_dir)
    print(f"\'{path_dir}\'-- directory made.")
    return path_dir

def load_property(yaml_path):
    with open(yaml_path) as rf:
        return sorted(list(yaml.safe_load(rf)))

def assemble_prompt(prompt):
    def fit_qunatifier(text):
        # if text[-1] != 's':
        return "an " + text if (text[0] in "aeiou") else "a " + text
        # return text
    
    name = prompt["NAME"]
    name = ' '.join([prompt["SZ"], prompt["CLR"], name])
    style = ' '.join([prompt["COND"], prompt["IMSTL"]]).lstrip(' ').rstrip(' ')
    style = fit_qunatifier(style)
    res_prompt = f"{style} of " + fit_qunatifier(name.lstrip(' ').rstrip(' '))
    return res_prompt
    
prompts_path = check_mkdir(r"./prompts")
color_props = load_property(r"./data/color.yaml")
style_props = load_property(r"./data/art_style.yaml")
size_props = load_property(r"./data/size.yaml")
image_styles = load_property(r"./data/image_type.yaml")


dset_path = check_mkdir(osp.join(prompts_path, args.dataset_name))
name_path = f"./data/classes/{args.dataset_name}.yaml"
names = load_property(name_path)

for name in names:
    prompt_file = osp.join(dset_path, name+".txt")
    
    with open(prompt_file, 'w') as wf:
        name_imstl = itertools.product([name], image_styles)
        name_imstl_size = itertools.product(name_imstl, size_props)
        name_imstl_size_con = itertools.product(name_imstl_size, style_props)
        name_imstl_size_con_clr = itertools.product(name_imstl_size_con, color_props)
    
        for item in name_imstl_size_con_clr:
            prompt = {"NAME": item[0][0][0][0], "IMSTL": item[0][0][0][1], 
                    #  "SZ": item[0][0][1], "COND": item[0][1], "CLR": item[1]}
                    "SZ": item[0][0][1], "COND": "", "CLR": item[1]}
            prompt = assemble_prompt(prompt)
            print(prompt)
            wf.write(prompt+'\n')
        wf.close()
