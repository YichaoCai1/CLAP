import os
import yaml
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', type=str, help="The directory name of your dataset.")
args = parser.parse_args()

dataset_path = os.path.join('data/datasets/', args.dataset_name)
domain1 = os.listdir(dataset_path)[0]
names_root = os.path.join(dataset_path, domain1)

names = [name.replace('_', ' ') for name in os.listdir(names_root)]

with open(os.path.join('data/classes/', f'{args.dataset_name}.yaml'), 'w') as file:
    yaml.dump(names, file)
    file.close()

