import os
import json
import random

out_path = '/home/james/JA-PDVC/data/custom'
if not os.path.exists(out_path):
    os.mkdir(out_path)

split_ratio = [90, 10]  # 70% training, 30% validation

with open('/home/james/mp4video/all_anno.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

keys = list(data.keys())
random.shuffle(keys)

train_idx = int(len(keys) * split_ratio[0] / 100)

train_dict = {key: data[key] for key in keys[:train_idx]}
val_dict = {key: data[key] for key in keys[train_idx:]}

with open(os.path.join(out_path, 'train_anno.json'), 'w') as file:
    json.dump(train_dict, file, indent=4)
with open(os.path.join(out_path, 'val_anno.json'), 'w') as file:
    json.dump(val_dict, file, indent=4)
