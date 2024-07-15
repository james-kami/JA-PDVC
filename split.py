import os
import numpy as np
import json
import random

out_path = '/home/james/JA-PDVC/data/custom'
if not os.path.exists(out_path):
    os.mkdir(out_path)
split_ratio = [60, 30]
with open('/home/james/mp4video/train_anno.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


keys = list(data.keys())
values = list(data.values())

selected_indices = random.sample(range(len(keys)), split_ratio[0])

selected_dict = {keys[i]: values[i] for i in selected_indices}

remaining_indices = [i for i in range(len(keys)) if i not in selected_indices]
remaining_dict = {keys[i]: values[i] for i in remaining_indices}

with open(out_path + '/train_anno.json', 'w') as selected_file:
    json.dump(selected_dict, selected_file, indent=4)

# 保存剩余的字典到另一个JSON文件
with open(out_path + '/val_anno.json', 'w') as remaining_file:
    json.dump(remaining_dict, remaining_file, indent=4)

