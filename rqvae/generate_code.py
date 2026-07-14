import collections
import json
import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE

import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

# Attention: Remember to change the dataset name and checkpoint path
# if you want to generate codes for other datasets.
dataset = "Sephora_sliding"
ckpt_path = f"./ckpt/{dataset}/Jul-10-2026_16-49-14/best_collision_model.pth"
output_file = f"../data/process_data/{dataset}/{dataset}_t5_rqvae.npy"
device = torch.device("cuda:1")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
args = ckpt["args"]
state_dict = ckpt["state_dict"]


data = EmbDataset(f'../data/process_data/{args.dataset}/item_emb.parquet')
print('args.num_emb_list',args.num_emb_list)
model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices = []
all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>","<f_{}>","<g_{}>"]

for d in tqdm(data_loader):
    d = d.to(device)
    indices = model.get_indices(d,use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    # print('indices',indices)
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))

        all_indices.append(code)
        all_indices_str.append(str(code))
    # break

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0

tt = 0
#There are often duplicate items in the dataset, and we no longer differentiate them
while True:
    if tt >= 30 or check_collision(all_indices_str):
        break

    collision_item_groups = get_collision_item(all_indices_str)
    print(collision_item_groups)
    print(len(collision_item_groups))
    for collision_items in collision_item_groups:
        d = data[collision_items].to(device)

        indices = model.get_indices(d, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices[item] = code
            all_indices_str[item] = str(code)
    tt += 1


print("All indices number: ",len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)


all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)

    
# initialize a list to store the converted codes
codes = []

# iterate through the dictionary and convert each list of indices to a code
for key, value in all_indices_dict.items():
    code = [int(item.split('_')[1].strip('>')) for item in value]
    codes.append(code)

# convert the list of codes to a numpy array
codes_array = np.array(codes)

# Add an extra dimension to all codes
codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

# Resolve duplicates by incrementing the last dimension
unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
duplicates = unique_codes[counts > 1]

if len(duplicates) > 0:
    print("Resolving duplicates in codes...")
    for duplicate in duplicates:
        duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
        for i, idx in enumerate(duplicate_indices):
            codes_array[idx, -1] = i  # Increment the last digit for resolving duplicates

new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
duplicates = new_unique_codes[new_counts > 1]

if len(duplicates) > 0:
    print("There still have duplicates:", duplicates)
else:
    print("There are no duplicates in the codes after resolution.")

# save the codes to a numpy file
print(f"Saving codes to {output_file}")
print(f"the first 5 codes: {codes_array[:5]}")
np.save(output_file, codes_array)

# ===== 新增：分析 codes_array 各维度分布 =====
print("\n" + "="*60)
print("Analysis of codes_array per dimension:")
print("="*60)

num_dims = codes_array.shape[1]
overall_max = codes_array.max()
overall_min = codes_array.min()

print(f"Overall min: {overall_min}, Overall max: {overall_max}")
print(f"Shape of codes_array: {codes_array.shape} (num_items x num_code_dimensions)\n")

for dim in range(num_dims):
    col = codes_array[:, dim]
    unique_vals = np.unique(col)
    print(f"Dimension {dim}:")
    print(f"  - Min: {col.min()}")
    print(f"  - Max: {col.max()}")
    print(f"  - Num unique values: {len(unique_vals)}")
    print(f"  - Most frequent value: {np.bincount(col).argmax()} (count: {np.bincount(col).max()})")
    
    large_vals = col[col > 600]
    if len(large_vals) > 0:
        print(f"  - ⚠️  Large values (>600): {sorted(np.unique(large_vals))[:10]}{'...' if len(np.unique(large_vals)) > 10 else ''}")
    
    if dim == num_dims - 1:
        print(f"  - 💡 This is the duplicate-resolution column (added for collision handling)")
    print()

large_token_mask = codes_array > 512
items_with_large_tokens = np.where(large_token_mask.any(axis=1))[0]
print(f"Number of items containing tokens > 512: {len(items_with_large_tokens)}")
if len(items_with_large_tokens) > 0:
    print(f"Example item codes with large tokens (first 3):")
    for i in items_with_large_tokens[:3]:
        print(f"  Item {i}: {codes_array[i]}")

analysis_file = output_file.replace(".npy", "_analysis.txt")
with open(analysis_file, "w") as f:
    f.write(f"Overall min: {overall_min}, Overall max: {overall_max}\n")
    f.write(f"Shape: {codes_array.shape}\n\n")
    for dim in range(num_dims):
        col = codes_array[:, dim]
        f.write(f"Dim {dim}: min={col.min()}, max={col.max()}, unique={len(np.unique(col))}\n")
    f.write(f"\nItems with tokens > 512: {len(items_with_large_tokens)}\n")

print(f"✅ Detailed analysis saved to: {analysis_file}")
print("="*60)