import os
import glob
import argparse
from typing import List, Tuple, Dict

import torch
import numpy as np
from tqdm import tqdm
from ase.io import read  
from collections import OrderedDict

from k_frequencies import EwaldPotential
from unique_boxes import update_unique_boxes

def build_cache(
    all_cells: torch.Tensor,
    dl: float,
    sigma: float,
    device: str = "cpu",
    out_name: str = "kvec_cache.pt",
) -> Tuple[str, int]:

    ewald = EwaldPotential(dl=dl, sigma=sigma).to(device)
    cache: Dict[str, Dict[str, torch.Tensor]] = {}

    for idx, cell in enumerate(all_cells):
        cell_key = tuple(round(x, 8) for x in cell.double().tolist())
        print(f"cell_key: {cell_key}")
        cell = cell.to(device)

        kvecs, _ = ewald(cell)

        cache[cell_key] = {
            "kvecs": kvecs.detach().cpu(),
        }

    torch.save(cache, out_name)
    print(f"Cache saved to {out_name}")
    
    return out_name, len(cache)


def main():
    # cp_dimer_train = "../custom_dataset/dimer_datasets/vama_updated_dimer_CP_4_train.xyz"
    # cp_dimer_test = "../custom_dataset/dimer_datasets/vama_updated_dimer_CP_4_test.xyz"
    # kvec_output_path = "../custom_dataset/dimer_datasets/kvec_cache.pt"

    bulk_water_train = "../custom_dataset/bc_water_dataset/train_bulk_water.xyz"
    bulk_water_test = "../custom_dataset/bc_water_dataset/test_bulk_water.xyz"
    kvec_output_path = "../custom_dataset/bc_water_dataset/kvec_cache_bulk_water.pt"

    file_list = [bulk_water_train, bulk_water_test]

    all_cells_found = []
    for file in file_list:
        all_cells_found.append(update_unique_boxes(file, all_cells=None, tol=1e-30))
    
    all_cells_found = torch.cat(all_cells_found, dim=0)
    print("Number of unique cells:", all_cells_found.shape)

    cache_path, num_entries = build_cache(
        all_cells=all_cells_found,
        dl=10,
        sigma=5,
        device="cpu",
        out_name=kvec_output_path,
    )
    print(f"Cached k-vectors for {num_entries} structures to {cache_path}")
    
    loaded_kvec_cache = torch.load(kvec_output_path)

    atoms_list = read(bulk_water_train, index=":")
    print(f"XYZ file has {len(atoms_list)} frames")
    
    for i in range(len(atoms_list)):
        atoms = atoms_list[i]
        box = torch.tensor(atoms.cell.lengths(), dtype=torch.float64)
        box_key = tuple(round(x, 8) for x in box.tolist())
        
        print(f"\nFrame {i+1}:")
        print(f"  Box dimensions: {box_key}")
        
        if box_key in loaded_kvec_cache:
            kvecs = loaded_kvec_cache[box_key]['kvecs']
            print(f"  Found in cache: kvecs shape = {kvecs.shape}")
            print(f"  First few kvecs: {kvecs[:3]}")
        else:
            print(f"  NOT found in cache!")

            min_diff = float('inf')
            closest_key = None
            for cache_key in loaded_kvec_cache.keys():
                diff = sum((a - b)**2 for a, b in zip(box_key, cache_key))**0.5
                if diff < min_diff:
                    min_diff = diff
                    closest_key = cache_key
            print(f"  Closest match: {closest_key} (diff = {min_diff:.6f})")

if __name__ == "__main__":
    main()
