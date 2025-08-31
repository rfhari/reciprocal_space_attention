import os
import glob
import argparse
from typing import List, Tuple, Dict

import torch
import numpy as np
from tqdm import tqdm
from ase.io import read  

from k_frequencies import EwaldPotential


def _read_cell_and_pos(xyz_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    print(f"Reading XYZ file: {xyz_path}")
    atoms_list = read(xyz_path, index=":")
    if not isinstance(atoms_list, list):      
        atoms_list = [atoms_list]

    frames: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for i, atoms in enumerate(atoms_list):
        print(f"Processing frame {i} with {atoms} atoms, {atoms.cell.lengths()}")

        # if not atoms.cell.diagonal():
        #     raise ValueError(
        #         f"Non-orthogonal cell in frame {i} of '{xyz_path}'. "
        #         "Current k-grid routine assumes diagonal cells."
        #     )

        cell_len = torch.tensor(atoms.cell.lengths(), dtype=torch.float32)   # (3,)
        pos      = torch.tensor(atoms.get_positions(), dtype=torch.float32)  # (N,3)
        frames.append((pos, cell_len))

    return frames


def _collect_files(root: str, patterns: List[str]) -> List[str]:
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(root, p), recursive=True))
    files = sorted(set(files))
    return files


@torch.no_grad()
def build_cache(
    data_dir: str,
    patterns: List[str],
    dl: float,
    sigma: float,
    device: str = "cpu",
    out_name: str = "kvec_cache.pt",
) -> Tuple[str, int]:

    file_list = patterns # collect_files(data_dir, patterns)
    print(f"Found {file_list} files matching patterns: {patterns}")

    if len(file_list) == 0:
        raise RuntimeError(
            f"No files matched under '{data_dir}' with patterns={patterns}"
        )

    ewald = EwaldPotential(dl=dl, sigma=sigma).to(device)
    cache: Dict[str, Dict[str, torch.Tensor]] = {}

    for path in tqdm(file_list, desc="Computing k-vectors"):
        rel_key = os.path.relpath(path, data_dir)

        for idx, (pos, cell) in enumerate(_read_cell_and_pos(path)):
            pos = pos.to(device)
            cell = cell.to(device)

            kvecs, factor = ewald(pos, cell)

            cache[rel_key] = {
                "kvecs": kvecs.detach().cpu(),
            }

    out_path = os.path.join(data_dir, out_name)
    torch.save({"dl": float(dl), "sigma": float(sigma), "cache": cache}, out_path)
    return out_path, len(cache)

def main():
    out_path, n = build_cache(
        data_dir="./",
        patterns=["train_bulk_water.xyz"],
        dl=10,
        sigma=5,
        device="cpu",
        out_name="water_kvec_cached.pt",
    )
    print(f"Cached k-vectors for {n} structures → {out_path}")


if __name__ == "__main__":
    main()
