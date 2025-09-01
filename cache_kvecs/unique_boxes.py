from typing import Optional
from collections import OrderedDict
import numpy as np
import torch
from ase.io import read

def update_unique_boxes(xyz_path: str, 
                        all_cells: Optional[torch.Tensor] = None,
                        tol: float = 1e-6) -> torch.Tensor:
    atoms_list = read(xyz_path, index=":")
    print(f"Reading XYZ file: {xyz_path} with {len(atoms_list)} frames")

    # currently only for orthorhombic (diagonal) boxes
    new_boxes = torch.tensor(np.asarray([atoms.cell.lengths() for atoms in atoms_list]), dtype=torch.float64)  # (T,3)
    rows = []

    if all_cells is not None and all_cells.numel() > 0:
        for r in all_cells.tolist():
            rows.append(r)

    for r in new_boxes.tolist():
        rows.append(r)

    return torch.tensor(rows, dtype=torch.float64)

cp_dimer_train = "../custom_dataset/dimer_datasets/vama_updated_dimer_CP_4_train.xyz"
cp_dimer_test = "../custom_dataset/dimer_datasets/vama_updated_dimer_CP_4_test.xyz"

bulk_water_train = "../custom_dataset/bc_water_dataset/train_bulk_water.xyz"
bulk_water_test = "../custom_dataset/bc_water_dataset/test_bulk_water.xyz"

all_cells = update_unique_boxes(bulk_water_train, all_cells=None, tol=1e-30)
print("Unique boxes1:", all_cells.shape)

all_cells = update_unique_boxes(bulk_water_test, all_cells=all_cells, tol=1e-30)
print("Unique boxes2:", all_cells.shape)