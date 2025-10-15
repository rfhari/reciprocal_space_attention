import math, torch
from torch import nn, einsum
from itertools import product
from typing import Dict
import numpy as np
from .k_frequencies_triclinic import  EwaldPotentialTriclinic
from mace.tools.scatter import scatter_sum
from pathlib import Path

class ReciprocalSpaceAttention(nn.Module): 
    def __init__(self, node_irreps, r_max: float,
                 hidden: int = 64, lebedev_M: int = 6):
        super().__init__()

        self.H = int(hidden) 
        self.qkv = nn.Linear(hidden, 3*hidden, bias=False)
        self.scale_q = 1 / math.sqrt(self.H)
        self.norm   = nn.LayerNorm(hidden)
        self.alpha  = nn.Parameter(torch.tensor(0.1))
        self.act    = nn.SiLU()   # SiLU

        self.kspace_freq = EwaldPotentialTriclinic(
            auto_sigma=True,   eps_real=1e-3,
            auto_cut=True,     eps_k=1e-4,
            eps_mass=1e-3,
            normalize_weights=True
        )
        self.r_cut = r_max   # use your SR cutoff as r_c for auto-sigma

    # rotary positional encoding ------------------------------------------------
    def _rope(self, h:torch.Tensor, pos:torch.Tensor,  cell:torch.Tensor) -> torch.Tensor:
        a, b = h[..., 0::2], h[..., 1::2]                    # (N,H/2)

        k_vecs, w_k = self.kspace_freq(pos, cell, r_cut=self.r_cut)  # (M,3),(M,)
        phase = pos @ k_vecs.T                       # (N,M)
        phase = phase[...,None]                               # (N,M,H/2)
        phase = phase.permute(1,0,2)                          # (M,N,H/2)

        cos, sin = phase.cos(), phase.sin()
        rot_a =  a.unsqueeze(0)*cos - b.unsqueeze(0)*sin
        rot_b =  a.unsqueeze(0)*sin + b.unsqueeze(0)*cos

        return torch.cat([rot_a, rot_b], dim=-1), w_k        # (M,N,H), (M,)

    # Graphwise rotary positional encoding iteratively ------------------------------------------------
    def _rope_graphwise(self,
                        h:    torch.Tensor,   # (N, H)
                        pos:  torch.Tensor,   # (N, D)
                        cell: torch.Tensor,
                        batch: torch.Tensor     
                    ) -> torch.Tensor:

        rot_blocks, w_blocks, M_sizes = [], [], []
        unique_graphs = torch.unique(batch)   # (G,)
        for g in unique_graphs:         
            idx = (batch == g)
            h_g   = h[idx]                   # (N_g, H)
            pos_g = pos[idx]                 # (N_g, 3)
            cell_g = cell[g]
            rot_g, w_g = self._rope(h_g, pos_g, cell_g)   # (M_g,N_g,H), (M_g,)
            rot_blocks.append(rot_g)
            w_blocks.append(w_g)
            M_sizes.append(rot_g.shape[0])

        M_max = max(M_sizes)    # ToDo: optimize biggest grid in this batch
        rot_pad, w_pad = [], []
        for rot_g, w_g in zip(rot_blocks, w_blocks):
            pad_M = M_max - rot_g.shape[0]
            if pad_M > 0:
                rot_g = torch.cat([rot_g,
                                   torch.zeros(pad_M, *rot_g.shape[1:],
                                               device=rot_g.device, dtype=rot_g.dtype)], dim=0)
                w_g   = torch.cat([w_g,
                                   torch.zeros(pad_M, device=w_g.device, dtype=w_g.dtype)], dim=0)
            rot_pad.append(rot_g);  w_pad.append(w_g)

        rot_all = torch.cat(rot_pad, dim=1)        # (M_max, N, H)
        w_all   = torch.stack(w_pad, dim=1)        # (M_max, G)
        return rot_all, w_all
    
    def forward(self, data: Dict[str, torch.Tensor], node_feat:torch.Tensor, kweights:torch.Tensor) -> torch.Tensor:
        pos = data['positions'].to(node_feat.dtype)                    # (N,3)

        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch = data["batch"]

        # box = data['cell'].view(-1, 3, 3).diagonal(dim1=-2, dim2=-1) 

        cell = data['cell'].view(-1, 3, 3)                            # (G,3,3)

        q, k, v = self.qkv(node_feat).chunk(3, dim=-1)         # (N, H) each
        q, k    = self.act(q), self.act(k)       # ψ
        
        (q_rot, w_q), (k_rot, w_k) = (
        self._rope_graphwise(x, pos, cell, batch) for x in (q, k)
        )  # q_rot,k_rot: (M,N,H); w_q,w_k: (M,G)
                
        G = int(batch[-1]) + 1
        w = w_q    # (M,G)
        w_node = w[:, batch]             # (M,N)

        if not hasattr(self, "scale_q") or self.scale_q is None:
            self.scale_q = 1.0 / math.sqrt(q_rot.shape[-1])
        q = q_rot * self.scale_q

        # K x V per node
        kv_node = k_rot.unsqueeze(-1) * v.unsqueeze(-2)   # (M,N,H,H)

        kv_graph = scatter_sum(kv_node, batch, dim=1, dim_size=G)  # (M,G,H,H)
        kv_node  = kv_graph[:, batch]                              # (M,N,H,H)

        beta = (q.unsqueeze(-1) * kv_node).sum(-2)                 # (M,N,H)

        # ---- weighted sum across k, using w_node ----
        update = (w_node[..., None] * beta).sum(0)                 # (N,H)

        return update