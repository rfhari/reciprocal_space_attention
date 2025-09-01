from __future__ import annotations
from e3nn import nn, o3
from mace.modules.wrapper_ops import Linear
import math, torch
from torch import nn, einsum
from itertools import product
from typing import Dict
import numpy as np
from .k_frequencies import EwaldPotential
from mace.tools.scatter import scatter_sum
from pathlib import Path

class RSA_MACE(nn.Module): 
    def __init__(self, 
                 node_irreps: o3.Irreps,
                 node_irreps_after_interaction: o3.Irreps,
                ):
        super().__init__()
        self.hidden = node_irreps[0][0]
        self.linear_readout = Linear(node_irreps, o3.Irreps(f"{self.hidden}x0e"))
        self.qkv = nn.Linear(self.hidden, 3*self.hidden, bias=False)
        self.scale_q = 1 / math.sqrt(self.hidden)
        self.norm   = nn.LayerNorm(self.hidden)
        self.act    = nn.SiLU()   
        self.kspace_freq = EwaldPotential()
        self.tp = o3.FullyConnectedTensorProduct(
            node_irreps_after_interaction, o3.Irreps(f"{self.hidden}x0e"), node_irreps_after_interaction,
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cached_kvecs = torch.load("./custom_dataset/bc_water_dataset/kvec_cache_bulk_water.pt", map_location=device)
        self.topk = 6

    def _rope(self, 
              h:torch.Tensor, 
              pos:torch.Tensor, 
              box:torch.Tensor) -> torch.Tensor:

        box_key = tuple(round(x, 8) for x in box.double().tolist())

        try:
            cached_kvecs = self.cached_kvecs[box_key]["kvecs"]
        except:
            raise KeyError(f"Box key {box_key} not found in cache")
  
        mu, factors = cached_kvecs[:self.topk], torch.ones_like(cached_kvecs[:self.topk,0])  # (M,3)        
        # mu, factors = self.kspace_freq(pos, box)                    # (N_k,3)

        phase = torch.matmul(pos, mu.T)                        # (N,M)
        phase = phase[...,None]                               # (N,N_k,H/2)
        phase = phase.permute(1,0,2)                          # (N_k,N,H/2)

        cos, sin = phase.cos(), phase.sin()

        a, b = h[..., 0::2], h[..., 1::2]                    # (N,H/2)
        a0, b0 = a.unsqueeze(0), b.unsqueeze(0) 
        rot_a =  a0*cos - b0*sin
        rot_b =  a0*sin + b0*cos
        
        # h = h.unsqueeze(0)
        # y = torch.stack((-h[..., 1::2], h[..., ::2]), dim=-1).reshape_as(h)
        # out = h * cos + y * sin
        # print(f"h: {h.shape}, y: {y.shape}, out: {out.shape}")

        # return out, factors             # (M,N,H)

        return torch.cat([rot_a, rot_b], dim=-1), factors             # (N_k,N,H)

    def _rope_graphwise(self,
                        h:    torch.Tensor,   # (N, H)
                        pos:  torch.Tensor,   # (N, D)
                        box:  torch.Tensor,   # (G, 3)
                        batch: torch.Tensor     
                    ) -> torch.Tensor:

        all_weights, out, nodes_per_graph = [], [], []
        unique_graphs = torch.unique(batch)   # (G,)
        for g in unique_graphs:         
            idx = (batch == g)
            h_g   = h[idx]                   # (N_g, H)
            pos_g = pos[idx]                 # (N_g, 3)
            box_g = box[g]                   # (3,)   
            out_g, weights = self._rope(h_g, pos_g, box_g)   # (N_k, N_g, H)
            out.append(out_g)
            all_weights.append(weights)  
            nodes_per_graph.append(int(idx.sum()))   

        out = torch.cat(out, dim=1)         # (N_k, N, H)
        all_weights = torch.cat(all_weights).view(-1, len(unique_graphs))
        nodes_per_graph = torch.tensor(nodes_per_graph, dtype=torch.int64, device=out.device)  # (G,)         

        return out, all_weights, nodes_per_graph                # (N_k, N, H)

    def forward(self, data: Dict[str, torch.Tensor], node_feat:torch.Tensor, node_feat_sr:torch.Tensor) -> torch.Tensor:
        scalars = self.linear_readout(node_feat)

        pos = data['positions'].to(node_feat.dtype)                    # (N,3)

        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch = data["batch"]

        box = data['cell'].view(-1, 3, 3).diagonal(dim1=-2, dim2=-1)
        q, k, v = self.qkv(scalars).chunk(3, dim=-1)         # (N, H) each
        q, k    = self.act(q), self.act(k)             
        (q, weights_q, nodes_per_graph_q), (k, weights_k, _) = (self._rope_graphwise(x, pos, box, batch) for x in (q, k))    # (N_k, N, H)
        q = q * self.scale_q
        kv_node = k.unsqueeze(-1) * v.unsqueeze(-2)    # (N_k,N,H,H)

        G  = int(batch[-1]) + 1          # number of graphs
        kv_graph = scatter_sum(          # (N_k,G,H,H)
            kv_node, batch, dim=1, dim_size=G)

        kv_node = kv_graph[:, batch]                        # (N_k,N,H,H)

        beta = (q.unsqueeze(-1) * kv_node).sum(-2)          # (N_k,N,H)
        weights_q = weights_q.repeat_interleave(nodes_per_graph_q, dim=1)   

        update = (beta).sum(0)   # (N,S)

        node_nl = self.tp(node_feat_sr, update)   # (N,F)
        
        return node_nl        