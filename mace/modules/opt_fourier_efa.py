from __future__ import annotations

import math, torch
from torch import nn, einsum
from itertools import product
from typing import Dict
import numpy as np
# from .k_frequencies import EwaldPotential
from line_profiler import profile
from mace.tools.scatter import scatter_sum
from pathlib import Path
from e3nn.o3 import Irreps

# ---------- helper: slice that contains ONLY the 0e channels --------------
# @profile
def scalar_slice(irreps: Irreps) -> slice:
    """
    Returns a slice that grabs *all* 0e channels at the front of `irreps`,
    assuming they are stored first (default MACE ordering).
    Works with both new and old e3nn iterators.
    """
    start = 0
    for mul_ir in irreps:
        # new API: _MulIr;  old API: Irrep
        ir   = mul_ir.ir if hasattr(mul_ir, "ir") else mul_ir
        mul  = mul_ir.mul if hasattr(mul_ir, "mul") else 1

        if ir.l != 0 or ir.p != 1:     # not a scalar-even channel
            break
        start += mul * ir.dim          # each copy contributes ir.dim dims
    return slice(0, start)             # [:start] are scalars

# @profile
def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class FourierEFA_MACE(nn.Module): 
    def __init__(self, node_irreps, r_max: float,
                 hidden: int):
        super().__init__()
        # self.H = hidden
        # self.scalar_sl = scalar_slice(node_irreps)
        # S = self.scalar_sl.stop                      # #scalar channels
        if hidden is None:                               # ← NEW
            hidden = node_irreps.dim                               # ← NEW
        # assert hidden % 2 == 0, "hidden must be even"    # ← keep existing check

        # assert S % 2 == 0, "scalar channel count must be even (real+imag)."
        
        # self.embed_dim = hidden
        # self.output_dim = 3 * hidden
        # self.qkv = LinearReadoutBlock(o3.Irreps(str(self.embed_dim) + "x0e"), o3.Irreps(self.output_dim + "x0e"))
        self.qkv = nn.Linear(hidden, 3*hidden, bias=False)

        # self.rope_weights = nn.Parameter(torch.ones(58)) # for dimers

        # self.rope_weights = nn.Parameter(torch.ones(58), requires_grad=True) # for water

        self.scale_q = 1 / math.sqrt(hidden)
        self.norm   = nn.LayerNorm(hidden)
        self.alpha  = nn.Parameter(torch.tensor(0.1))
        self.act    = nn.SiLU()   # SiLU
        # self.kspace_freq = EwaldPotential()

    # rotary positional encoding ------------------------------------------------
    # @profile
    # def _rope(self, h:torch.Tensor, pos:torch.Tensor, box:torch.Tensor) -> torch.Tensor:
    #     # h : (N,H) → (M,N,H)
    #     a, b = h[..., 0::2], h[..., 1::2]                    # (N,H/2)
    #     # u, factors = self.kspace_freq(pos, box)                    # (M,3)
        
    #     # if self.rope_weights is None:
    #         # init = torch.ones(u.shape[0], device=h.device, dtype=h.dtype)
    #         # self.rope_weights = torch.nn.Parameter(init, requires_grad=True)
        
    #     phase = torch.matmul(pos, u.T)                        # (N,M)
    #     phase = phase[...,None]                               # (N,M,H/2)
    #     phase = phase.permute(1,0,2)                          # (M,N,H/2)

    #     cos, sin = phase.cos(), phase.sin()
    #     a0, b0 = a.unsqueeze(0), b.unsqueeze(0) 
    #     rot_a =  a0*cos - b0*sin
    #     rot_b =  a0*sin + b0*cos
        
    #     return torch.cat([rot_a, rot_b], dim=-1), factors             # (M,N,H)

    # Graphwise rotary positional encoding iteratively ------------------------------------------------
    @profile
    def _rope_graphwise_old(self,
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
            out_g, weights = self._rope(h_g, pos_g, box_g)   # (M, N_g, H)
            out.append(out_g)
            all_weights.append(weights)  
            nodes_per_graph.append(int(idx.sum()))   
            # all_weights.append(weights)      

        out = torch.cat(out, dim=1)         # (M, N, H)
        all_weights = torch.cat(all_weights).view(-1, len(unique_graphs))
        nodes_per_graph = torch.tensor(nodes_per_graph, dtype=torch.int64, device=out.device)  # (G,)         
        # print('kfreq weights', out.shape, all_weights.shape, nodes_per_graph.shape)

        return out, all_weights, nodes_per_graph                # (M, N, H)
    
    @profile
    def _rope_graphwise(self,
                        h:      torch.Tensor,   # (N, H)
                        pos:    torch.Tensor,   # (N, 3)
                        box:    torch.Tensor,   # (G, 3)
                        batch:  torch.Tensor    # (N,)N
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorised rotary encoding for **all graphs**.
        Returns
            rot_h  : (M, N, H)
            w_g    : (M, G)   – k-space weights per graph
        """
        # --------  build / fetch k-grid  -----------------------------------
        k_vec, w_g = self._build_kgrid(box, dl=10.0, sigma=5.0, topk=6)

        G, M, _ = k_vec.shape
        
        # H_half  = h.shape[-1] // 2

        # --------  cos/sin phase for every node ----------------------------
        box_inv   = 1.0 / box                                           # (G,3)
        # pos_frac  = pos * box_inv[batch]                                # (N,3)

        # phase   (N,M)  = Σ_d  k_vec[batch,m,d] * pos_frac[n,d]
        phase = torch.einsum('nd,nmd->nm', pos, 
                              k_vec[batch])                              # (N,M)

        phase = phase.permute(1,0)                                      # (M,N)

        cos, sin = phase.cos(), phase.sin()                             # (M,N)


        # a, b   = h[..., 0::2], h[..., 1::2]                             # (N,H/2)
        # a0, b0 = a.unsqueeze(0), b.unsqueeze(0)                     # (1,N,H/2)
        cos, sin = cos.unsqueeze(-1), sin.unsqueeze(-1)          # (M,N,1)
        
        # rot_a  = a0*cos - b0*sin
        # rot_b  = a0*sin + b0*cos
        # rot_h  = torch.cat([rot_a, rot_b], dim=-1)                      # (M,N,H)

        w_mg   = w_g.transpose(0,1)                                      # (M,G)
        return rot_h, w_mg

    # forward ------------------------------------------------------------------
    @profile
    def forward(self, data: Dict[str, torch.Tensor], node_feat:torch.Tensor) -> torch.Tensor:
        #sl = self.scalar_sl
        scalars = node_feat # [:, sl]                 # (N,S)

        # --------------------------------  variables  ---------------------------
        pos = data['positions'].to(node_feat.dtype)                    # (N,3)

        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch = data["batch"]

        # --------------------------------  inside forward()  -------------------------
        box = data['cell'].view(-1, 3, 3).diagonal(dim1=-2, dim2=-1)
        q, k, v = self.qkv(scalars).chunk(3, dim=-1)         # (N, H) each
        q, k    = self.act(q), self.act(k)             # ψ
        # (q, weights_q), (k, weights_k) = (self._rope_graphwise(x, pos, box, batch) for x in (q, k))    # (M, N, H)
        q, w_mg = self._rope_graphwise(q, pos, box, batch)
        k, _ = self._rope_graphwise(k, pos, box, batch)
        # v  = self._rope(v, pos)                     # (M, N, H)
        q    = q * self.scale_q
        # print("q:", q.shape, "k:", k.shape, "v:", v.shape, weights_q.shape, nodes_per_graph_q.shape)

        # --- k ⊗ v  (outer product node-wise) --------------------------------
        kv_node = k.unsqueeze(-1) * v.unsqueeze(-2)    # (M,N,H,H)

        # accumulate over the nodes that belong to the same graph
        G  = int(batch[-1]) + 1          # number of graphs
        kv_graph = scatter_sum(          # (M,G,H,H)
            kv_node, batch, dim=1, dim_size=G)

        # broadcast graph-wise result back to every node
        kv_node = kv_graph[:, batch]                        # (M,N,H,H)

        # β_{mn} = ⟨q_{mn}, kv_{mn·}⟩
        beta = (q.unsqueeze(-1) * kv_node).sum(-2)          # (M,N,H)
        # weights_q = weights_q.repeat_interleave(nodes_per_graph_q, dim=1)   
        weights_q = w_mg[:, batch]

        # print("beta:", beta.shape, weights_q)

        update = (weights_q.unsqueeze(-1) * beta).sum(0) 
        # update = (beta).sum(0)   # (N,S)
        # print("update", update.shape)
        # print('kweights:', kweights)

        return update
    
    # ------------------------------------------------------------------
    # inside FourierEFA_MACE
    # ------------------------------------------------------------------
    @profile
    def _build_kgrid(self,
                    box: torch.Tensor,     # (G, 3) – a, b, c in Å
                    dl: float = 10.0,      # grid resolution
                    sigma: float = 5.0,    # Gaussian width
                    topk: int  = 6,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return
            k_vec   : (G, M, 3)   selected k-vectors in Cartesian units [Å⁻¹]
            factor  : (G, M)      positive weights ∑_m factor[g,m] = 1
        identical (up to tie order) to the loop version you used before.
        """
        device = box.device
        G      = box.shape[0]

        two_pi  = 2.0 * torch.pi
        k2_max  = (two_pi / dl) ** 2
        sig2    = sigma * sigma

        # 1) global integer range that covers the *largest* box ------------------
        nk_max = torch.clamp((box / dl).floor().to(torch.int64).max(), min=1)
        k_int  = torch.arange(-nk_max, nk_max + 1, device=device)            # (K,)
        # k_int  = torch.arange(0, nk_max + 1, device=device)            # (K,)

        kx, ky, kz = torch.meshgrid(k_int, k_int, k_int, indexing="ij")      # (K,K,K)
        k_int      = torch.stack((kx, ky, kz), dim=-1).reshape(-1, 3)        # (K³,3)
        k_int      = k_int[(k_int != 0).any(dim=-1)]                         # drop (0,0,0)
        K          = k_int.shape[0]                                          # total ints

        # topk = K

        # 2) scale by each box length -------------------------------------------
        box_inv = 1.0 / box                                                  # (G,3)
        k_cart  = two_pi * (k_int[None, :, :] * box_inv[:, None, :])         # (G,K,3)
        k_sq    = (k_cart ** 2).sum(-1)                                      # (G,K)

        # 3) mask inside k² cut-off ---------------------------------------------
        mask = (k_sq <= k2_max) & (k_sq > 0)                                 # (G,K)
        k_sq.masked_fill_(~mask, 1e10)                                       # big number

        # 4) per-graph top-k (smallest k²) --------------------------------------
        print("k_sq:", k_sq.shape, topk)
        k_sq_top, idx = torch.topk(k_sq, topk, dim=1, largest=False, sorted=True)  # (G,M)
        k_vec = torch.gather(k_cart, 1, idx.unsqueeze(-1).expand(-1, -1, 3))        # (G,M,3)

        # 5) Ewald weight 2π·exp(-σ²k²/2)/k²  + softmax  -------------------------
        factor_raw = two_pi * torch.exp(-0.5 * sig2 * k_sq_top) / k_sq_top          # (G,M)
        factor     = torch.softmax(torch.log(factor_raw), dim=1)                    # (G,M)

        factor_ones = torch.ones_like(factor, dtype=box.dtype)

        return k_vec, factor_ones
            
    # ----------------------------------------------------------------------------- 