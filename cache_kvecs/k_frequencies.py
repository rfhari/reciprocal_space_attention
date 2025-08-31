import torch
import torch.nn as nn
from itertools import product
from typing import Dict
import numpy as np
from torch.nn import functional as F

# from line_profiler import profile

class EwaldPotential(nn.Module):
    def __init__(self,
                 dl=10.0,    # grid resolution
                 sigma=5.0,  # width of the Gaussian on each atom
                ):
        super().__init__()
        self.dl = dl
        self.sigma = sigma
        self.sigma_sq_half = sigma ** 2 / 2.0
        self.twopi = 2.0 * torch.pi
        self.twopi_sq = self.twopi ** 2
        self.k_sq_max = (self.twopi / self.dl) ** 2
        # self.topk_select = 170 # number of k-vectors to select
        
    def forward(self, r_raw, box):
        device = r_raw.device

        r = r_raw / box

        nk  = torch.clamp((box / self.dl).int(), min=1)            

        kx = torch.arange(-nk[0], nk[0] + 1, device=device)
        ky = torch.arange(-nk[1], nk[1] + 1, device=device)
        kz = torch.arange(-nk[2], nk[2] + 1, device=device)

        kx_term = (kx / box[0]) ** 2
        ky_term = (ky / box[1]) ** 2
        kz_term = (kz / box[2]) ** 2

        kx_sq = kx_term.view(-1, 1, 1)
        ky_sq = ky_term.view(1, -1, 1)
        kz_sq = kz_term.view(1, 1, -1)

        k_sq = self.twopi_sq * (kx_sq + ky_sq + kz_sq)
        mask = (k_sq <= self.k_sq_max) & (k_sq > 0)

        idx_x, idx_y, idx_z = torch.nonzero(mask, as_tuple=True)
        kx_sel = kx[idx_x]      
        ky_sel = ky[idx_y]
        kz_sel = kz[idx_z]

        kvec = torch.stack((kx_sel / box[0],
                            ky_sel / box[1],
                            kz_sel / box[2]), dim=-1)              
        
        kvec_sel = self.twopi * kvec 
        k_sq_sel = k_sq[mask]

        k_sq_check = self.twopi_sq * (
                        (kx_sel / box[0]) ** 2 +
                        (ky_sel / box[1]) ** 2 +
                        (kz_sel / box[2]) ** 2
                    )
        
        assert torch.allclose(k_sq_sel, k_sq_check), "k_sq_sel and k_sq_check do not match"

        volume = box.prod()      # orthogonal box only    
        
        factor = torch.full_like(k_sq_sel, self.twopi, dtype=box.dtype) 
        # factor /= volume                                                
        factor *= torch.exp(-self.sigma_sq_half * k_sq_sel) / k_sq_sel

        # factor_ones = torch.ones_like(factor, dtype=box.dtype)

        print("second Nk:", box, factor.shape, kvec_sel.shape, k_sq_sel.shape, kvec.shape)
        
        # learnable_factor = nn.Parameter(factor_ones, requires_grad=True)
        
        # sorted_idx = torch.argsort(k_sq_sel)
        # topk = sorted_idx[:self.topk_select] 
        # kvec_sel = kvec_sel[topk]
        # factor   = factor[topk]
        # k_sq_sel = k_sq_sel[topk]

        # log_weights = torch.log(factor)
        # factor = F.softmax(log_weights, dim=-1)

        return kvec_sel, factor