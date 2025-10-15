import torch, math
from torch import nn

def _reciprocal(cell: torch.Tensor) -> torch.Tensor:
    return 2.0 * math.pi * torch.inverse(cell).T  # (3,3)

class EwaldPotentialTriclinic(nn.Module):
    """
    Triclinic reciprocal grid with Gaussian (Ewald-like) damping and
    automatic k-selection by cumulative mass.

    Parameters
    ----------
    dl : fallback real-space grid resolution (Å) to set a coarse k-sphere
         if auto_cut is False.
    sigma : Gaussian width (Å). If auto_sigma=True, this is ignored.
    auto_sigma : choose sigma from (r_cut, eps_real).
    eps_real : target real-space split tolerance (for sigma).
    auto_cut : choose k_max from (sigma, eps_k).
    eps_k    : tolerance for reciprocal-space tail (for k_max).
    eps_mass : cumulative-mass coverage; keep k until cumulative_w >= 1 - eps_mass.
    normalize_weights : if True, return w_k normalized to sum 1 (default).
    """
    def __init__(self,
                 dl: float = 10.0,
                 sigma: float = 5.0,
                 auto_sigma: bool = True,
                 eps_real: float = 1e-3,
                 auto_cut: bool = True,
                 eps_k: float = 1e-4,
                 eps_mass: float = 1e-3,
                 normalize_weights: bool = True):
        super().__init__()
        self.dl = dl
        self._sigma_user = sigma
        self.auto_sigma = auto_sigma
        self.eps_real = eps_real
        self.auto_cut = auto_cut
        self.eps_k = eps_k
        self.eps_mass = eps_mass
        self.normalize_weights = normalize_weights

        self.two_pi = 2.0 * math.pi

    @torch.no_grad()
    def forward(self,
                r_cart: torch.Tensor,      # (N,3) 
                cell:   torch.Tensor,      # (3,3)
                r_cut:  float | None = None  
               ):
        device, dtype = cell.device, cell.dtype
        recip = _reciprocal(cell)                 # (3,3)
        b_len = recip.norm(dim=1)                 # |b1|,|b2|,|b3|

        # -- decide sigma --
        if self.auto_sigma:
            assert r_cut is not None and r_cut > 0.0, \
                "auto_sigma=True requires r_cut (SR cutoff)."
            # alpha = sqrt(-ln eps_real) / r_cut
            alpha = math.sqrt(-math.log(self.eps_real)) / float(r_cut)
            sigma = 1.0 / (math.sqrt(2.0) * alpha)
        else:
            sigma = float(self._sigma_user)

        sigma_sq_half = (sigma * sigma) / 2.0

        # -- decide k_max --
        if self.auto_cut:
            # k_max = 2 alpha sqrt(-ln eps_k)
            alpha = 1.0 / (math.sqrt(2.0) * sigma)
            k_max = 2.0 * alpha * math.sqrt(-math.log(self.eps_k))
        else:
            # fallback to sphere implied by dl
            k_max = (self.two_pi / float(self.dl))

        # integer bounds along each reciprocal basis
        n_max = torch.ceil(torch.tensor(k_max, device=device, dtype=dtype) / b_len).to(torch.long)
        
        # currently only considering positive k-vectors, but can be symmetrically extended to include negative k-vectors
        nx = torch.arange(0, n_max[0] + 1, device=device) 
        ny = torch.arange(0, n_max[1] + 1, device=device) 
        nz = torch.arange(0, n_max[2] + 1, device=device) 
        nx_m, ny_m, nz_m = torch.meshgrid(nx, ny, nz, indexing="ij")
        n_vec = torch.stack((nx_m, ny_m, nz_m), dim=-1).reshape(-1, 3).to(dtype)  # (P,3)

        # k = n @ recip, remove k=0
        kvec = n_vec @ recip                                # (P,3)
        k_sq = (kvec * kvec).sum(dim=1)                     # (P,)
        mask = k_sq > 0
        kvec = kvec[mask]
        k_sq = k_sq[mask]

        # raw importance weights: w_k = exp(-\sigma^2 k^2 / 2) / k^2
        w = torch.exp(-sigma_sq_half * k_sq) / k_sq         

        # sort by descending importance and keep until cumulative mass >= 1 - eps_mass
        w_sorted, idx = torch.sort(w, descending=True)
        cum = torch.cumsum(w_sorted, dim=0)
        total = cum[-1]
        if total <= 0:
            # degenerate case
            sel = idx[:1]
        else:
            cutoff_mass = (1.0 - self.eps_mass) * total
            m_keep = int(torch.searchsorted(cum, torch.tensor(cutoff_mass, device=device)).item()) + 1
            sel = idx[:m_keep]

        kvec_sel = kvec[sel]   # (M,3)
        w_sel    = w[sel]      # (M,)

        if self.normalize_weights:
            w_sum = w_sel.sum()
            if w_sum > 0:
                w_sel = w_sel / w_sum
        
        return kvec_sel, w_sel  # (M,3), (M,)

