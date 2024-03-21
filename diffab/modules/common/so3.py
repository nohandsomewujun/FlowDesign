import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import quaternion_to_rotation_matrix, matrix_to_quaternion


def log_rotation(R):
    trace = R[..., range(3), range(3)].sum(-1)
    if torch.is_grad_enabled():
        # The derivative of acos at -1.0 is -inf, so to stablize the gradient, we use -0.9999
        min_cos = -0.999
    else:
        min_cos = -1.0
    cos_theta = ( (trace-1) / 2 ).clamp_min(min=min_cos)
    sin_theta = torch.sqrt(1 - cos_theta**2)
    theta = torch.acos(cos_theta)
    coef = ((theta+1e-8)/(2*sin_theta+2e-8))[..., None, None]
    logR = coef * (R - R.transpose(-1, -2))
    return logR


def skewsym_to_so3vec(S):
    x = S[..., 1, 2]
    y = S[..., 2, 0]
    z = S[..., 0, 1]
    w = torch.stack([x,y,z], dim=-1)
    return w


def so3vec_to_skewsym(w):
    x, y, z = torch.unbind(w, dim=-1)
    o = torch.zeros_like(x)
    S = torch.stack([
        o, z, -y,
        -z, o, x,
        y, -x, o,
    ], dim=-1).reshape(w.shape[:-1] + (3, 3))
    return S


def exp_skewsym(S):
    x = torch.linalg.norm(skewsym_to_so3vec(S), dim=-1)
    I = torch.eye(3).to(S).view([1 for _ in range(S.dim()-2)] + [3, 3])
    
    sinx, cosx = torch.sin(x), torch.cos(x)
    b = (sinx + 1e-8) / (x + 1e-8)
    c = (1-cosx + 1e-8) / (x**2 + 2e-8)  # lim_{x->0} (1-cosx)/(x^2) = 0.5

    S2 = S @ S
    return I + b[..., None, None]*S + c[..., None, None]*S2


def so3vec_to_rotation(w):
    return exp_skewsym(so3vec_to_skewsym(w))


def rotation_to_so3vec(R):
    logR = log_rotation(R)
    w = skewsym_to_so3vec(logR)
    return w

def rotation_to_quaternion(R):
    return matrix_to_quaternion(R)


def random_uniform_so3(size, device='cpu'):
    q = F.normalize(torch.randn(list(size)+[4,], device=device), dim=-1)    # (..., 4)
    return rotation_to_so3vec(quaternion_to_rotation_matrix(q))

def compute_global_R_from_standard_to_vec(vec):
    '''
        XCLIU: compute the global R to make the generation eqvariant
        the rotation matrix R is computed as the rotation matrix from (1,1,1) to unit vector vec
        vec : (B * 3)
    '''
    vec = F.normalize(vec, dim=-1)
    vec_std = torch.ones_like(vec, device=vec.device)
    vec_std = F.normalize(vec_std, dim=-1)
    v = torch.cross(vec_std, vec, dim=-1)
    c = (vec*vec_std).sum(dim=-1)
    R = torch.eye(3, device=vec.device).unsqueeze(0).repeat(vec.shape[0], 1, 1)
    skew = torch.zeros_like(R, device=R.device)
    skew[:, 1, 0] = v[:, 2]
    skew[:, 0, 1] = -v[:, 2]
    skew[:, 2, 0] = -v[:, 1]
    skew[:, 0, 2] = v[:, 1]
    skew[:, 2, 1] = v[:, 0]
    skew[:, 1, 2] = -v[:, 0]
    
    R = R + skew + torch.matmul(skew, skew) / (1. + c[:, None, None])

    return R

def compute_global_R_from_vec_to_standard(vec):
    '''
        XCLIU: compute the global R to make the generation eqvariant
        the rotation matrix R is computed as the rotation matrix from unit vector vec to (1,1,1)
        vec : (B * 3)
    '''
    vec = F.normalize(vec, dim=-1)
    vec_std = torch.ones_like(vec, device=vec.device)
    vec_std = F.normalize(vec_std, dim=-1)
    v = torch.cross(vec, vec_std, dim=-1)
    c = (vec*vec_std).sum(dim=-1)
    R = torch.eye(3, device=vec.device).unsqueeze(0).repeat(vec.shape[0], 1, 1)
    skew = torch.zeros_like(R, device=R.device)
    skew[:, 1, 0] = v[:, 2]
    skew[:, 0, 1] = -v[:, 2]
    skew[:, 2, 0] = -v[:, 1]
    skew[:, 0, 2] = v[:, 1]
    skew[:, 2, 1] = v[:, 0]
    skew[:, 1, 2] = -v[:, 0]
    
    R = R + skew + torch.matmul(skew, skew) / (1. + c[:, None, None])

    return R


def grassmann_product(q_1: torch.Tensor, q_2: torch.Tensor):
    N, L = q_1.shape[0], q_1.shape[1]
    G = torch.zeros_like(q_1)
    G[:, :, 0] = q_1[:, :, 0]*q_2[:, :, 0]-torch.sum(q_1[:, :, 1:]*q_2[:, :, 1:], dim=-1)
    G[:, :, 1:] = q_1[..., 0].unsqueeze(-1)*q_2[..., 1:]+q_2[..., 0].unsqueeze(-1)*q_1[..., 1:]+torch.cross(q_1[..., 1:], q_2[..., 1:], dim=-1)
    return G

def quaternion_diff(q_1_normalized: torch.Tensor, q_2_normalized: torch.Tensor):
    """
    Args:
        q1: (N, L, 4)
        q2: (N, L, 4)
    """
    N, L = q_2_normalized.shape[0], q_2_normalized.shape[1]
    q_2_star = torch.zeros((N, L, 4), dtype=q_2_normalized.dtype, device=q_2_normalized.device)
    q_2_star[:, :, 0] = q_2_normalized[:, :, 0]
    q_2_star[:, :, 1:] = -q_2_normalized[:, :, 1:]
    G = grassmann_product(q_1_normalized, q_2_star)
    return G



class ApproxAngularDistribution(nn.Module):

    def __init__(self, stddevs, std_threshold=0.1, num_bins=8192, num_iters=1024):
        super().__init__()
        self.std_threshold = std_threshold
        self.num_bins = num_bins
        self.num_iters = num_iters
        self.register_buffer('stddevs', torch.FloatTensor(stddevs))
        self.register_buffer('approx_flag', self.stddevs <= std_threshold)
        self._precompute_histograms()

    @staticmethod
    def _pdf(x, e, L):
        """
        Args:
            x:  (N, )
            e:  Float
            L:  Integer
        """
        x = x[:, None]  # (N, *)
        c = ((1 - torch.cos(x)) / math.pi)  # (N, *)
        l = torch.arange(0, L)[None, :]  # (*, L)
        a = (2*l+1) * torch.exp(-l*(l+1)*(e**2))  # (*, L)
        b = (torch.sin( (l+0.5)* x ) + 1e-6) / (torch.sin( x / 2 ) + 1e-6) # (N, L)
        
        f = (c * a * b).sum(dim=1)
        return f

    def _precompute_histograms(self):
        X, Y = [], []
        for std in self.stddevs:
            std = std.item()
            x = torch.linspace(0, math.pi, self.num_bins)   # (n_bins,)
            y = self._pdf(x, std, self.num_iters)    # (n_bins,)
            y = torch.nan_to_num(y).clamp_min(0)
            X.append(x)
            Y.append(y)
        self.register_buffer('X', torch.stack(X, dim=0))  # (n_stddevs, n_bins)
        self.register_buffer('Y', torch.stack(Y, dim=0))  # (n_stddevs, n_bins)

    def sample(self, std_idx):
        """
        Args:
            std_idx:  Indices of standard deviation.
        Returns:
            samples:  Angular samples [0, PI), same size as std.
        """
        size = std_idx.size()
        std_idx = std_idx.flatten() # (N,)
        
        # Samples from histogram
        prob = self.Y[std_idx]  # (N, n_bins)
        bin_idx = torch.multinomial(prob[:, :-1], num_samples=1).squeeze(-1)    # (N,)
        bin_start = self.X[std_idx, bin_idx]    # (N,)
        bin_width = self.X[std_idx, bin_idx+1] - self.X[std_idx, bin_idx]
        samples_hist = bin_start + torch.rand_like(bin_start) * bin_width    # (N,)

        # Samples from Gaussian approximation
        mean_gaussian = self.stddevs[std_idx]*2
        std_gaussian = self.stddevs[std_idx]
        samples_gaussian = mean_gaussian + torch.randn_like(mean_gaussian) * std_gaussian
        samples_gaussian = samples_gaussian.abs() % math.pi

        # Choose from histogram or Gaussian
        gaussian_flag = self.approx_flag[std_idx]
        samples = torch.where(gaussian_flag, samples_gaussian, samples_hist)

        return samples.reshape(size)


def random_normal_so3(std_idx, angular_distrib, device='cpu'):
    size = std_idx.size()
    u = F.normalize(torch.randn(list(size)+[3,], device=device), dim=-1)
    theta = angular_distrib.sample(std_idx)
    w = u * theta[..., None]
    return w

