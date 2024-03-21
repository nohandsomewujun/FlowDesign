import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from diffab.modules.common.layers import clampped_one_hot
from diffab.modules.common.so3 import ApproxAngularDistribution, random_normal_so3, random_uniform_so3, so3vec_to_rotation, rotation_to_so3vec, rotation_to_quaternion
from diffab.modules.common.geometry import euler_angles_to_matrix, matrix_to_euler_angles

class AminoacidCategoricalTransition(nn.Module):

    def __init__(self, num_steps, num_classes=20):
        super().__init__()
        self.num_classes = num_classes

    def interpolate(self, x_0, mask_generate, t, mask_template_generate=None, fix_zero=False, x_template=None, template_enable=False):
        """
        Args:
            x_0: (N, L, 20)
            mask_generate: (N, L)
            t: (N, )
        Returns:

        """
        N, L = mask_generate.size()
        x_template = torch.masked_select(x_template, torch.logical_and(mask_template_generate[..., None], template_enable[..., None, None]))
        s_init_template = x_0.clone().masked_scatter(torch.logical_and(mask_generate[..., None], template_enable[..., None, None]), x_template)
        s_init_template = s_init_template + torch.randn_like(x_0, device=x_0.device)
        s_init_template = torch.where(mask_generate[..., None], s_init_template, x_0)

        s_init = torch.randn_like(x_0, device=x_0.device)
        s_init = torch.where(mask_generate[..., None], s_init, x_0)

        s_init = torch.where(template_enable[..., None, None], s_init_template, s_init)
        # linear 
        s_interp = t[:, None, None]*x_0 + (1.-t[:, None, None])*s_init
    
        return s_interp, s_init

class PositionTransition(nn.Module):

    def __init__(self, num_steps):
        super().__init__()

    def interpolate(self, p_0, mask_generate, t, mask_template_generate=None, p_template=None, template_enable=False):
        """
        Args:
            p_0:    (N, L, 3)
            mask_generate:  (N, L)
            t:      (N,)
        """
        N, L = mask_generate.size()

        p_template = torch.masked_select(p_template, torch.logical_and(mask_template_generate[..., None], template_enable[..., None, None]))
        p_init_template = p_0.clone().masked_scatter(torch.logical_and(mask_generate[..., None].expand_as(p_0), template_enable[..., None, None]), p_template)
        e_rand = torch.randn_like(p_0)
        p_init_template = e_rand + p_init_template
        p_init_template = torch.where(mask_generate[..., None].expand_as(p_0), p_init_template, p_0)

        aa_mask = p_0.norm(dim=-1)
        aa_mask = ~(aa_mask==0)
        context_mask = torch.logical_and(aa_mask, ~mask_generate)
        p_avg = (p_0*context_mask[:, :, None]).sum(dim=1) / context_mask.sum(dim=1)[:, None]
        e_rand = torch.randn_like(p_0)
        p_init = e_rand + p_avg.detach().clone()[:, None, :]
        p_init = torch.where(mask_generate[..., None].expand_as(p_0), p_init, p_0)

        p_init = torch.where(template_enable[..., None, None], p_init_template, p_init)

        p_interp = t[:, None, None] * p_0 + (1.-t[:, None, None])*p_init
        return p_interp, p_init
    
class QuaternionTransition(nn.Module):
    
    def __init__(self, num_steps) -> None:
        super().__init__()
    
    def interpolate(self, q_0, mask_generate, t, mask_template_generate=None, template_enable=False, q_template=None, pdbid=None):
        """
        Args:
            q_0:    (N, L, 4)
            mask_generate:  (N, L)
            t:      (N,)
        """
        N, L = mask_generate.size()
        q_template = torch.masked_select(q_template, torch.logical_and(mask_template_generate[..., None], template_enable[..., None, None]))

        try:
            q_init_template = q_0.clone().masked_scatter(torch.logical_and(mask_generate[..., None].expand_as(q_0), template_enable[..., None, None]), q_template)
        except:
            print(pdbid)
            raise TypeError
        
        v_init = random_uniform_so3([N, L], device=q_0.device)
        R_init = so3vec_to_rotation(v_init)
        q_init = rotation_to_quaternion(R_init)
        q_init = torch.where(mask_generate[..., None].expand_as(q_0), q_init, q_0)

        q_init = torch.where(template_enable[..., None, None], q_init_template, q_init)

        q_init_normalized = q_init / (torch.norm(q_init, dim=-1, keepdim=True)+1e-8)
        q_0_normalized = q_0 / (torch.norm(q_0, dim=-1, keepdim=True)+1e-8)

        cos_theta = torch.sum(q_init_normalized*q_0_normalized, dim=-1)
        cos_theta = cos_theta.clamp(-1, 1)
        
        if torch.is_grad_enabled():
            min_cos = -0.999
        else:
            min_cos = -1
        cos_theta = cos_theta.clamp_min(min=min_cos)
        theta = torch.arccos(cos_theta) # (N, L)
        
        alpha = torch.sin((1.-t[:, None])*theta) / (torch.sin(theta)+1e-8) # (N, L)
        beta = torch.sin(t[:, None]*theta) / (torch.sin(theta)+1e-8)
        q_interp = alpha[:, :, None]*q_init + beta[:, :, None]*q_0
        q_interp = torch.where(mask_generate[..., None].expand_as(q_0), q_interp, q_0)
        
        return q_interp, q_init        

class EulerTrans(nn.Module):

    def __init__(self, num_steps) -> None:
        super().__init__()
    
    def interpolate(self, e_0, mask_generate, t):
        """
        Args:
            e_0:    (N, L, 3)
        """
        N, L = mask_generate.size()
        v_init = random_uniform_so3([N, L], device=e_0.device)
        R_init = so3vec_to_rotation(v_init)
        e_init = matrix_to_euler_angles(R_init, "XYZ")
        e_init = torch.where(mask_generate[..., None].expand_as(e_0), e_init, e_0)

        e_interp = t[:, None, None] * e_0 + (1.-t[:, None, None])*e_init
        
        return e_interp, e_init


