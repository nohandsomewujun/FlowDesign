import torch
import torch.nn as nn

from diffab.modules.common.geometry import construct_3d_basis
from diffab.modules.common.so3 import rotation_to_so3vec, so3vec_to_rotation, random_uniform_so3
from diffab.modules.encoders.residue import ResidueEmbedding
from diffab.modules.encoders.pair import PairEmbedding
from diffab.modules.rectified_flow.rectflow_finetune import RectFlowGenerator
from diffab.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom
from ._base import register_model

import pickle

resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}

@register_model('rectflow_seq_only')
class RectFlowFintuneAntibodyDesign(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]
        self.residue_embed = ResidueEmbedding(cfg.res_feat_dim, num_atoms)
        self.pair_embed = PairEmbedding(cfg.pair_feat_dim, num_atoms)


        self.rectflow_seq_only = RectFlowGenerator(
            cfg.res_feat_dim,
            cfg.pair_feat_dim,
            **cfg.diffusion,
        )

    def encode(self, batch, remove_structure, remove_sequence):
        """
        Returns:
            res_feat:   (N, L, res_feat_dim)
            pair_feat:  (N, L, L, pair_feat_dim)
        """
        # This is used throughout embedding and encoding layers
        #   to avoid data leakage.
        context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], 
            ~batch['generate_flag']     # Context means ``not generated''
        )

        structure_mask = context_mask if remove_structure else None
        sequence_mask = context_mask if remove_sequence else None

        res_feat = self.residue_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            fragment_type = batch['fragment_type'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
        )

        pair_feat = self.pair_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
        )

        R = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )

        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA] ### CA only

        template = batch['template']

        templateR = construct_3d_basis(
            template['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            template['pos_heavyatom'][:, :, BBHeavyAtom.C],
            template['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )
        templatep = template['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        return res_feat, pair_feat, R, p, templateR, templatep
    
    def forward(self, batch):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        mask_anchor = batch['anchor_flag']
        res_feat, pair_feat, R_0, p_0, R_template, p_template = self.encode(
            batch,
            remove_structure = self.cfg.get('train_structure', True),
            remove_sequence = self.cfg.get('train_sequence', True)
        )

        s_0 = batch['aa']
        template = batch['template']
        s_template = template['aa']
        mask_template_generate = template['generate_flag']

        template_enable = batch['template_enable']
        template_mask = torch.rand(template_enable.shape, device=template_enable.device) < 0.3
        template_enable = template_enable & template_mask

        pdbid = batch['id']
        
        loss_dict = self.rectflow_seq_only(
            R_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res,
            mask_anchor, R_template, p_template, s_template, 
            mask_template_generate,
            denoise_structure = self.cfg.get('train_structure', True),
            denoise_sequence  = self.cfg.get('train_sequence', True),
            template_enable = template_enable, 
            pdbid=pdbid
        )
        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        batch, 
        sample_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        mask_anchor = batch['anchor_flag']
        res_feat, pair_feat, R_0, p_0, R_template, p_template = self.encode(
            batch,
            remove_structure = sample_opt.get('sample_structure', True),
            remove_sequence = sample_opt.get('sample_sequence', True)
        )
        s_0 = batch['aa']
        template = batch['template']
        s_template = template['aa']
        mask_template_generate = template['generate_flag']

        traj = self.rectflow_seq_only.sample(R_0, p_0, s_0, res_feat,\
                                              pair_feat, mask_generate,\
                                                  mask_res, mask_anchor,\
                                                  template_enable=True,\
                                                  R_template=R_template,\
                                                  p_template=p_template,\
                                                  s_template=s_template,\
                                                  mask_template_generate=mask_template_generate,\
                                                      **sample_opt)
        return traj
