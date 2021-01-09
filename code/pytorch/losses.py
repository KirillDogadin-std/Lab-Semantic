"""Module that describes loss functions."""
import torch
from torch import nn
from pytorch.utils import l2_normalize


def relation_logistic_loss(phs, prs, pts, nhs, nrs, nts):
    pos_distance = phs + prs - pts
    neg_distance = nhs + nrs - nts
    pos_score = -torch.sum(torch.square(pos_distance), dim=1)
    neg_score = -torch.sum(torch.square(neg_distance), dim=1)
    pos_loss = torch.sum(torch.log(1 + torch.exp(-pos_score)))
    neg_loss = torch.sum(torch.log(1 + torch.exp(neg_score)))
    loss = pos_loss + neg_loss
    return loss


def attribute_logistic_loss(phs, pas, pvs, pws, nhs, nas, nvs, nws):
    pos_distance = phs + pas - pvs
    neg_distance = nhs + nas - nvs
    pos_score = -torch.sum(torch.square(pos_distance), dim=1)
    neg_score = -torch.sum(torch.square(neg_distance), dim=1)
    pos_score = torch.log(1 + torch.exp(-pos_score))
    neg_score = torch.log(1 + torch.exp(neg_score))
    pos_score = torch.multiply(pos_score, pws)
    neg_score = torch.multiply(neg_score, nws)
    pos_loss = torch.sum(pos_score)
    neg_loss = torch.sum(neg_score)
    loss = torch.add(pos_loss, neg_loss)
    return loss


def relation_logistic_loss_wo_negs(phs, prs, pts):
    pos_distance = phs + prs - pts
    pos_score = -torch.sum(torch.square(pos_distance), dim=1)
    loss = torch.sum(torch.log(1 + torch.exp(-pos_score)))
    return loss


def attribute_logistic_loss_wo_negs(phs, pas, pvs):
    pos_distance = phs + pas - pvs
    pos_score = -torch.sum(torch.square(pos_distance), dim=1)
    loss = torch.sum(torch.log(1 + torch.exp(-pos_score)))
    return loss


def logistic_loss_wo_negs(phs, pas, pvs, pws):
    pos_distance = phs + pas - pvs
    pos_score = -torch.sum(torch.square(pos_distance), dim=1)
    pos_score = torch.log(1 + torch.exp(-pos_score))
    pos_score = torch.multiply(pos_score, pws)
    loss = torch.sum(pos_score)
    return loss


def orthogonal_loss(mapping, eye):
    loss = torch.sum(torch.sum(torch.pow(torch.matmul(mapping, mapping.t()) - eye, 2), 1))
    return loss


def space_mapping_loss(view_embeds, shared_embeds, mapping, eye, orthogonal_weight, norm_w=0.0001):
    mapped_ents2 = torch.matmul(view_embeds, mapping)
    mapped_ents2 = l2_normalize(mapped_ents2)
    map_loss = torch.sum(torch.sum(torch.square(shared_embeds - mapped_ents2), 1))
    norm_loss = torch.sum(torch.sum(torch.square(mapping), 1))
    loss = map_loss + orthogonal_weight * orthogonal_loss(mapping, eye) + norm_w * norm_loss
    return loss


def alignment_loss(ents1, ents2):
    distance = ents1 - ents2
    loss = torch.sum(torch.sum(torch.square(distance), dim=1))
    return loss


class MultiKELoss(nn.Module):
    """Descrubes losses that are used in MULTI-KE methodology."""
    def __init__(self, cv_name_weight, cv_weight, orthogonal_weight=2, eye=None):
        super(MultiKELoss, self).__init__()
        self.cv_name_weight = cv_name_weight
        self.cv_weight = cv_weight
        self.orthogonal_weight = orthogonal_weight

        self.cfg = {
            'rv': self.relation_triple_loss,
            'av': self.attribute_triple_loss,
            'ckgrtv': self.cross_kg_relation_triple_loss,
            'ckgatv': self.cross_kg_attribute_triple_loss,
            'ckgrrv': self.cross_kg_relation_reference_loss,
            'ckgarv': self.cross_kg_attribute_reference_loss,
            'cnv': self.cross_name_view_loss
        }
        if eye is not None:
            self.eye = eye
            self.cfg['mv'] = self.mapping_loss

    def relation_triple_loss(self, rel_phs, rel_prs, rel_pts, rel_nhs, rel_nrs, rel_nts):
        loss = relation_logistic_loss(rel_phs, rel_prs, rel_pts, rel_nhs, rel_nrs, rel_nts)
        return loss

    def attribute_triple_loss(self, pos_score, attr_pos_ws):
        pos_score = torch.log(1 + torch.exp(-pos_score))
        pos_score = torch.multiply(pos_score, attr_pos_ws)
        loss = torch.sum(pos_score)
        return loss

    def cross_kg_relation_triple_loss(self, ckge_rel_phs, ckge_rel_prs, ckge_rel_pts):
        loss = 2 * relation_logistic_loss_wo_negs(ckge_rel_phs, ckge_rel_prs, ckge_rel_pts)
        return loss

    def cross_kg_attribute_triple_loss(self, pos_score):
        loss = 2 * torch.sum(torch.log(1 + torch.exp(-pos_score)))
        return loss

    def cross_kg_relation_reference_loss(self, ckgp_rel_phs, ckgp_rel_prs, ckgp_rel_pts, ckgp_rel_pos_ws):
        loss = 2 * logistic_loss_wo_negs(ckgp_rel_phs, ckgp_rel_prs, ckgp_rel_pts, ckgp_rel_pos_ws)
        return loss

    def cross_kg_attribute_reference_loss(self, pos_score, ckga_attr_pos_ws):
        pos_score = torch.log(1 + torch.exp(-pos_score))
        pos_score = torch.multiply(pos_score, ckga_attr_pos_ws)
        loss = torch.sum(pos_score)
        # loss = torch.sum(torch.log(1 + torch.exp(-pos_score)))
        return loss

    def cross_name_view_loss(self, final_cn_phs, cn_hs_names, cr_hs, ca_hs):
        loss = self.cv_name_weight * alignment_loss(final_cn_phs, cn_hs_names)
        loss += alignment_loss(final_cn_phs, cr_hs)
        loss += alignment_loss(final_cn_phs, ca_hs)
        loss = self.cv_weight * loss
        return loss

    def mapping_loss(self, final_ents, nv_ents, rv_ents, av_ents, nv_mapping, rv_mapping, av_mapping):
        nv_space_mapping_loss = space_mapping_loss(nv_ents, final_ents, nv_mapping, self.eye, self.orthogonal_weight)
        rv_space_mapping_loss = space_mapping_loss(rv_ents, final_ents, rv_mapping, self.eye, self.orthogonal_weight)
        av_space_mapping_loss = space_mapping_loss(av_ents, final_ents, av_mapping, self.eye, self.orthogonal_weight)
        loss = nv_space_mapping_loss + rv_space_mapping_loss + av_space_mapping_loss
        return loss

    def forward(self, preds, weights, view):
        return self.cfg[view](*preds, *weights)
