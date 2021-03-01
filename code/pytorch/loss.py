import torch
from torch import nn
from torch.nn import functional as F
from pytorch.utils import l2_normalize


def positive_loss(pos_score, pws=None):
    pos_score = torch.log(1 + torch.exp(-pos_score))
    if pws is not None:
        pos_score = torch.multiply(pos_score, pws)
    loss = torch.sum(pos_score)
    return loss


def logistic_loss(pos_score, neg_score, pws=None, nws=None):
    pos_score = torch.log(1 + torch.exp(-pos_score))
    neg_score = torch.log(1 + torch.exp(neg_score))
    if None not in (pws, nws):
        pos_score = torch.multiply(pos_score, pws)
        neg_score = torch.multiply(neg_score, nws)
    pos_loss = torch.sum(pos_score)
    neg_loss = torch.sum(neg_score)
    loss = pos_loss + neg_loss
    return loss


def margin_loss(pos_score, neg_score, margin, pws=None, nws=None):
    if None not in (pws, nws):
        pos_score = torch.multiply(pos_score, pws)
        neg_score = torch.multiply(neg_score, nws)
    loss = torch.sum(F.relu(margin - pos_score + neg_score))
    return loss


def orthogonal_loss(mapping, eye):
    loss = torch.sum(torch.sum(torch.pow(torch.matmul(mapping, mapping.t()) - eye, 2), dim=1))
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
            self.cfg['mv'] = self.multi_view_loss

    def relation_triple_loss(self, pos_score, neg_score):
        loss = logistic_loss(pos_score, neg_score)
        return loss

    def attribute_triple_loss(self, pos_score, attr_pos_ws):
        loss = positive_loss(pos_score, attr_pos_ws)
        return loss

    def cross_kg_relation_triple_loss(self, pos_score):
        loss = 2 * positive_loss(pos_score)
        return loss

    def cross_kg_attribute_triple_loss(self, pos_score):
        loss = 2 * positive_loss(pos_score)
        return loss

    def cross_kg_relation_reference_loss(self, pos_score, ckgp_rel_pos_ws):
        loss = 2 * positive_loss(pos_score, ckgp_rel_pos_ws)
        return loss

    def cross_kg_attribute_reference_loss(self, pos_score, ckga_attr_pos_ws):
        loss = positive_loss(pos_score, ckga_attr_pos_ws)
        return loss

    def cross_name_view_loss(self, final_cn_phs, cn_hs_names, cr_hs, ca_hs):
        loss = self.cv_name_weight * alignment_loss(final_cn_phs, cn_hs_names)
        loss += alignment_loss(final_cn_phs, cr_hs)
        loss += alignment_loss(final_cn_phs, ca_hs)
        loss = self.cv_weight * loss
        return loss

    def multi_view_loss(self, final_ents, nv_ents, rv_ents, av_ents, nv_mapping, rv_mapping, av_mapping):
        nv_space_mapping_loss = space_mapping_loss(nv_ents, final_ents, nv_mapping, self.eye, self.orthogonal_weight)
        rv_space_mapping_loss = space_mapping_loss(rv_ents, final_ents, rv_mapping, self.eye, self.orthogonal_weight)
        av_space_mapping_loss = space_mapping_loss(av_ents, final_ents, av_mapping, self.eye, self.orthogonal_weight)
        loss = nv_space_mapping_loss + rv_space_mapping_loss + av_space_mapping_loss
        return loss

    def forward(self, preds, weights, view):
        """
        Calculate the loss of specified view.

        Parameters
        ----------
        preds: list
            Scores predicted from model.
        weights: list
            Weights of samples.
        view: str
            Name of view.

        Returns
        -------
        loss : the calculated loss
        """
        return self.cfg[view](*preds, *weights)
