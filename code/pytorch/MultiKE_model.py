import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf

from pytorch.utils import L2Normalize


class Conv(nn.Module):

    def __init__(self, input_dim, output_dim=2, kernel_size=(2, 4), activ=nn.Tanh, num_layers=2):
        super(Conv, self).__init__()
        in_dim, layers = 1, []
        for i in range(num_layers):
            layers += [
                nn.Conv2d(in_dim, output_dim, kernel_size, stride=1, padding=0),
                nn.ZeroPad2d((1, 2, 0, 1)),
                activ()
            ]
            in_dim = output_dim

        self.bn = nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01, affine=True)
        self.conv_block = nn.Sequential(*layers)
        self.l2_normalize = L2Normalize()
        self.fc = nn.Sequential(
            nn.Linear(2 * output_dim * input_dim, input_dim, bias=True),
            activ()
        )

    def forward(self, attr_hs, attr_as, attr_vs):
        x = torch.stack([attr_as, attr_vs], dim=1).unsqueeze(3)  # Nx2xDx1
        x_bn = self.bn(x.permute(0, 2, 1, 3)).permute(0, 3, 2, 1)
        x_conv = self.conv_block(x_bn)  # Nx2x2xD
        x_conv_norm = self.l2_normalize(x_conv.permute(0, 2, 3, 1), dim=2)  # Nx2xDx2
        x_fc = self.fc(x_conv_norm.flatten(1))
        x_fc_norm = self.l2_normalize(x_fc)  # Important!!
        score = -torch.sum(torch.square(attr_hs - x_fc_norm), dim=1)
        return score


class MultiKE(nn.Module):

    def __init__(self, num_entities, num_relations, num_attributes, embed_dim, value_vectors, local_name_vectors):
        super(MultiKE, self).__init__()
        self.l2_normalize = L2Normalize()
        self.register_buffer('literal_embeds', torch.from_numpy(value_vectors))
        self.register_buffer('name_embeds', torch.from_numpy(local_name_vectors))

        # Relation view
        self.rv_ent_embeds = nn.Parameter(torch.Tensor(num_entities, embed_dim))
        self.rel_embeds = nn.Parameter(torch.Tensor(num_relations, embed_dim))

        # Attribute view
        self.av_ent_embeds = nn.Parameter(torch.Tensor(num_entities, embed_dim))
        self.attr_embeds = nn.Parameter(torch.Tensor(num_attributes, embed_dim))  # False important!
        self.attr_conv = Conv(embed_dim)
        self.attr_triple_conv = Conv(embed_dim)
        self.attr_ref_conv = Conv(embed_dim)

        # Shared embeddings
        self.ent_embeds = nn.Parameter(torch.Tensor(num_entities, embed_dim))

        # Shared combination
        self.nv_mapping = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.rv_mapping = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.av_mapping = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.register_buffer('eye', torch.eye(embed_dim))

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'embeds' in name:
                nn.init.xavier_normal_(param)
            elif 'mapping' in name:
                nn.init.orthogonal_(param)

    def relation_view(self, rel_pos_hs, rel_pos_rs, rel_pos_ts, rel_neg_hs, rel_neg_rs, rel_neg_ts):
        rv_ent_embeds = self.l2_normalize(self.rv_ent_embeds)
        rel_embeds = self.l2_normalize(self.rel_embeds)
        rel_phs = torch.index_select(rv_ent_embeds, dim=0, index=rel_pos_hs)
        rel_prs = torch.index_select(rel_embeds, dim=0, index=rel_pos_rs)
        rel_pts = torch.index_select(rv_ent_embeds, dim=0, index=rel_pos_ts)
        rel_nhs = torch.index_select(rv_ent_embeds, dim=0, index=rel_neg_hs)
        rel_nrs = torch.index_select(rel_embeds, dim=0, index=rel_neg_rs)
        rel_nts = torch.index_select(rv_ent_embeds, dim=0, index=rel_neg_ts)
        return rel_phs, rel_prs, rel_pts, rel_nhs, rel_nrs, rel_nts

    def attribute_view(self, attr_pos_hs, attr_pos_as, attr_pos_vs):
        # Question: Why not l2_normalize? like in `relation_view`
        attr_phs = torch.index_select(self.av_ent_embeds, dim=0, index=attr_pos_hs)
        attr_pas = torch.index_select(self.attr_embeds, dim=0, index=attr_pos_as)
        attr_pvs = torch.index_select(self.literal_embeds, dim=0, index=attr_pos_vs)
        pos_score = self.attr_conv(attr_phs, attr_pas, attr_pvs)
        return pos_score

    def cross_kg_relation_triple(self):
        ckge_rel_phs = torch.index_select(self.rv_ent_embeds, dim=0, index=self.ckge_rel_pos_hs)
        ckge_rel_prs = torch.index_select(self.rel_embeds, dim=0, index=self.ckge_rel_pos_rs)
        ckge_rel_pts = torch.index_select(self.rv_ent_embeds, dim=0, index=self.ckge_rel_pos_ts)
        return (ckge_rel_phs, ckge_rel_prs, ckge_rel_pts)

    def cross_kg_attribute_triple(self, ckge_attr_pos_hs, ckge_attr_pos_as, ckge_attr_pos_vs):
        ckge_attr_phs = torch.index_select(self.av_ent_embeds, dim=0, index=ckge_attr_pos_hs)
        ckge_attr_pas = torch.index_select(self.attr_embeds, dim=0, index=ckge_attr_pos_as)
        ckge_attr_pvs = torch.index_select(self.literal_embeds, dim=0, index=ckge_attr_pos_vs)
        pos_score = self.attr_triple_conv(ckge_attr_phs, ckge_attr_pas, ckge_attr_pvs)
        return pos_score

    def cross_kg_relation_reference(self):
        ckgp_rel_phs = torch.index_select(self.rv_ent_embeds, dim=0, index=self.ckgp_rel_pos_hs)
        ckgp_rel_prs = torch.index_select(self.rel_embeds, dim=0, index=self.ckgp_rel_pos_rs)
        ckgp_rel_pts = torch.index_select(self.rv_ent_embeds, dim=0, index=self.ckgp_rel_pos_ts)
        return ckgp_rel_phs, ckgp_rel_prs, ckgp_rel_pts

    def cross_kg_attribute_reference(self, ckga_attr_pos_hs, ckga_attr_pos_as, ckga_attr_pos_vs):
        ckga_attr_phs = torch.index_select(self.av_ent_embeds, dim=0, index=ckga_attr_pos_hs)
        ckga_attr_pas = torch.index_select(self.attr_embeds, dim=0, index=ckga_attr_pos_as)
        ckga_attr_pvs = torch.index_select(self.literal_embeds, dim=0, index=ckga_attr_pos_vs)
        pos_score = self.attr_ref_conv(ckga_attr_phs, ckga_attr_pas, index=ckga_attr_pvs)
        return pos_score

    def cross_name_view(self):
        final_cn_phs = torch.index_select(self.ent_embeds, dim=0, index=self.cn_hs)
        cn_hs_names = torch.index_select(self.name_embeds, dim=0, index=self.cn_hs)
        cr_hs = torch.index_select(self.rv_ent_embeds, dim=0, index=self.cn_hs)
        ca_hs = torch.index_select(self.av_ent_embeds, dim=0, index=self.cn_hs)
        return (final_cn_phs, cn_hs_names, cr_hs, ca_hs)

    def multi_view(self):
        final_ents = torch.index_select(self.ent_embeds, dim=0, index=self.entities)
        nv_ents = torch.index_select(self.name_embeds, dim=0, index=self.entities)
        rv_ents = torch.index_select(self.rv_ent_embeds, dim=0, index=self.entities)
        av_ents = torch.index_select(self.av_ent_embeds, dim=0, index=self.entities)

        return final_ents, nv_ents, rv_ents, av_ents

    @staticmethod
    def valid(model, embed_choice='avg', w=(1, 1, 1)):
        pass

    @staticmethod
    def test(model, embed_choice='avg', w=(1, 1, 1)):
        pass


if __name__ == '__main__':
    conv = Conv(75)
    score = conv(torch.rand(32, 75), torch.rand(32, 75), torch.rand(32, 75))
    print(conv)
    model = MultiKE(200000, 550, 1086, 75, np.zeros((909911, 75)), np.zeros((200000, 75)))
