import gc

import torch
import torch.nn as nn
import numpy as np

from pytorch.utils import l2_normalize
from base.evaluation import valid


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
        self.fc = nn.Sequential(
            nn.Linear(2 * output_dim * input_dim, input_dim, bias=True),
            activ()
        )

    def forward(self, attr_hs, attr_as, attr_vs):
        x = torch.stack([attr_as, attr_vs], dim=1).unsqueeze(3)  # Nx2xDx1
        x_bn = self.bn(x.permute(0, 2, 1, 3)).permute(0, 3, 2, 1)
        x_conv = self.conv_block(x_bn)  # Nx2x2xD
        x_conv_norm = l2_normalize(x_conv.permute(0, 2, 3, 1), dim=2)  # Nx2xDx2
        x_fc = self.fc(x_conv_norm.flatten(1))
        x_fc_norm = l2_normalize(x_fc)  # Important!!
        score = -torch.sum(torch.square(attr_hs - x_fc_norm), dim=1)
        return score


class MultiKENet(nn.Module):

    def __init__(self, num_entities, num_relations, num_attributes, embed_dim, value_vectors, local_name_vectors):
        super(MultiKENet, self).__init__()
        self.register_buffer('literal_embeds', torch.from_numpy(value_vectors), persistent=False)
        self.register_buffer('name_embeds', torch.from_numpy(local_name_vectors), persistent=False)

        # Relation view
        self.rv_ent_embeds = nn.Parameter(torch.Tensor(num_entities, embed_dim))
        self.rel_embeds = nn.Parameter(torch.Tensor(num_relations, embed_dim))

        # Attribute view
        self.av_ent_embeds = nn.Parameter(torch.Tensor(num_entities, embed_dim))
        self.attr_embeds = nn.Parameter(torch.Tensor(num_attributes, embed_dim))  # False important!
        self.attr_conv = Conv(embed_dim)

        # Shared embeddings
        self.ent_embeds = nn.Parameter(torch.Tensor(num_entities, embed_dim))

        # Shared combination
        self.nv_mapping = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.rv_mapping = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.av_mapping = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.register_buffer('eye', torch.eye(embed_dim), persistent=False)

        self.lookups_cfg = {
            'rv': self.relation_triple_lookup,
            'av': self.attribute_triple_lookup,
            'ckgrtv': self.cross_kg_relation_triple_lookup,
            'ckgatv': self.cross_kg_attribute_triple_lookup,
            'ckgrrv': self.cross_kg_relation_reference_lookup,
            'ckgarv': self.cross_kg_attribute_reference_lookup,
            'cnv': self.cross_name_view_lookup,
            'mv': self.multi_view_entities_lookup
        }
        self.params_cfg = {
            'rv': [self.rv_ent_embeds, self.rel_embeds],
            'av': [self.av_ent_embeds, self.attr_embeds, self.attr_conv],
            'ckgrtv': [self.rv_ent_embeds, self.rel_embeds],
            'ckgatv': [self.av_ent_embeds, self.attr_embeds, self.attr_conv],
            'ckgrrv': [self.rv_ent_embeds, self.rel_embeds],
            'ckgarv': [self.av_ent_embeds, self.attr_embeds, self.attr_conv],
            'cnv': [self.ent_embeds, self.rv_ent_embeds, self.av_ent_embeds],
            'mv': [self.ent_embeds, self.nv_mapping, self.rv_mapping, self.av_mapping]
        }

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'embeds' in name:
                nn.init.xavier_normal_(param)
            elif 'mapping' in name:
                nn.init.orthogonal_(param)

    def parameters(self, view, recurse=True):
        params = []
        for param in self.params_cfg[view]:
            if isinstance(param, nn.Module):
                params += list(param.parameters(recurse=recurse))
            else:
                params.append(param)
        return params

    def relation_triple_lookup(self, rel_pos_hs, rel_pos_rs, rel_pos_ts, rel_neg_hs, rel_neg_rs, rel_neg_ts):
        rv_ent_embeds = l2_normalize(self.rv_ent_embeds)
        rel_embeds = l2_normalize(self.rel_embeds)
        rel_phs = torch.index_select(rv_ent_embeds, dim=0, index=rel_pos_hs)
        rel_prs = torch.index_select(rel_embeds, dim=0, index=rel_pos_rs)
        rel_pts = torch.index_select(rv_ent_embeds, dim=0, index=rel_pos_ts)
        rel_nhs = torch.index_select(rv_ent_embeds, dim=0, index=rel_neg_hs)
        rel_nrs = torch.index_select(rel_embeds, dim=0, index=rel_neg_rs)
        rel_nts = torch.index_select(rv_ent_embeds, dim=0, index=rel_neg_ts)
        return rel_phs, rel_prs, rel_pts, rel_nhs, rel_nrs, rel_nts

    def attribute_triple_lookup(self, attr_pos_hs, attr_pos_as, attr_pos_vs):
        av_ent_embeds = l2_normalize(self.av_ent_embeds)
        attr_phs = torch.index_select(av_ent_embeds, dim=0, index=attr_pos_hs)
        attr_pas = torch.index_select(self.attr_embeds, dim=0, index=attr_pos_as)
        attr_pvs = torch.index_select(self.literal_embeds, dim=0, index=attr_pos_vs)
        pos_score = self.attr_conv(attr_phs, attr_pas, attr_pvs)
        return pos_score

    def cross_kg_relation_triple_lookup(self, ckge_rel_pos_hs, ckge_rel_pos_rs, ckge_rel_pos_ts):
        rv_ent_embeds = l2_normalize(self.rv_ent_embeds)
        rel_embeds = l2_normalize(self.rel_embeds)
        ckge_rel_phs = torch.index_select(rv_ent_embeds, dim=0, index=ckge_rel_pos_hs)
        ckge_rel_prs = torch.index_select(rel_embeds, dim=0, index=ckge_rel_pos_rs)
        ckge_rel_pts = torch.index_select(rv_ent_embeds, dim=0, index=ckge_rel_pos_ts)
        return ckge_rel_phs, ckge_rel_prs, ckge_rel_pts

    def cross_kg_attribute_triple_lookup(self, ckge_attr_pos_hs, ckge_attr_pos_as, ckge_attr_pos_vs):
        av_ent_embeds = l2_normalize(self.av_ent_embeds)
        ckge_attr_phs = torch.index_select(av_ent_embeds, dim=0, index=ckge_attr_pos_hs)
        ckge_attr_pas = torch.index_select(self.attr_embeds, dim=0, index=ckge_attr_pos_as)
        ckge_attr_pvs = torch.index_select(self.literal_embeds, dim=0, index=ckge_attr_pos_vs)
        pos_score = self.attr_conv(ckge_attr_phs, ckge_attr_pas, ckge_attr_pvs)
        return pos_score

    def cross_kg_relation_reference_lookup(self, ckgp_rel_pos_hs, ckgp_rel_pos_rs, ckgp_rel_pos_ts):
        rv_ent_embeds = l2_normalize(self.rv_ent_embeds)
        rel_embeds = l2_normalize(self.rel_embeds)
        ckgp_rel_phs = torch.index_select(rv_ent_embeds, dim=0, index=ckgp_rel_pos_hs)
        ckgp_rel_prs = torch.index_select(rel_embeds, dim=0, index=ckgp_rel_pos_rs)
        ckgp_rel_pts = torch.index_select(rv_ent_embeds, dim=0, index=ckgp_rel_pos_ts)
        return ckgp_rel_phs, ckgp_rel_prs, ckgp_rel_pts

    def cross_kg_attribute_reference_lookup(self, ckga_attr_pos_hs, ckga_attr_pos_as, ckga_attr_pos_vs):
        av_ent_embeds = l2_normalize(self.av_ent_embeds)
        ckga_attr_phs = torch.index_select(av_ent_embeds, dim=0, index=ckga_attr_pos_hs)
        ckga_attr_pas = torch.index_select(self.attr_embeds, dim=0, index=ckga_attr_pos_as)
        ckga_attr_pvs = torch.index_select(self.literal_embeds, dim=0, index=ckga_attr_pos_vs)
        pos_score = self.attr_conv(ckga_attr_phs, ckga_attr_pas, index=ckga_attr_pvs)
        return pos_score

    def cross_name_view_lookup(self, cn_hs):
        ent_embeds = l2_normalize(self.ent_embeds)
        rv_ent_embeds = l2_normalize(self.rv_ent_embeds)
        av_ent_embeds = l2_normalize(self.av_ent_embeds)
        final_cn_phs = torch.index_select(ent_embeds, dim=0, index=cn_hs)
        cn_hs_names = torch.index_select(self.name_embeds, dim=0, index=cn_hs)
        cr_hs = torch.index_select(rv_ent_embeds, dim=0, index=cn_hs)
        ca_hs = torch.index_select(av_ent_embeds, dim=0, index=cn_hs)
        return final_cn_phs, cn_hs_names, cr_hs, ca_hs

    def multi_view_entities_lookup(self, entities):
        ent_embeds = l2_normalize(self.ent_embeds)
        rv_ent_embeds = l2_normalize(self.rv_ent_embeds)
        av_ent_embeds = l2_normalize(self.av_ent_embeds)
        final_ents = torch.index_select(ent_embeds, dim=0, index=entities)
        nv_ents = torch.index_select(self.name_embeds, dim=0, index=entities)
        rv_ents = torch.index_select(rv_ent_embeds, dim=0, index=entities)
        av_ents = torch.index_select(av_ent_embeds, dim=0, index=entities)
        return final_ents, nv_ents, rv_ents, av_ents

    def forward(self, inputs, view):
        return self.lookups_cfg[view](*inputs)

    @staticmethod
    def valid(model, kgs, embed_choice='avg', w=(1, 1, 1), top_k=[1], num_threads=1):
        if embed_choice == 'nv':
            ent_embeds = model.name_embeds
        elif embed_choice == 'rv':
            ent_embeds = l2_normalize(model.rv_ent_embeds)
        elif embed_choice == 'av':
            ent_embeds = l2_normalize(model.av_ent_embeds)
        elif embed_choice == 'final':
            ent_embeds = l2_normalize(model.ent_embeds)
        elif embed_choice == 'avg':
            ent_embeds = w[0] * model.name_embeds + \
                         w[1] * l2_normalize(model.rv_ent_embeds) + \
                         w[2] * l2_normalize(model.av_ent_embeds)
        else:  # 'final'
            ent_embeds = l2_normalize(model.ent_embeds)
        print(embed_choice, 'valid results:')
        embeds1 = ent_embeds[kgs.valid_entities1, ]
        embeds2 = ent_embeds[kgs.valid_entities2 + kgs.test_entities2, ]
        _, mrr_12 = valid(embeds1, embeds2, None, top_k, num_threads, normalize=True)
        del embeds1, embeds2
        gc.collect()
        return mrr_12

    @staticmethod
    def test(model, kgs, embed_choice='avg', w=(1, 1, 1), top_k=[1], num_threads=1):
        if embed_choice == 'nv':
            ent_embeds = model.name_embeds
        elif embed_choice == 'rv':
            ent_embeds = l2_normalize(model.rv_ent_embeds)
        elif embed_choice == 'av':
            ent_embeds = l2_normalize(model.av_ent_embeds)
        elif embed_choice == 'final':
            ent_embeds = l2_normalize(model.ent_embeds)
        elif embed_choice == 'avg':
            ent_embeds = w[0] * model.name_embeds + \
                         w[1] * l2_normalize(model.rv_ent_embeds) + \
                         w[2] * l2_normalize(model.av_ent_embeds)
        else:  # 'final'
            ent_embeds = l2_normalize(model.ent_embeds)
        print(embed_choice, 'test results:')
        embeds1 = ent_embeds[kgs.test_entities1, ]
        embeds2 = ent_embeds[kgs.test_entities2, ]
        _, mrr_12 = valid(embeds1, embeds2, None, top_k, num_threads, normalize=True)
        del embeds1, embeds2
        gc.collect()
        return mrr_12


if __name__ == '__main__':
    conv = Conv(75)
    score = conv(torch.rand(32, 75), torch.rand(32, 75), torch.rand(32, 75))
    print(conv)
    model = MultiKENet(200000, 550, 1086, 75, np.zeros((909911, 75), dtype=np.float32), np.zeros((200000, 75), dtype=np.float32))
    inputs = torch.randint(75, (32,)), torch.randint(75, (32,)), torch.randint(75, (32,))
    outputs = model(inputs, 'av')