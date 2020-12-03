from utils import *


def xavier_init(dim1, dim2, is_l2_norm):
    ret = torch.empty(dim1, dim2)
    ret = nn.Parameter(
        nn.init.xavier_normal_(ret)
    )
    if is_l2_norm:
        ret = nn.functional.normalize(ret, dim=1, p=2)

    return ret


def orthogonal_init(dim1, dim2):
    ret = torch.empty(dim1, dim2)
    ret = nn.Parameter(
        nn.init.orthogonal_(ret)
    )

    return ret


def _define_variables(self):
    self.literal_embeds = torch.Tensor(self.data.valueS_vectors, dtype=torch.float32)
    self.name_embeds = torch.Tensor(self.data.local_name_vectors, dtype=torch.float32)

    self.rv_ent_embeds = xavier_init(self.kgs.entities_num, self.args.dim, True)
    self.rel_embeds = xavier_init(self.kgs.entities_num, self.args.dim, True)
    self.av_ent_embeds = xavier_init(self.kgs.entities_num, self.args.dim, True)
    self.attr_embeds = xavier_init(self.kgs.entities_num, self.args.dim, False)
    self.ent_embeds = xavier_init(self.kgs.entities_num, self.args.dim, True)

    self.nv_mapping = orthogonal_init(self.args.dim, self.args.dim)
    self.rv_mapping = orthogonal_init(self.args.dim, self.args.dim)
    self.av_mapping = orthogonal_init(self.args.dim, self.args.dim)

    self.eye_mat = torch.Tensor(np.eye(self.args.dim), dtype=torch.float32)


def _define_relation_view_graph(self):
    rel_phs = self.rv_ent_embeds[self.rel_pos_hs]
    rel_prs = self.rel_embeds[self.rel_pos_rs]
    rel_pts = self.rv_ent_embeds[self.rel_pos_ts]
    rel_nhs = self.rv_ent_embeds[self.rel_neg_hs]
    rel_nrs = self.rel_embeds[self.rel_neg_rs]
    rel_nts = self.rv_ent_embeds[self.rel_neg_ts]

    self.relation_loss = relation_logistic_loss(rel_phs, rel_prs, rel_pts, rel_nhs, rel_nrs, rel_nts)
    self.relation_optimizer = generate_optimizer(
        self.relation_loss, self.args.learning_rate, opt=self.args.optimizer)
