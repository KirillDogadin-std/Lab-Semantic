import multiprocessing as mp
import numpy as np
import os
import random
import time
from utils import torch, nn

from losses_torch import relation_logistic_loss, logistic_loss_wo_negs,\
    relation_logistic_loss_wo_negs


def xavier_init(dim1, dim2, is_l2_norm):
    ret = torch.empty(dim1, dim2)
    ret = nn.Parameter(
        nn.init.xavier_normal_(ret)
    )
    if is_l2_norm:
        ret = nn.functional.normalize(ret, dim=1, p=2)

    return ret


def dict2file(file, dic):
    if dic is None:
        return
    with open(file, 'w', encoding='utf8') as f:
        for i, j in dic.items():
            f.write(str(i) + '\t' + str(j) + '\n')
        f.close()
    print(file, "saved.")


def save_embeddings(folder, kgs, ent_embeds, nv_ent_embeds, rv_ent_embeds, av_ent_embeds, rel_embeds, attr_embeds):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if ent_embeds is not None:
        np.save(folder + 'ent_embeds.npy', ent_embeds)
    if ent_embeds is not None:
        np.save(folder + 'nv_ent_embeds.npy', nv_ent_embeds)
    if ent_embeds is not None:
        np.save(folder + 'rv_ent_embeds.npy', rv_ent_embeds)
    if ent_embeds is not None:
        np.save(folder + 'av_ent_embeds.npy', av_ent_embeds)
    if rel_embeds is not None:
        np.save(folder + 'rel_embeds.npy', rel_embeds)
    if attr_embeds is not None:
        np.save(folder + 'attr_embeds.npy', attr_embeds)
    dict2file(folder + 'kg1_ent_ids', kgs.kg1.entities_id_dict)
    dict2file(folder + 'kg2_ent_ids', kgs.kg2.entities_id_dict)
    dict2file(folder + 'kg1_rel_ids', kgs.kg1.relations_id_dict)
    dict2file(folder + 'kg2_rel_ids', kgs.kg2.relations_id_dict)
    dict2file(folder + 'kg1_attr_ids', kgs.kg1.attributes_id_dict)
    dict2file(folder + 'kg2_attr_ids', kgs.kg2.attributes_id_dict)
    print("Embeddings saved!")


def conv(attr_hs, attr_as, attr_vs, dim, feature_map_size=2, kernel_size=[2, 4], activation=nn.Tanh, layer_num=2):
    # print("feature map size", feature_map_size)
    # print("kernel size", kernel_size)
    # print("layer_num", layer_num)

    class Conv(nn.Module):
        def __init__(self, sizes):
            super(Conv, self).__init__()
            self.norm_channels = sizes[3]

            self.in_channels = sizes[1]
            self.out_channels = sizes[2]

            self.con = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=[2, 4])
            self.con_consequent = nn.Conv2d(self.in_channels + 1, self.out_channels, kernel_size=[2, 4])
            self.nrm = nn.BatchNorm2d(self.norm_channels)

            self.activ_func_conv = activation()

        def forward(self, input, conv_times, pad=(2, 1, 1, 0)):

            sizes = list(input.shape)

            input = torch.reshape(input, [-1, sizes[3], sizes[2], 1])
            res = self.nrm(input)
            res = torch.reshape(res, [-1, 1, sizes[2], sizes[3]])

            padded = nn.functional.pad(res, pad)
            res = self.con(padded)
            res = self.activ_func_conv(res)

            for i in range(conv_times-1):
                padded = nn.functional.pad(res, pad)
                res = self.con_consequent(padded)
                res = self.activ_func_conv(res)

            return res

    attr_as = torch.reshape(attr_as, [-1, 1, dim])
    attr_vs = torch.reshape(attr_vs, [-1, 1, dim])

    input_avs = torch.cat([attr_as, attr_vs], 1)
    input_shape = list(input_avs.shape)
    input_layer = torch.reshape(input_avs, [-1, 1, input_shape[1], input_shape[0]])

    c = Conv(list(input_layer.shape))
    _conv = c.forward(input_layer, layer_num)

    _conv = nn.functional.normalize(_conv, dim=1, p=2)
    _shape = list(_conv.shape)
    _flat = torch.reshape(_conv, [-1, _shape[3] * _shape[2] * _shape[1]])
    # print("_flat", _flat.shape)

    class FlatProcessing(nn.Module):
        def __init__(self, in_, out_, activation_):
            super(FlatProcessing, self).__init__()
            self.pipeline = nn.Sequential(
                nn.Linear(in_, out_),
                activation_()
            )

        def forward(self, arg):
            return self.pipeline(arg)
    breakpoint()
    denser = FlatProcessing(_flat.shape[1], dim, activation_=activation)
    dense = denser.forward(_flat)
    dense = nn.functional.normalize(dense, dim=1, p=2)
    # print("dense", dense.shape)
    score = -torch.sum(torch.square(attr_hs - dense), 1)
    return score


def get_optimizer(model, opt, learning_rate):
    if opt == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif opt == 'Adadelta':
        # To match the exact form in the original paper use 1.0.
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    elif opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:  # opt == 'SGD'
        optimizer = torch.optim.optim.SGD(model.parameters(), lr=learning_rate)
    return optimizer


def generate_optimizer(model, loss, learning_rate, var_list=None, opt='SGD'):
    optimizer = get_optimizer(model, opt, learning_rate)
    optimizer.zero_grad()
    loss.backward()
    return optimizer


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
    self.relation_optimizer = generate_optimizer(self.model,
                                                 self.relation_loss,
                                                 self.args.learning_rate,
                                                 opt=self.args.optimizer)
    return self.relation_loss, self.relation_optimizer


def _define_attribute_view_graph(self, **kwargs):

    self.attr_pos_hs = kwargs['attr_pos_hs']
    self.attr_pos_as = kwargs['attr_pos_as']
    self.attr_pos_vs = kwargs['attr_pos_vs']
    self.attr_pos_ws = kwargs['attr_pos_ws']

    attr_phs = self.av_ent_embeds[self.attr_pos_hs]
    attr_pas = self.attr_embeds[self.attr_pos_as]
    attr_pvs = self.literal_embeds[self.attr_pos_vs]
    pos_score = conv(attr_phs, attr_pas, attr_pvs, self.args.dim)
    pos_score = torch.log(1 + torch.exp(-pos_score))
    pos_score = torch.multiply(pos_score, self.attr_pos_ws)
    pos_loss = torch.reduce_sum(pos_score)
    self.attribute_loss = pos_loss
    self.attribute_optimizer = generate_optimizer(self.attribute_loss, self.args.learning_rate,
                                                  opt=self.args.optimizer)

    return self.attribute_loss, self.attribute_optimizer


def _define_cross_kg_name_view_graph(self):
    pass


def _define_cross_kg_entity_reference_relation_view_graph(self):
    ckge_rel_phs = self.rv_ent_embeds[self.ckge_rel_pos_hs]
    ckge_rel_prs = self.rel_embeds[self.ckge_rel_pos_rs]
    ckge_rel_pts = self.rv_ent_embeds[self.ckge_rel_pos_ts]
    self.ckge_relation_loss = 2 * \
        relation_logistic_loss_wo_negs(ckge_rel_phs, ckge_rel_prs, ckge_rel_pts)
    self.ckge_relation_optimizer = generate_optimizer(self.model, self.ckge_relation_loss, self.args.learning_rate,
                                                      opt=self.args.optimizer)


def _define_cross_kg_entity_reference_attribute_view_graph(self):
    ckge_attr_phs = self.av_ent_embeds[self.ckge_attr_pos_hs]
    ckge_attr_pas = self.attr_embeds[self.ckge_attr_pos_as]
    ckge_attr_pvs = self.literal_embeds[self.ckge_attr_pos_vs]

    pos_score = conv(ckge_attr_phs, ckge_attr_pas, ckge_attr_pvs, self.args.dim)
    self.ckge_attribute_loss = 2 * torch.sum(torch.log(1 + torch.exp(-pos_score)))
    self.ckge_attribute_optimizer = generate_optimizer(self.model, self.ckge_attribute_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)


def _define_cross_kg_relation_reference_graph(self):
    ckgp_rel_phs = self.rv_ent_embeds[self.ckgp_rel_pos_hs]
    ckgp_rel_prs = self.rel_embeds[self.ckgp_rel_pos_rs]
    ckgp_rel_pts = self.rv_ent_embeds[self.ckgp_rel_pos_ts]

    self.ckgp_relation_loss = 2 * logistic_loss_wo_negs(ckgp_rel_phs, ckgp_rel_prs, ckgp_rel_pts,
                                                        self.ckgp_rel_pos_ws)
    self.ckgp_relation_optimizer = generate_optimizer(self.model, self.ckgp_relation_loss, self.args.learning_rate,
                                                      opt=self.args.optimizer)


def _define_cross_kg_attribute_reference_graph(self):
    ckga_attr_phs = self.av_ent_embeds[self.ckga_attr_pos_hs]
    ckga_attr_pas = self.attr_embeds[self.ckga_attr_pos_as]
    ckga_attr_pvs = self.literal_embeds[self.ckga_attr_pos_vs]

    pos_score = conv(ckga_attr_phs, ckga_attr_pas, ckga_attr_pvs, self.args.dim)
    pos_score = torch.log(1 + torch.exp(-pos_score))
    pos_score = torch.multiply(pos_score, self.ckga_attr_pos_ws)
    pos_loss = torch.reduce_sum(pos_score)
    self.ckga_attribute_loss = pos_loss
    # self.ckga_attribute_loss = tf.reduce_sum(tf.log(1 + tf.exp(-pos_score)))
    self.ckga_attribute_optimizer = generate_optimizer(self.model, self.ckga_attribute_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)


def _define_common_space_learning_graph(self):
    final_cn_phs = self.ent_embeds[self.cn_hs]
    cn_hs_names = self.name_embeds[self.cn_hs]
    cr_hs = self.rv_ent_embeds[self.cn_hs]
    ca_hs = self.av_ent_embeds[self.cn_hs]

    self.cross_name_loss = self.args.cv_name_weight * alignment_loss(final_cn_phs, cn_hs_names)
    self.cross_name_loss += alignment_loss(final_cn_phs, cr_hs)
    self.cross_name_loss += alignment_loss(final_cn_phs, ca_hs)
    self.cross_name_optimizer = generate_optimizer(self.args.cv_weight * self.cross_name_loss,
                                                   self.args.ITC_learning_rate,
                                                   opt=self.args.optimizer)


def _define_space_mapping_graph(self):
    final_ents = self.ent_embeds[self.entities]
    nv_ents = self.name_embeds[self.entities]
    rv_ents = self.rv_ent_embeds[self.entities]
    av_ents = self.av_ent_embeds[self.entities]

    nv_space_mapping_loss = space_mapping_loss(nv_ents, final_ents, self.nv_mapping, self.eye_mat,
                                               self.args.orthogonal_weight)
    rv_space_mapping_loss = space_mapping_loss(rv_ents, final_ents, self.rv_mapping, self.eye_mat,
                                               self.args.orthogonal_weight)
    av_space_mapping_loss = space_mapping_loss(av_ents, final_ents, self.av_mapping, self.eye_mat,
                                                self.args.orthogonal_weight)
    self.shared_comb_loss = nv_space_mapping_loss + rv_space_mapping_loss + av_space_mapping_loss
    
    # TODO
    opt_vars = [v for v in tf.trainable_variables() if v.name.startswith("shared")]

    self.shared_comb_optimizer = generate_optimizer(self.shared_comb_loss,
                                                    self.args.learning_rate,
                                                    var_list=opt_vars,
                                                    opt=self.args.optimizer)


def eval_kg1_ent_embeddings(self):
    embeds = self.rv_ent_embeds[self.kgs.kg1.entities_list]
    return embeds


def eval_kg2_ent_embeddings(self):
    embeds = self.rv_ent_embeds[self.kgs.kg2.entities_list]
    return embeds


def eval_kg1_useful_ent_embeddings(self):
    embeds = self.rv_ent_embeds[self.kgs.useful_entities_list1]
    return embeds


def eval_kg2_useful_ent_embeddings(self):
    embeds = self.rv_ent_embeds[self.kgs.useful_entities_list2]
    return embeds


def save(self):
    ent_embeds = self.ent_embeds
    nv_ent_embeds = self.name_embeds
    rv_ent_embeds = self.rv_ent_embeds
    av_ent_embeds = self.av_ent_embeds
    rel_embeds = self.rel_embeds
    att_embeds = self.attr_embeds
    save_embeddings(self.out_folder, self.kgs, ent_embeds, nv_ent_embeds, rv_ent_embeds, av_ent_embeds,
                    rel_embeds, att_embeds)


def train_relation_view_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
    start = time.time()
    epoch_loss = 0
    trained_samples_num = 0
    for steps_task in steps_tasks:
        mp.Process(target=bat.generate_relation_triple_batch_queue,
                    args=(self.kgs.kg1.local_relation_triples_list, self.kgs.kg2.local_relation_triples_list,
                            self.kgs.kg1.local_relation_triples_set, self.kgs.kg2.local_relation_triples_set,
                            self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                            self.args.batch_size, steps_task,
                            batch_queue, neighbors1, neighbors2, self.args.neg_triple_num)).start()
    for i in range(triple_steps):
        batch_pos, batch_neg = batch_queue.get()
        batch_loss, _ = _define_relation_view_graph({'rel_pos_hs': [x[0] for x in batch_pos],
                                                    'rel_pos_rs': [x[1] for x in batch_pos],
                                                    'rel_pos_ts': [x[2] for x in batch_pos],
                                                    'rel_neg_hs': [x[0] for x in batch_neg],
                                                    'rel_neg_rs': [x[1] for x in batch_neg],
                                                    'rel_neg_ts': [x[2] for x in batch_neg]})
        self.relation_optimizer.zero_grad()
        self.relation_loss.backward()
        self.relation_optimizer.step()
        trained_samples_num += len(batch_pos)
        epoch_loss += batch_loss
    epoch_loss /= trained_samples_num
    random.shuffle(self.kgs.kg1.local_relation_triples_list)
    random.shuffle(self.kgs.kg2.local_relation_triples_list)
    end = time.time()
    print('epoch {} of rel. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss, end - start))


# Train section
def train_attribute_view_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
    start = time.time()
    epoch_loss = 0
    trained_samples_num = 0
    for steps_task in steps_tasks:
        mp.Process(target=generate_attribute_triple_batch_queue,
                   args=(self.predicate_align_model.attribute_triples_w_weights1,
                         self.predicate_align_model.attribute_triples_w_weights2,
                         self.predicate_align_model.attribute_triples_w_weights_set1,
                         self.predicate_align_model.attribute_triples_w_weights_set2,
                         self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                         self.args.attribute_batch_size, steps_task,
                         batch_queue, neighbors1, neighbors2, 0)).start()
    for i in range(triple_steps):
        batch_pos, batch_neg = batch_queue.get()
        # self.attribute_loss, self.attribute_optimizer:
        batch_loss, _ = _define_attribute_view_graph({'attr_pos_hs': [x[0] for x in batch_pos],
                                                      'attr_pos_as': [x[1] for x in batch_pos],
                                                      'attr_pos_vs': [x[2] for x in batch_pos],
                                                      'attr_pos_ws': [x[3] for x in batch_pos]})
        self.attribute_optimizer.zero_grad()
        self.attribute_loss.backward()
        self.attribute_optimizer.step()

        trained_samples_num += len(batch_pos)
        epoch_loss += batch_loss
    epoch_loss /= trained_samples_num
    random.shuffle(self.predicate_align_model.attribute_triples_w_weights1)
    random.shuffle(self.predicate_align_model.attribute_triples_w_weights2)
    end = time.time()
    print('epoch {} of att. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss, end - start))
