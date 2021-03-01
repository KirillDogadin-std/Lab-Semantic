import torch
import torch.nn as nn
import numpy as np
from sklearn import preprocessing
from torch.utils.data import DataLoader

from pytorch.utils import l2_normalize
from pytorch.data import TestDataset
from pytorch.finding.similarity import sim
from pytorch.finding.alignment import greedy_alignment


def _compute_weight(embeds1, embeds2, embeds3):
    other_embeds = (embeds1 + embeds2 + embeds3) / 3
    other_embeds = preprocessing.normalize(other_embeds)
    embeds1 = preprocessing.normalize(embeds1)
    sim_mat = np.matmul(embeds1, other_embeds.T)
    weights = np.diag(sim_mat)
    print(weights.shape, np.mean(weights))
    return np.mean(weights)


def wva(embeds1, embeds2, embeds3):
    weight1 = _compute_weight(embeds1, embeds2, embeds3)
    weight2 = _compute_weight(embeds2, embeds1, embeds3)
    weight3 = _compute_weight(embeds3, embeds1, embeds2)
    return weight1, weight2, weight3


def test(embeds1, embeds2, mapping, top_k, num_threads, metric='inner', normalize=False, csls_k=0, accurate=False):
    if mapping is None:
        alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(embeds1, embeds2, top_k, num_threads,
                                                                      metric, normalize, csls_k, accurate)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(test_embeds1_mapped, embeds2, top_k, num_threads,
                                                                      metric, normalize, csls_k, accurate)
    return alignment_rest_12, hits1_12, mrr_12


class TransE(nn.Module):

    def __init__(self):
        super(TransE, self).__init__()

    def forward(self, head, relation, tail):
        distance = head[0] + relation[0] - tail[0]
        score = torch.sum(torch.square(distance), dim=1)
        return -score

    def lookup(self, input, index=None):
        outputs = torch.index_select(input, dim=1, index=index) if index is not None else input
        return outputs.squeeze(0)


class ConvE(nn.Module):

    def __init__(self, input_dim, output_dim=2, kernel_size=(2, 4), activ=nn.Tanh, num_layers=2):
        super(ConvE, self).__init__()
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
        x_conv_normed = l2_normalize(x_conv.permute(0, 2, 3, 1), dim=2)  # Nx2xDx2
        x_fc = self.fc(x_conv_normed.flatten(1))
        x_fc_normed = l2_normalize(x_fc)  # Important!!
        score = torch.sum(torch.square(attr_hs - x_fc_normed), dim=1)
        return -score


class MDE(nn.Module):

    def __init__(self, num_vectors, kind='mean', gamma=12.):
        super(MDE, self).__init__()
        self.kind = kind
        self.gamma = gamma
        if kind == 'conv':
            self.mapping = nn.Conv2d(num_vectors, 1, kernel_size=1)
        elif kind == 'fc':
            self.mapping = nn.Linear(num_vectors, 1)

    def forward(self, heads, relations, tails):
        a = heads[0] + relations[0] - tails[0]
        b = heads[1] + tails[1] - relations[1]
        c = tails[2] + relations[2] - heads[2]
        d = heads[3] - relations[3] * tails[3]

        e = heads[4] + relations[4] - tails[4]
        f = heads[5] + tails[5] - relations[5]
        g = tails[6] + relations[6] - heads[6]
        i = heads[7] - relations[7] * tails[7]

        score_a = (torch.norm(a, p=2, dim=1) + torch.norm(e, p=2, dim=1)) / 2.0
        score_b = (torch.norm(b, p=2, dim=1) + torch.norm(f, p=2, dim=1)) / 2.0
        score_c = (torch.norm(c, p=2, dim=1) + torch.norm(g, p=2, dim=1)) / 2.0
        score_d = (torch.norm(d, p=2, dim=1) + torch.norm(i, p=2, dim=1)) / 2.0
        score = (1.5 * score_a + 3.0 * score_b + 1.5 * score_c + 3.0 * score_d) / 9.0
        # score = self.gamma - score
        return -score

    def lookup(self, input, index=None):
        outputs = torch.index_select(input, dim=1, index=index) if index is not None else input
        if outputs.size(0) == 1:
            output = outputs.squeeze(0)
        else:
            if self.kind == 'conv':
                output = self.mapping(outputs.unsqueeze(0)).squeeze(0).squeeze(1)
            elif self.kind == 'fc':
                output = self.mapping(torch.flatten(outputs, 1).T).view(outputs.size(1), outputs.size(2))
            else:
                output = torch.mean(outputs, dim=0)
        return output


class MultiKENet(nn.Module):

    def __init__(self, num_entities, num_relations, num_attributes, embed_dim, value_vectors, local_name_vectors, mode='transe', num_vectors=1, shared_space=False):
        super(MultiKENet, self).__init__()
        self.mode = mode
        self.num_vectors = num_vectors

        self.register_buffer('literal_embeds', torch.from_numpy(value_vectors), persistent=True)
        self.register_buffer('name_embeds', torch.from_numpy(local_name_vectors), persistent=True)

        # Relation view
        self.rv_ent_embeds = nn.Parameter(torch.Tensor(num_vectors, num_entities, embed_dim))
        self.rel_embeds = nn.Parameter(torch.Tensor(num_vectors, num_relations, embed_dim))
        self.embedding = MDE(num_vectors, 'mean') if mode == 'mde' else TransE()

        # Attribute view
        self.av_ent_embeds = nn.Parameter(torch.Tensor(num_entities, embed_dim))
        self.attr_embeds = nn.Parameter(torch.Tensor(num_attributes, embed_dim))  # False important!
        self.attr_embedding = ConvE(embed_dim)
        self.attr_triple_embedding = ConvE(embed_dim)
        self.attr_ref_embedding = ConvE(embed_dim)

        # Shared embeddings
        self.ent_embeds = nn.Parameter(torch.Tensor(num_entities, embed_dim))

        self.cfg = {
            # params, lookup
            'rv': [(self._parameters['rv_ent_embeds'], self._parameters['rel_embeds'], self.embedding), self.relation_triple_lookup],
            'av': [(self._parameters['av_ent_embeds'], self.attr_embeds, self.attr_embedding), self.attribute_triple_lookup],
            'ckgrtv': [(self._parameters['rv_ent_embeds'], self._parameters['rel_embeds'], self.embedding), self.cross_kg_relation_triple_lookup],
            'ckgatv': [(self._parameters['av_ent_embeds'], self.attr_embeds, self.attr_triple_embedding), self.cross_kg_attribute_triple_lookup],
            'ckgrrv': [(self._parameters['rv_ent_embeds'], self._parameters['rel_embeds'], self.embedding), self.cross_kg_relation_reference_lookup],
            'ckgarv': [(self._parameters['av_ent_embeds'], self.attr_embeds, self.attr_ref_embedding), self.cross_kg_attribute_reference_lookup],
            'cnv': [(self._parameters['ent_embeds'], self._parameters['rv_ent_embeds'], self._parameters['av_ent_embeds'], self.embedding), self.cross_name_view_lookup]
        }
        if shared_space:
            # Shared combination
            self.nv_mapping = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.rv_mapping = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.av_mapping = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.register_buffer('eye', torch.eye(embed_dim), persistent=False)

            self.cfg['mv'] = [(self._parameters['ent_embeds'], self.nv_mapping, self.rv_mapping, self.av_mapping), self.multi_view_entities_lookup]

        self._init_parameters()

    def __getattr__(self, name):
        attr = super().__getattr__(name)
        if name in ['rv_ent_embeds', 'rel_embeds', 'av_ent_embeds', 'ent_embeds']:
            return l2_normalize(attr, dim=-1)
        else:
            return attr

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'rv_ent_embeds' in name or 'rel_embeds' in name:
                for i in range(param.size(0)):
                    nn.init.xavier_normal_(param[i])
            elif '_embeds' in name:
                nn.init.xavier_normal_(param)
            elif '_mapping' in name:
                nn.init.orthogonal_(param)
            elif 'bn' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                else:
                    nn.init.xavier_uniform_(param)

    def parameters(self, view, recurse=True):
        params = []
        for param in self.cfg[view][0]:
            if isinstance(param, nn.Module):
                params += list(param.parameters(recurse=recurse))
            else:
                params.append(param)
        return params

    def relation_triple_lookup(self, rel_pos_hs, rel_pos_rs, rel_pos_ts, rel_neg_hs, rel_neg_rs, rel_neg_ts):
        rel_phs = torch.index_select(self.rv_ent_embeds, dim=1, index=rel_pos_hs)
        rel_prs = torch.index_select(self.rel_embeds, dim=1, index=rel_pos_rs)
        rel_pts = torch.index_select(self.rv_ent_embeds, dim=1, index=rel_pos_ts)
        rel_nhs = torch.index_select(self.rv_ent_embeds, dim=1, index=rel_neg_hs)
        rel_nrs = torch.index_select(self.rel_embeds, dim=1, index=rel_neg_rs)
        rel_nts = torch.index_select(self.rv_ent_embeds, dim=1, index=rel_neg_ts)
        pos_score = self.embedding(rel_phs, rel_prs, rel_pts)
        neg_score = self.embedding(rel_nhs, rel_nrs, rel_nts)
        return pos_score, neg_score

    def attribute_triple_lookup(self, attr_pos_hs, attr_pos_as, attr_pos_vs):
        attr_phs = torch.index_select(self.av_ent_embeds, dim=0, index=attr_pos_hs)
        attr_pas = torch.index_select(self.attr_embeds, dim=0, index=attr_pos_as)
        attr_pvs = torch.index_select(self.literal_embeds, dim=0, index=attr_pos_vs)
        pos_score = self.attr_embedding(attr_phs, attr_pas, attr_pvs)
        return pos_score,

    def cross_kg_relation_triple_lookup(self, ckge_rel_pos_hs, ckge_rel_pos_rs, ckge_rel_pos_ts):
        ckge_rel_phs = torch.index_select(self.rv_ent_embeds, dim=1, index=ckge_rel_pos_hs)
        ckge_rel_prs = torch.index_select(self.rel_embeds, dim=1, index=ckge_rel_pos_rs)
        ckge_rel_pts = torch.index_select(self.rv_ent_embeds, dim=1, index=ckge_rel_pos_ts)
        pos_score = self.embedding(ckge_rel_phs, ckge_rel_prs, ckge_rel_pts)
        return pos_score,

    def cross_kg_attribute_triple_lookup(self, ckge_attr_pos_hs, ckge_attr_pos_as, ckge_attr_pos_vs):
        ckge_attr_phs = torch.index_select(self.av_ent_embeds, dim=0, index=ckge_attr_pos_hs)
        ckge_attr_pas = torch.index_select(self.attr_embeds, dim=0, index=ckge_attr_pos_as)
        ckge_attr_pvs = torch.index_select(self.literal_embeds, dim=0, index=ckge_attr_pos_vs)
        pos_score = self.attr_triple_embedding(ckge_attr_phs, ckge_attr_pas, ckge_attr_pvs)
        return pos_score,

    def cross_kg_relation_reference_lookup(self, ckgp_rel_pos_hs, ckgp_rel_pos_rs, ckgp_rel_pos_ts):
        ckgp_rel_phs = torch.index_select(self.rv_ent_embeds, dim=1, index=ckgp_rel_pos_hs)
        ckgp_rel_prs = torch.index_select(self.rel_embeds, dim=1, index=ckgp_rel_pos_rs)
        ckgp_rel_pts = torch.index_select(self.rv_ent_embeds, dim=1, index=ckgp_rel_pos_ts)
        pos_score = self.embedding(ckgp_rel_phs, ckgp_rel_prs, ckgp_rel_pts)
        return pos_score,

    def cross_kg_attribute_reference_lookup(self, ckga_attr_pos_hs, ckga_attr_pos_as, ckga_attr_pos_vs):
        ckga_attr_phs = torch.index_select(self.av_ent_embeds, dim=0, index=ckga_attr_pos_hs)
        ckga_attr_pas = torch.index_select(self.attr_embeds, dim=0, index=ckga_attr_pos_as)
        ckga_attr_pvs = torch.index_select(self.literal_embeds, dim=0, index=ckga_attr_pos_vs)
        pos_score = self.attr_ref_embedding(ckga_attr_phs, ckga_attr_pas, ckga_attr_pvs)
        return pos_score,

    def cross_name_view_lookup(self, cn_hs):
        final_cn_phs = torch.index_select(self.ent_embeds, dim=0, index=cn_hs)
        cn_hs_names = torch.index_select(self.name_embeds, dim=0, index=cn_hs)
        cr_hs = self.embedding.lookup(self.rv_ent_embeds, index=cn_hs)
        ca_hs = torch.index_select(self.av_ent_embeds, dim=0, index=cn_hs)
        return final_cn_phs, cn_hs_names, cr_hs, ca_hs

    def multi_view_entities_lookup(self, entities):
        final_ents = torch.index_select(self.ent_embeds, dim=0, index=entities)
        nv_ents = torch.index_select(self.name_embeds, dim=0, index=entities)
        rv_ents = self.embedding.lookup(self.rv_ent_embeds, index=entities)
        av_ents = torch.index_select(self.av_ent_embeds, dim=0, index=entities)
        return final_ents, nv_ents, rv_ents, av_ents, self.nv_mapping, self.rv_mapping, self.av_mapping

    def forward(self, inputs, view):
        """
        Forward pass of the specified view to calculate the scores of a batch of data.

        Parameters
        ----------
        inputs: list
            Inputs consist of positive samples (and negative samples).
        view: str
            Name of view.

        Returns
        -------
        outputs : A list of predicted scores
        """
        return self.cfg[view][1](*inputs)

    @staticmethod
    @torch.no_grad()
    def predict(args, model, dataloader, kgs, top_k=1, min_sim_value=None, output_filename=None):
        """
        Compute pairwise similarity between the two collections of embeddings.

        Parameters
        ----------
        args: ARGs
            Arguments provided to run the experiment.
        model: MultiKENet
            MultiKE model.
        dataloader: DataLoader
            The dataloader to provide the data for prediction.
        kgs: KGs
            Instance of KGs class which provides the two kgs.
        top_k : int
            The k for top k retrieval, can be None (but then min_sim_value should be set).
        min_sim_value : float, optional
            the minimum value for the confidence.
        output_filename : str, optional
            The name of the output file. It is formatted as tsv file with entity1, entity2, confidence.

        Returns
        -------
        topk_neighbors_w_sim : A list of tuples of form (entity1, entity2, confidence)
        """
        model.eval()
        embeds = []
        for entities in dataloader:
            entities = entities.long().to(model.ent_embeds.device)
            embeds.append(torch.index_select(model.ent_embeds, dim=0, index=entities).cpu())

        embeds = torch.cat(embeds, dim=0).numpy()
        num_kg1_ents = len(dataloader.dataset.kg1)
        embeds1 = embeds[:num_kg1_ents]
        embeds2 = embeds[num_kg1_ents:]

        sim_mat = sim(embeds1, embeds2, args.eval_metric, args.eval_norm, csls_k=0)

        # search for correspondences which match top_k and/or min_sim_value
        matched_entities_indexes = set()
        if top_k:
            assert top_k > 0
            # top k for entities in kg1
            for i in range(sim_mat.shape[0]):
                for rank_index in np.argpartition(-sim_mat[i, :], top_k)[:top_k]:
                    matched_entities_indexes.add((i, rank_index))

            # top k for entities in kg2
            for i in range(sim_mat.shape[1]):
                for rank_index in np.argpartition(-sim_mat[:, i], top_k)[:top_k]:
                    matched_entities_indexes.add((rank_index, i))

            if min_sim_value:
                matched_entities_indexes.intersection(map(tuple, np.argwhere(sim_mat > min_sim_value)))
        elif min_sim_value:
            matched_entities_indexes = set(map(tuple, np.argwhere(sim_mat > min_sim_value)))
        else:
            raise ValueError("Either top_k or min_sim_value should have a value")

        # build id to URI map
        kg1_id_to_uri = {v: k for k, v in kgs.kg1.entities_id_dict.items()}
        kg2_id_to_uri = {v: k for k, v in kgs.kg2.entities_id_dict.items()}

        topk_neighbors_w_sim = [(kg1_id_to_uri[kgs.kg1.entities_list[i]], kg2_id_to_uri[kgs.kg2.entities_list[j]],
                                 sim_mat[i, j]) for i, j in matched_entities_indexes]

        if output_filename is not None:
            with open(output_filename, 'w', encoding='utf8') as file:
                for entity1, entity2, confidence in topk_neighbors_w_sim:
                    file.write(str(entity1) + "\t" + str(entity2) + "\t" + str(confidence) + "\n")
            print(output_filename, "saved")
        return topk_neighbors_w_sim

    @staticmethod
    @torch.no_grad()
    def predict_entities(args, model, entities_file_path, kgs, output_filename=None):
        """
        Compute the confidence of given entities if they match or not.

        Parameters
        ----------
        args: ARGs
            Arguments provided to run the experiment.
        model: MultiKENet
            MultiKE model.
        entities_file_path : str
            A path pointing to a file formatted as (entity1, entity2) with tab separated (tsv-file).
            If given, the similarity of the entities is retrieved and returned (or also written to file if output_file_name is given).
            The parameters top_k and min_sim_value do not play a role, if this parameter is set.
        kgs: KGs
            Instance of KGs class which provides the two kgs.
        output_filename : str, optional
            The name of the output file. It is formatted as tsv file with entity1, entity2, confidence.

        Returns
        -------
        topk_neighbors_w_sim : A list of tuples of form (entity1, entity2, confidence)
        """
        model.eval()
        kg1_entities = []
        kg2_entities = []
        with open(entities_file_path, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                entities = line.strip('\n').split('\t')
                kg1_entities.append(kgs.kg1.entities_id_dict[entities[0]])
                kg2_entities.append(kgs.kg2.entities_id_dict[entities[1]])
        kg1_distinct_entities = list(set(kg1_entities))
        kg2_distinct_entities = list(set(kg2_entities))

        kg1_mapping = {entity_id : index for index, entity_id in enumerate(kg1_distinct_entities)}
        kg2_mapping = {entity_id : index for index, entity_id in enumerate(kg2_distinct_entities)}

        dataset = TestDataset(kg1_distinct_entities, kg2_distinct_entities)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=args.pin_memory)
        embeds = []
        for entities in dataloader:
            entities = entities.long().to(model.ent_embeds.device)
            embeds.append(torch.index_select(model.ent_embeds, dim=0, index=entities).cpu())

        embeds = torch.cat(embeds, dim=0).numpy()
        num_kg1_ents = len(kg1_distinct_entities)
        embeds1 = embeds[:num_kg1_ents]
        embeds2 = embeds[num_kg1_ents:]

        sim_mat = sim(embeds1, embeds2, args.eval_metric, args.eval_norm, csls_k=0)

        # map back with entities_id_dict to be sure that the right uri is chosen
        kg1_id_to_uri = {v: k for k, v in kgs.kg1.entities_id_dict.items()}
        kg2_id_to_uri = {v: k for k, v in kgs.kg2.entities_id_dict.items()}

        topk_neighbors_w_sim = []
        for entity1_id, entity2_id in zip(kg1_entities, kg2_entities):
            topk_neighbors_w_sim.append((
                kg1_id_to_uri[entity1_id],
                kg2_id_to_uri[entity2_id],
                sim_mat[kg1_mapping[entity1_id], kg2_mapping[entity2_id]]
            ))

        if output_filename is not None:
            with open(output_filename, 'w', encoding='utf8') as file:
                for entity1, entity2, confidence in topk_neighbors_w_sim:
                    file.write(str(entity1) + "\t" + str(entity2) + "\t" + str(confidence) + "\n")
            print(output_filename, "saved")
        return topk_neighbors_w_sim

    @staticmethod
    @torch.no_grad()
    def embeds(model, dataloader, embed_choice='rv', w=(1, 1, 1)):
        """
        Get the embeddings for the provided data.

        Parameters
        ----------
        model: MultiKENet
            MultiKE model.
        dataloader: DataLoader
            Dataloader that provides the data needs to be embedded.
        embed_choice: str
            Embedding type.
        w: tuple, optional
            Weights for avg choice.

        Returns
        -------
        embeds : Embedded results
        """
        model.eval()
        if embed_choice == 'nv':
            ent_embeds = model.name_embeds
        elif embed_choice == 'rv':
            ent_embeds = model.rv_ent_embeds
        elif embed_choice == 'av':
            ent_embeds = model.av_ent_embeds
        elif embed_choice == 'final':
            ent_embeds = model.ent_embeds
        # elif embed_choice == 'avg':
        #     ent_embeds = w[0] * model.name_embeds + w[1] * model.rv_ent_embeds + w[2] * model.av_ent_embeds
        else:  # 'final'
            ent_embeds = model.ent_embeds
        embeds = []
        for entities, in dataloader:
            entities = entities.long().to(ent_embeds.device)
            if embed_choice == 'rv':
                embeds.append(model.embedding.lookup(model.rv_ent_embeds, entities).cpu())
            else:
                embeds.append(torch.index_select(ent_embeds, dim=0, index=entities).cpu())

        embeds = torch.cat(embeds, dim=0).numpy()
        return embeds

    @staticmethod
    @torch.no_grad()
    def test(args, model, dataloader, embed_choice='rv', w=(1, 1, 1), accurate=False):
        """
        Test the model.

        Parameters
        ----------
        args: ARGs
            Arguments provided to run the experiment.
        model: MultiKENet
            MultiKE model.
        dataloader: DataLoader
            Dataloader that provides test data.
        embed_choice: str, optional
            Embedding type.
        w: tuple, optional
            Weights for avg choice.
        accurate: bool, optional
            Flag to get accurate result.

        Returns
        -------
        metric : The calculated metric
        """
        model.eval()
        if embed_choice == 'nv':
            ent_embeds = model.name_embeds
        elif embed_choice == 'rv':
            ent_embeds = model.rv_ent_embeds
        elif embed_choice == 'av':
            ent_embeds = model.av_ent_embeds
        elif embed_choice == 'final':
            ent_embeds = model.ent_embeds
        # elif embed_choice == 'avg':
        #     ent_embeds = w[0] * model.name_embeds + w[1] * model.rv_ent_embeds + w[2] * model.av_ent_embeds
        else:  # 'final'
            ent_embeds = model.ent_embeds
        print(embed_choice, "test results:")
        embeds = []
        for entities in dataloader:
            entities = entities.long().to(ent_embeds.device)
            if embed_choice == 'rv':
                embeds.append(model.embedding.lookup(model.rv_ent_embeds, entities).cpu())
            else:
                embeds.append(torch.index_select(ent_embeds, dim=0, index=entities).cpu())

        embeds = torch.cat(embeds, dim=0).numpy()
        num_kg1_ents = len(dataloader.dataset.kg1)
        embeds1 = embeds[:num_kg1_ents]
        embeds2 = embeds[num_kg1_ents:]
        _, hits1_12, mrr_12 = test(embeds1, embeds2, None, args.top_k, args.num_test_workers, args.eval_metric, args.eval_norm, args.csls, accurate)
        return mrr_12 if args.stop_metric == 'mrr' else hits1_12

    @staticmethod
    @torch.no_grad()
    def test_wva(args, model, dataloader, accurate=False):
        """
        Perform WVA model testing.

        Parameters
        ----------
        args
            Arguments provided to run the experiment.
        model
            MultiKE model.
        dataloader
            Dataloader that provides the data.
        accurate, optional
            Flag to get accurate result.

        Returns
        -------
        metric : The calculated metric
        """
        model.eval()
        nv_ent_embeds, rv_ent_embeds, av_ent_embeds = [], [], []
        for entities in dataloader:
            entities = entities.long().to(model.ent_embeds.device)
            nv_ent_embeds.append(torch.index_select(model.name_embeds, dim=0, index=entities).cpu())
            rv_ent_embeds.append(model.embedding.lookup(model.rv_ent_embeds, entities).cpu())
            av_ent_embeds.append(torch.index_select(model.av_ent_embeds, dim=0, index=entities).cpu())

        nv_ent_embeds = torch.cat(nv_ent_embeds, dim=0).numpy()
        rv_ent_embeds = torch.cat(rv_ent_embeds, dim=0).numpy()
        av_ent_embeds = torch.cat(av_ent_embeds, dim=0).numpy()
        num_kg1_ents = len(dataloader.dataset.kg1)
        nv_ent_embeds1 = nv_ent_embeds[:num_kg1_ents]
        nv_ent_embeds2 = nv_ent_embeds[num_kg1_ents:]
        rv_ent_embeds1 = rv_ent_embeds[:num_kg1_ents]
        rv_ent_embeds2 = rv_ent_embeds[num_kg1_ents:]
        av_ent_embeds1 = av_ent_embeds[:num_kg1_ents]
        av_ent_embeds2 = av_ent_embeds[num_kg1_ents:]
        weight11, weight21, weight31 = wva(nv_ent_embeds1, rv_ent_embeds1, av_ent_embeds1)
        weight12, weight22, weight32 = wva(nv_ent_embeds2, rv_ent_embeds2, av_ent_embeds2)

        weight1 = weight11 + weight12
        weight2 = weight21 + weight22
        weight3 = weight31 + weight32
        all_weight = weight1 + weight2 + weight3
        weight1 /= all_weight
        weight2 /= all_weight
        weight3 /= all_weight

        print("wva test results:")
        print("weights", weight1, weight2, weight3)

        embeds1 = weight1 * nv_ent_embeds1 + weight2 * rv_ent_embeds1 + weight3 * av_ent_embeds1
        embeds2 = weight1 * nv_ent_embeds2 + weight2 * rv_ent_embeds2 + weight3 * av_ent_embeds2

        _, hits1_12, mrr_12 = test(embeds1, embeds2, None, args.top_k, args.num_test_workers, args.eval_metric, args.eval_norm, args.csls, accurate)
        return mrr_12 if args.stop_metric == 'mrr' else hits1_12
