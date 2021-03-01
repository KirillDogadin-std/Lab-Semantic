import gc
import os
import time
import random
import multiprocessing

import numpy as np
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from pytorch.load.kgs import read_kgs
from pytorch.predicate_alignment import PredicateAlignModel
from pytorch.literal_encoder import encode_literals
from pytorch.utils import task_divide, merge_dic, l2_normalize


def read_local_name(file_path, entities_set):
    print("read local names from", file_path)
    entity_local_name = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split('\t')
            assert len(line) == 2
            ln = line[1]
            if ln.endswith(')'):
                ln = ln.split('(')[0]
            entity_local_name[line[0]] = ln.replace('_', ' ')

    for e in entities_set:
        if e not in entity_local_name:
            entity_local_name[e] = ''
    assert len(entity_local_name) == len(entities_set)
    return entity_local_name


def clear_attribute_triples(attribute_triples):
    print("before clear:", len(attribute_triples))
    # step 1
    attribute_triples_new = set()
    num_attrs = {}
    for (e, a, _) in attribute_triples:
        num_ents = 1
        if a in num_attrs:
            num_ents += num_attrs[a]
        num_attrs[a] = num_ents
    attr_set = set(num_attrs.keys())
    attr_set_new = set()
    for a in attr_set:
        if num_attrs[a] >= 10:
            attr_set_new.add(a)
    for (e, a, v) in attribute_triples:
        if a in attr_set_new:
            attribute_triples_new.add((e, a, v))
    attribute_triples = attribute_triples_new
    print("after step 1:", len(attribute_triples))

    # step 2
    attribute_triples_new = []
    number_literals, literals_string = [], []
    for (e, a, v) in attribute_triples:
        v = v.strip('"')
        if '"^^' in v:
            v = v[:v.index('"^^')]
        if v.endswith('"@en'):
            v = v[:v.index('"@en')]
        if v.endswith('"@eng'):
            v = v[:v.index('"@eng')]
        if is_number(v):
            number_literals.append(v)
        else:
            literals_string.append(v)
        v = v.replace('.', '').replace('(', '').replace(')', '').replace(',', '').replace('"', '')
        v = v.replace('_', ' ').replace('-', ' ').replace('/', ' ')
        if 'http' in v:
            continue
        attribute_triples_new.append((e, a, v))
    attribute_triples = attribute_triples_new
    print("after step 2:", len(attribute_triples))
    return attribute_triples, number_literals, literals_string


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def generate_sup_attribute_triples(sup_links, av_dict1, av_dict2):
    def generate_sup_attribute_triples_one_link(e1, e2, av_dict):
        new_triples = set()
        for a, v in av_dict.get(e1, set()):
            new_triples.add((e2, a, v))
        return new_triples
    new_triples1, new_triples2 = set(), set()
    for ent1, ent2 in sup_links:
        new_triples1 |= (generate_sup_attribute_triples_one_link(ent1, ent2, av_dict1))
        new_triples2 |= (generate_sup_attribute_triples_one_link(ent2, ent1, av_dict2))
    print("supervised attribute triples: {}, {}".format(len(new_triples1), len(new_triples2)))
    return new_triples1, new_triples2


def generate_dict(literal_list, literal_vectors_list):
    assert len(literal_list) == len(literal_vectors_list)
    dic = {}
    for i in range(len(literal_list)):
        dic[literal_list[i]] = literal_vectors_list[i]
    return dic


def generate_literal_id_dic(literal_list):
    literal_id_dic = {}
    print("literal id", len(literal_list), len(set(literal_list)))
    for i in range(len(literal_list)):
        literal_id_dic[literal_list[i]] = i
    assert len(literal_list) == len(literal_id_dic)
    return literal_id_dic


def find_neighbours(frags, entity_list, sub_embed, embed, k):
    """
    Find neighbours by entity fragments

    Parameters
    ----------
    frags
        Fragments of entity list.
    entity_list
        Entity list.
    sub_embed
        Embeddings corresponding to fragments.
    embed
        Embeddings.
    k
        Number of neighbours.
    """
    dic = {}
    sim_mat = np.matmul(sub_embed, embed.T)
    for i in range(sim_mat.shape[0]):
        sort_index = np.argpartition(-sim_mat[i, :], k)
        neighbors_index = sort_index[0:k]
        neighbors = entity_list[neighbors_index]
        dic[frags[i]] = neighbors
    return dic


def _generate_neighbours(entity_embeds, entity_list, num_neighbors, num_threads):
    ent_frags = task_divide(entity_list, num_threads)
    ent_frag_indexes = task_divide(np.arange(0, entity_list.shape[0]), num_threads)

    pool = multiprocessing.Pool(processes=len(ent_frags))
    results = []
    for i in range(len(ent_frags)):
        results.append(pool.apply_async(find_neighbours, args=(ent_frags[i], entity_list, entity_embeds[ent_frag_indexes[i], :], entity_embeds, num_neighbors)))
    pool.close()
    pool.join()

    dic = {}
    for res in results:
        dic = merge_dic(dic, res.get())

    del results
    gc.collect()
    return dic


def _generate_neighbours_single_thread(entity_embeds, entity_list, num_neighbors, num_threads):
    ent_frags = task_divide(entity_list, num_threads)
    ent_frag_indexes = task_divide(np.arange(0, entity_list.shape[0]), num_threads)
    results = {}
    for i in range(len(ent_frags)):
        dic = find_neighbours(ent_frags[i], np.array(entity_list), entity_embeds[ent_frag_indexes[i], :], entity_embeds, num_neighbors)
        results = merge_dic(results, dic)
    return results


class DataModel:

    def __init__(self, args):
        self.args = args
        self.kgs = read_kgs(args.dataset, args.dataset_division, args.alignment_module, args.ordered, False)
        if os.path.exists(os.path.join(args.dataset, 'entity_local_name_1')) and os.path.exists(os.path.join(args.dataset, 'entity_local_name_2')):
            entity_local_name = read_local_name(os.path.join(args.dataset, 'entity_local_name_1'), set(self.kgs.kg1.entities_id_dict.keys()))
            entity_local_name.update(read_local_name(os.path.join(args.dataset, 'entity_local_name_2'), set(self.kgs.kg2.entities_id_dict.keys())))
            print("total local names:", len(entity_local_name))
            entity_local_name_dict = entity_local_name
        else:
            entity_local_name_dict = self._get_local_name_by_name_triple()
        self.local_name_vectors, self.value_vectors = self._generate_literal_vectors(entity_local_name_dict)
        self.predicate_align = PredicateAlignModel(args, self.kgs)
        self.neighbors1, self.neighbors2 = {}, {}
        self.dataloader1 = DataLoader(TensorDataset(torch.from_numpy(self.kgs.all_entities[:, 0])), self.args.batch_size,
                                shuffle=False, num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)
        self.dataloader2 = DataLoader(TensorDataset(torch.from_numpy(self.kgs.all_entities[:, 1])), self.args.batch_size,
                                shuffle=False, num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)

    def _get_local_name_by_name_triple(self, name_attribute_list=None):
        dataset = os.path.basename(os.path.dirname(self.args.dataset + '/'))
        if name_attribute_list is None:
            if 'D_Y' in dataset:
                name_attribute_list = {'skos:prefLabel', 'http://dbpedia.org/ontology/birthName'}
            elif 'D_W' in dataset:
                name_attribute_list = {'http://www.wikidata.org/entity/P373', 'http://www.wikidata.org/entity/P1476'}
            else:
                name_attribute_list = {}

        local_triples = self.kgs.kg1.local_attribute_triples_set | self.kgs.kg2.local_attribute_triples_set
        triples = []
        for h, a, v in local_triples:
            v = v.strip('"')
            if v.endswith('"@eng'):
                v = v.rstrip('"@eng')
            triples.append((h, a, v))
        id_ent_dict = {}
        for e, e_id in self.kgs.kg1.entities_id_dict.items():
            id_ent_dict[e_id] = e
        for e, e_id in self.kgs.kg2.entities_id_dict.items():
            id_ent_dict[e_id] = e
        print(len(id_ent_dict))

        name_ids = set()
        for a, a_id in self.kgs.kg1.attributes_id_dict.items():
            if a in name_attribute_list:
                name_ids.add(a_id)
        for a, a_id in self.kgs.kg2.attributes_id_dict.items():
            if a in name_attribute_list:
                name_ids.add(a_id)

        for a, a_id in self.kgs.kg1.attributes_id_dict.items():
            if a_id in name_ids:
                print(a)
        for a, a_id in self.kgs.kg2.attributes_id_dict.items():
            if a_id in name_ids:
                print(a)
        print(name_ids)

        local_name_dict = {}
        ents = self.kgs.kg1.entities_set | self.kgs.kg2.entities_set
        print(len(ents))
        for (e, a, v) in triples:
            if a in name_ids:
                local_name_dict[id_ent_dict[e]] = v
        print("after name_ids:", len(local_name_dict))
        for e in ents:
            if id_ent_dict[e] not in local_name_dict:
                local_name_dict[id_ent_dict[e]] = id_ent_dict[e].split('/')[-1].replace('_', ' ')
        print("total local names:", len(local_name_dict))
        return local_name_dict

    def _generate_literal_vectors(self, entity_local_name_dict):
        cleaned_attribute_triples_list1, _, _ = clear_attribute_triples(self.kgs.kg1.local_attribute_triples_list)
        cleaned_attribute_triples_list2, _, _ = clear_attribute_triples(self.kgs.kg2.local_attribute_triples_list)
        value_list = [v for (_, _, v) in cleaned_attribute_triples_list1 + cleaned_attribute_triples_list2]
        local_name_list = list(entity_local_name_dict.values())
        literal_list = list(set(value_list + local_name_list))
        print("literal nums:", len(local_name_list), len(value_list), len(literal_list))
        literal_vectors = encode_literals(self.args, literal_list)
        assert literal_vectors.shape[0] == len(literal_list)
        literal_id_dic = generate_literal_id_dic(literal_list)

        name_ordered_list = []
        print("total entities:", self.kgs.num_entities)
        entity_id_uris_dic = dict(zip(self.kgs.kg1.entities_id_dict.values(), self.kgs.kg1.entities_id_dict.keys()))
        entity_id_uris_dic2 = dict(zip(self.kgs.kg2.entities_id_dict.values(), self.kgs.kg2.entities_id_dict.keys()))
        entity_id_uris_dic.update(entity_id_uris_dic2)
        print("total entities ids:", len(entity_id_uris_dic))
        assert len(entity_id_uris_dic) == self.kgs.num_entities
        for i in range(self.kgs.num_entities):
            assert i in entity_id_uris_dic
            entity_uri = entity_id_uris_dic.get(i)
            assert entity_uri in entity_local_name_dict
            entity_name = entity_local_name_dict.get(entity_uri)
            entity_name_index = literal_id_dic.get(entity_name)
            name_ordered_list.append(entity_name_index)
        print("name_ordered_list", len(name_ordered_list))
        name_vectors = literal_vectors[name_ordered_list, ]
        print("entity name embeddings:", type(name_vectors), name_vectors.shape)

        literal_set, values_set = set(literal_list), set()
        attribute_triples_list1, attribute_triples_list2 = set(), set()
        for h, a, v in cleaned_attribute_triples_list1:
            if v in literal_set:
                values_set.add(v)
                attribute_triples_list1.add((h, a, v))

        for h, a, v in cleaned_attribute_triples_list2:
            if v in literal_set:
                values_set.add(v)
                attribute_triples_list2.add((h, a, v))
        print("selected attribute triples", len(attribute_triples_list1), len(attribute_triples_list2))
        values_id_dic = {}
        values_list = list(values_set)
        for i in range(len(values_list)):
            values_id_dic[values_list[i]] = i
        id_attribute_triples1 = set([(h, a, int(values_id_dic[v])) for (h, a, v) in attribute_triples_list1])
        id_attribute_triples2 = set([(h, a, int(values_id_dic[v])) for (h, a, v) in attribute_triples_list2])
        self.kgs.kg1.set_attributes(id_attribute_triples1)
        self.kgs.kg2.set_attributes(id_attribute_triples2)
        sup_triples1, sup_triples2 = generate_sup_attribute_triples(self.kgs.get_entities('train'),
                                                                    self.kgs.kg1.av_dict, self.kgs.kg2.av_dict)
        self.kgs.kg1.add_sup_attribute_triples(sup_triples1)
        self.kgs.kg2.add_sup_attribute_triples(sup_triples2)
        value_ordered_list = []
        for i in range(len(values_id_dic)):
            value = values_list[i]
            value_index = literal_id_dic.get(value)
            value_ordered_list.append(value_index)
        print("value_ordered_list", len(value_ordered_list))
        value_vectors = literal_vectors[value_ordered_list, ]
        print("value embeddings:", type(value_vectors), value_vectors.shape)
        if self.args.literal_normalize:
            name_vectors = preprocessing.normalize(name_vectors)
            value_vectors = preprocessing.normalize(value_vectors)
        return name_vectors, value_vectors

    @torch.no_grad()
    def update_predicate_alignment(self, model):
        rel_embeds = model.embedding.lookup(model.rel_embeds).cpu().numpy()
        self.predicate_align.update_predicate_alignment(rel_embeds)
        attr_embeds = model.attr_embeds.cpu().numpy()
        self.predicate_align.update_predicate_alignment(attr_embeds, predicate_type='attribute')

    def generate_neighbours(self, model, truncated_epsilon):
        start_time = time.time()
        num_neighbors1 = int((1 - truncated_epsilon) * self.kgs.kg1.num_entities)
        num_neighbors2 = int((1 - truncated_epsilon) * self.kgs.kg2.num_entities)

        entity_embeds1 = model.embeds(model, self.dataloader1)
        self.neighbors1 = _generate_neighbours(entity_embeds1, self.kgs.all_entities[:, 0], num_neighbors1, self.args.num_workers)
        entity_embeds2 = model.embeds(model, self.dataloader2)
        self.neighbors2 = _generate_neighbours(entity_embeds2, self.kgs.all_entities[:, 1], num_neighbors2, self.args.num_workers)
        num_ents = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
        end_time = time.time()
        print("neighbor dict:", len(self.neighbors1), type(self.neighbors2))
        print("generating neighbors of {} entities costs {:.3f} s.".format(num_ents, end_time - start_time))


class TrainDataset(Dataset):

    def __init__(self, data_model, batch_size, view, num_neg_triples=0):
        super(TrainDataset, self).__init__()
        self.data_model = data_model
        self.batch_size = batch_size
        self.view = view
        self.num_neg_triples = num_neg_triples
        self.regenerate()

    def regenerate(self):
        if self.view == 'ckgrtv':
            self.kg1 = self.data_model.kgs.kg1.sup_relation_triples_list
            self.kg2 = self.data_model.kgs.kg2.sup_relation_triples_list
        elif self.view == 'ckgatv':
            self.kg1 = self.data_model.kgs.kg1.sup_attribute_triples_list
            self.kg2 = self.data_model.kgs.kg2.sup_attribute_triples_list
        elif self.view in ['cnv', 'mv']:
            self.kg1 = self.data_model.kgs.kg1.entities_list
            self.kg2 = self.data_model.kgs.kg2.entities_list
        elif self.view == 'ckgrrv':
            self.kg1 = self.data_model.predicate_align.sup_relation_alignment_triples1
            self.kg2 = self.data_model.predicate_align.sup_relation_alignment_triples2
        elif self.view == 'ckgarv':
            self.kg1 = self.data_model.predicate_align.sup_attribute_alignment_triples1
            self.kg2 = self.data_model.predicate_align.sup_attribute_alignment_triples2
        elif self.view == 'rv':
            self.kg1 = self.data_model.kgs.kg1.local_relation_triples_list
            self.kg2 = self.data_model.kgs.kg2.local_relation_triples_list
        elif self.view == 'av':
            self.kg1 = self.data_model.predicate_align.attribute_triples_w_weights1
            self.kg2 = self.data_model.predicate_align.attribute_triples_w_weights2

        if self.view not in ['rv', 'av']:
            total = len(self.kg1) + len(self.kg2)
            steps = int(total / self.batch_size)
            self.indices = [idx for _ in range(steps) for idx in random.sample(range(total), self.batch_size)]
            self.indices += [idx for idx in random.sample(range(total), total - (steps * self.batch_size))]
        else:
            kg1_len = len(self.kg1)
            kg2_len = len(self.kg2)
            total = kg1_len + kg2_len
            steps = int(total / self.batch_size)
            batch_size1 = int(kg1_len / total * self.batch_size)
            batch_size2 = self.batch_size - batch_size1
            kg1_indices = list(range(kg1_len))
            kg2_indices = list(range(kg1_len, total))
            random.shuffle(kg1_indices)
            random.shuffle(kg2_indices)
            self.indices = []
            for i in range(steps):
                self.indices += kg1_indices[i * batch_size1:(i + 1) * batch_size1]
                self.indices += kg2_indices[i * batch_size2:(i + 1) * batch_size2]
            self.indices += kg1_indices[steps * batch_size1:kg1_len]
            self.indices += kg2_indices[steps * batch_size2:kg2_len]

    def get_negs_fast(self, pos, triples_set, entities, neighbors, max_try=10):
        head, relation, tail = pos
        neg_triples = []
        nums_to_sample = self.num_neg_triples
        head_candidates = neighbors[head].tolist() if head in neighbors else entities
        tail_candidates = neighbors[tail].tolist() if tail in neighbors else entities
        for i in range(max_try):
            corrupt_head_prob = np.random.binomial(1, 0.5)
            if corrupt_head_prob:
                neg_heads = random.sample(head_candidates, nums_to_sample)
                i_neg_triples = {(h2, relation, tail) for h2 in neg_heads}
            else:
                neg_tails = random.sample(tail_candidates, nums_to_sample)
                i_neg_triples = {(head, relation, t2) for t2 in neg_tails}
            if i == max_try - 1:
                neg_triples += list(i_neg_triples)
                break
            else:
                i_neg_triples = list(i_neg_triples - triples_set)
                neg_triples += i_neg_triples
            if len(neg_triples) == self.num_neg_triples:
                break
            else:
                nums_to_sample = self.num_neg_triples - len(neg_triples)
        assert len(neg_triples) == self.num_neg_triples
        return neg_triples

    def get_negs(self, pos, triples_set, entities, neighbors, max_try=10):
        neg_triples = []
        head, relation, tail = pos
        head_candidates = neighbors[head].tolist() if head in neighbors else entities
        tail_candidates = neighbors[tail].tolist() if tail in neighbors else entities
        for i in range(self.num_neg_triples):
            n = 0
            while True:
                corrupt_head_prob = np.random.binomial(1, 0.5)
                neg_head = head
                neg_tail = tail
                if corrupt_head_prob:
                    neg_head = random.choice(head_candidates)
                else:
                    neg_tail = random.choice(tail_candidates)
                if (neg_head, relation, neg_tail) not in triples_set:
                    neg_triples.append((neg_head, relation, neg_tail))
                    break
                n += 1
                if n == max_try:
                    neg_tail = random.choice(entities)
                    neg_triples.append((head, relation, neg_tail))
                    break
        assert len(neg_triples) == self.num_neg_triples
        return neg_triples

    def gen_negs_attr(self, pos, triples_set, entities, neighbors):
        neg_triples = []
        head, attribute, value, w = pos
        candidates = neighbors[head].tolist() if head in neighbors else entities
        for i in range(self.num_neg_triples):
            while True:
                neg_head = random.choice(candidates)
                if (neg_head, attribute, value, w) not in triples_set:
                    break
            neg_triples.append((neg_head, attribute, value, w))
        assert len(neg_triples) == self.num_neg_triples
        return neg_triples

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        kg1_len = len(self.kg1)
        pos = self.kg2[idx - kg1_len] if idx >= kg1_len else self.kg1[idx]
        if self.view in ['rv', 'av']:
            neighbors = self.data_model.neighbors2 if idx >= kg1_len else self.data_model.neighbors1
            kg = self.data_model.kgs.kg2 if idx >= kg1_len else self.data_model.kgs.kg1
            if self.view == 'rv':
                negs = self.get_negs_fast(pos, kg.local_relation_triples_set, kg.entities_list, neighbors)
                nhs = [x[0] for x in negs]
                nrs = [x[1] for x in negs]
                nts = [x[2] for x in negs]
                return list(pos) + [nhs, nrs, nts], []
            # else:
            #     triples_set = self.data_model.predicate_align.attribute_triples_w_weights_set1 if idx >= kg1_len \
            #         else self.data_model.predicate_align.attribute_triples_w_weights_set2
            #     negs = self.gen_negs_attr(pos, triples_set, kg.entities_list, neighbors)

        inputs = pos[:3] if self.view not in ['cnv', 'mv'] else [pos]
        weights = []
        if self.view in ['ckgarv', 'ckgrrv', 'av']:
            weights.append(pos[3])
        return inputs, weights


class TestDataset(Dataset):

    def __init__(self, kg1_entities, kg2_entities):
        super(TestDataset, self).__init__()
        self.kg1 = kg1_entities
        self.kg2 = kg2_entities

    def __len__(self):
        return len(self.kg1) + len(self.kg2)

    def __getitem__(self, index):
        kg1_len = len(self.kg1)
        inputs = self.kg2[index - kg1_len] if index >= kg1_len else self.kg1[index]
        return inputs
