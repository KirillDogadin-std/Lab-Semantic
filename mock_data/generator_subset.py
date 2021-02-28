import numpy as np
import random as rnd
import re

### Editable block {
DEPTH = 3
path_base = '../../dataset_lab/BootEA_DBP_WD_100K/'
ent_name = [path_base + 'entity_local_name_1', path_base + 'entity_local_name_2']
pred_name = [path_base + 'predicate_local_name_1', path_base + 'predicate_local_name_2']
ent_links = path_base + 'ent_links'
attr_triples = [path_base + 'attr_triples_1', path_base + 'attr_triples_2']
rel_triples = [path_base + 'rel_triples_1', path_base + 'rel_triples_2']

### }

def grab_ids_and_rels(file_path):
    """
    Get the lines of the relation triples from file

    Returns
    -------
    ids of items to participate in the relations,
    relation lines
    predicate ids
    """

    _ids = set()
    rels = set()
    preds = set()
    no_further_rels = set()

    with open(file_path, 'r') as opened:
        lines = opened.readlines()

    sampled_core = rnd.sample(lines, 1)[0]
    core_id = sampled_core.split('\t')[0]
    _ids.add(core_id)

    for iterator in range(DEPTH):
        print(len(no_further_rels))
        for line in lines:
            line_id = line.split('\t')[0]
            if line_id in _ids:
                rels.add(line)
                new_id = line.split('\t')[2].strip('\n')
                _ids.add(new_id)
                preds.add(line.split('\t')[1])
                if iterator == (DEPTH - 1):
                    no_further_rels.add(line.split('\t')[2].strip('\n'))

    return _ids, rels, preds, no_further_rels


def grab_mapping(file_path, _ids, no_further_rels):
    with open(file_path, 'r') as opened:
        lines = opened.readlines()

    mapped = set()
    mapping = set()
    nfr = set()

    for line in lines:
        mapped_from = line.split('\t')[0]
        if mapped_from in _ids:
            mapped.add(line.split('\t')[1].strip('\n'))
            mapping.add(line)
        
        if mapped_from in no_further_rels:
            nfr.add(mapped_from)

    return mapped, mapping, nfr


def grab_attrs(file_path, _ids):

    ret = set()

    with open(file_path, 'r') as opened:
        lines = opened.readlines()

    for line in lines:
        if line.split('\t')[0] in _ids:
            ret.add(line)

    return ret


def grab_ids_by_rels(file_path, _ids, no_further_rels):
    """
    grab relations and preds from 2nd file by known ids
    """
    ret = set()
    preds = set()
    with open(file_path, 'r') as opened:
        lines = opened.readlines()

    for line in lines:
        _from = line.split('\t')[0]
        if _from in _ids and _from not in no_further_rels:
            ret.add(line)
            preds.add(line.split('\t')[1])

    return ret, preds


def predicate_name_by_id(file_path, ids):

    ret = set()

    with open(file_path, 'r') as opened:
        lines = opened.readlines()

    for line in lines:
        if line.split('\t')[0] in ids:
            ret.add(line)

    return ret


def ent_name_by_id(file_path, ids):

    ret = set()

    with open(file_path, 'r') as opened:
        lines = opened.readlines()

    for line in lines:
        if line.split('\t')[0] in ids:
            ret.add(line)

    return ret


rel_ids1, rel_lines1, pred_ids1, no_further_rels1 = grab_ids_and_rels(rel_triples[0])
rel_ids2, links_lines, no_further_rels2 = grab_mapping(ent_links, rel_ids1, no_further_rels1)
attr_lines1 = grab_attrs(attr_triples[0], rel_ids1)
attr_lines2 = grab_attrs(attr_triples[1], rel_ids2)
rel_lines2, pred_ids2 = grab_ids_by_rels(rel_triples[1], rel_ids2, no_further_rels2)

pred_names1 = predicate_name_by_id(pred_name[0], pred_ids1)
pred_names2 = predicate_name_by_id(pred_name[1], pred_ids2)

ent_names1 = ent_name_by_id(ent_name[0], rel_ids1) 
ent_names2 = ent_name_by_id(ent_name[1], rel_ids2) 

with open('attr_triples_1', 'w') as opened:
    opened.write(''.join(attr_lines1))
with open('attr_triples_2', 'w') as opened:
    opened.write(''.join(attr_lines2))
with open('ent_links', 'w') as opened:
    opened.write(''.join(links_lines))
with open('entity_local_name_1', 'w') as opened:
    opened.write(''.join(ent_names1))
with open('entity_local_name_2', 'w') as opened:
    opened.write(''.join(ent_names2))
with open('predicate_local_name_1', 'w') as opened:
    opened.write(''.join(pred_names1))
with open('predicate_local_name_2', 'w') as opened:
    opened.write(''.join(pred_names2))
with open('rel_triples_1', 'w') as opened:
    opened.write(''.join(rel_lines1))
with open('rel_triples_2', 'w') as opened:
    opened.write(''.join(rel_lines2))

split = [0.6, 0.2, 0.2]
links_lines = list(links_lines)
assert split[0] + split[1] + split[2] == 1, 'sanity check for data split fail'
the_samples_train = ''.join(links_lines[:int(split[0]*len(links_lines))])
the_samples_test = ''.join(links_lines[int(split[0]*len(links_lines)):-int(split[2]*len(links_lines))])
the_samples_valid = ''.join(links_lines[int(-split[2]*len(links_lines)):])
with open('631/train_links', 'w') as opened:
    opened.write(the_samples_train)
with open('631/test_links', 'w') as opened:
    opened.write(the_samples_test)
with open('631/valid_links', 'w') as opened:
    opened.write(the_samples_valid) 
