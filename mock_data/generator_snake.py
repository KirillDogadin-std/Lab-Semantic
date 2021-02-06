import numpy as np
import random as rnd
import re

attr_finder = re.compile('^.*\\t(.*\\t.*)$')

LINES_TO_READ_FROM_FILES = 1000
N_LINES_TO_SAMPLE = 320


def choose_entities(file_path):
    with open(file_path) as opened:
        hunred_lines = opened.readlines()[:LINES_TO_READ_FROM_FILES]
        chosen_lines = rnd.sample(hunred_lines, N_LINES_TO_SAMPLE) 

    return chosen_lines


def choose_predicates(file_path):
    with open(file_path) as opened:
        hunred_lines = opened.readlines()[:LINES_TO_READ_FROM_FILES]
        chosen_lines = rnd.sample(hunred_lines, int(N_LINES_TO_SAMPLE/10))

    return chosen_lines


def generate_realation_triples(chosen_relations, chosen_entities):
    result = set()
    used = set()
    loops = int(N_LINES_TO_SAMPLE/6)

    for i in range(loops):
        chosen_entities_ch = chosen_entities[i*5:(i+1)*5]
        for _ in range(6):
            rel = rnd.sample(chosen_relations, 1)
            ent1 = rnd.sample(chosen_entities_ch[1:], 1)[0]
            core_ent = chosen_entities[0]

            line = '{}\t{}\t{}\n'.format(core_ent.split('\t')[0], rel[0].split('\t')[0], ent1.split('\t')[0])
            result.add(line)
            used.add(ent1)
            used.add(core_ent)

    for i in range(loops):
        ch1 = chosen_entities[i*5:(i+1)*5]
        ch2 = chosen_entities[(i+1)*5:(i+2)*5]
        for _ in range(3):
            rel = rnd.sample(chosen_relations, 1)[0]
            ent1 = rnd.sample(ch1, 1)[0]
            ent2 = rnd.sample(ch2, 1)[0]

            e1 = ent1.split('\t')[0]
            e2 = ent2.split('\t')[0]
            line = '{}\t{}\t{}\n'.format(e1, rel.split('\t')[0], e2)
            result.add(line)
            used.add(ent1)
            used.add(ent2)
    return list(result), list(used)


def generate_attribute_triples(chosen_entities, file_path):
    result = []
    attr_map = dict()

    with open(file_path) as opened:
        lines = opened.readlines()
        for line in lines:
            words = line.split('\t')
            k = words[0]
            v = line
            attr_map[k] = v

    for ent in chosen_entities:
        attr_count = 1  # rnd.randint(1, 3)

        for _ in range(attr_count):
            _id = ent.split('\t')[0]
            attr = attr_map[_id]
            result.append(attr)

    return result


def choose_attributes(file_path):
    result = []

    with open(file_path) as opened:
        hunred_lines = opened.readlines()[:LINES_TO_READ_FROM_FILES]
        chosen_lines = rnd.sample(hunred_lines, N_LINES_TO_SAMPLE)

        for line in chosen_lines:
            found_attr = attr_finder.match(line).group(1)
            result.append(found_attr)

    return result


def _split_chunk(chunk):
    result = {}

    for line in chunk:
        k, v = line.split('\t')
        result[k] = v

    return result


def find_ent_links(chosen_ents, file_path):
    result = []
    links_dict = {}

    with open(file_path) as opened:
        lines = opened.readlines()
        map = _split_chunk(lines)

        for ent in chosen_ents:
            _ent = ent.split('\t')[0]
            linked_ent = map.get(_ent, None)
            if linked_ent is not None:
                result.append('{}\t{}'.format(_ent, linked_ent))
                links_dict[_ent] = linked_ent

    return result, links_dict


def copy_rels(ents1, ents2, triples1, preds2):
    map = {ents1[k].split('\t')[0]: ents2[k].split('\t')[0] for k in range(len(ents1))}

    pmap = {}
    relset = set()
    for e in triples1:
        r = e.split('\t')[1]
        relset.add(r)

    rels_will_be_mapped_to = rnd.sample(preds2, len(relset))

    for i, t in enumerate(relset):
        pmap[t] = rels_will_be_mapped_to[i]

    ret = []
    for t in triples1:
        a, r, b = t.split('\t')
        s = (map[a] + '\t' +
             pmap[r].split('\t')[0] + '\t' +
             map[b.strip('\n')] + '\n')
        ret.append(s)

    return ret


def main(ent_name, pred_name, ent_links, attr_triples):
    ents = choose_entities(ent_name)
    preds = choose_predicates(pred_name[0])
    preds2 = choose_predicates(pred_name[1])
    r_triples, ents = generate_realation_triples(preds, ents)
    a_triples = generate_attribute_triples(ents, attr_triples[0])

    links_lines, links_dict = find_ent_links(ents, ent_links)
    ents2 = [links_dict[k.split('\t')[0]][:-1]+'\t'+k.split('\t')[1] for k in ents]
    a_triples2 = generate_attribute_triples(ents2, attr_triples[1])
    r_triples2 = copy_rels(ents, ents2, r_triples, preds2)

    return a_triples, a_triples2, links_lines, ents, ents2, preds, preds2, r_triples, r_triples2


path_base = '../../dataset_lab/BootEA_DBP_WD_100K/'
ent_name = path_base + 'entity_local_name_1'
pred_name = [path_base + 'predicate_local_name_1', path_base + 'predicate_local_name_2']
ent_links = path_base + 'ent_links'
attr_triples = [path_base + 'attr_triples_1', path_base + 'attr_triples_2']

if __name__ == '__main__':
    a_triples, a_triples2, links_lines, ents, ents2, preds, preds2, r_triples, r_triples2 = main(ent_name, pred_name, ent_links, attr_triples)
    with open('attr_triples_1', 'w') as opened:
        opened.write(''.join(a_triples))
    with open('attr_triples_2', 'w') as opened:
        opened.write(''.join(a_triples2))
    with open('ent_links', 'w') as opened:
        opened.write(''.join(links_lines))
    with open('entity_local_name_1', 'w') as opened:
        opened.write(''.join(ents))
    with open('entity_local_name_2', 'w') as opened:
        opened.write(''.join(ents2))
    with open('predicate_local_name_1', 'w') as opened:
        opened.write(''.join(preds))
    with open('predicate_local_name_2', 'w') as opened:
        opened.write(''.join(preds2))
    with open('rel_triples_1', 'w') as opened:
        opened.write(''.join(r_triples))
    with open('rel_triples_2', 'w') as opened:
        opened.write(''.join(r_triples2))

    split = [0.6, 0.2, 0.2]
    assert split[0] + split[1] + split[2] == 1, 'sanity check for data split fail'
    the_samples_train = ''.join(links_lines[:int(split[0]*N_LINES_TO_SAMPLE)])
    the_samples_test = ''.join(links_lines[int(split[0]*N_LINES_TO_SAMPLE):-int(split[2]*N_LINES_TO_SAMPLE)])
    the_samples_valid = ''.join(links_lines[int(-split[2]*N_LINES_TO_SAMPLE):])
    with open('631/train_links', 'w') as opened:
        opened.write(the_samples_train)
    with open('631/test_links', 'w') as opened:
        opened.write(the_samples_test)
    with open('631/valid_links', 'w') as opened:
        opened.write(the_samples_valid) 
