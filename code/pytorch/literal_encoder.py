import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing
from gensim.models.word2vec import Word2Vec

from pytorch.utils import l2_normalize, get_optimizer


def read_word2vec(path, vector_dimension=300):
    print(path)
    word2vec = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split(' ')
            if len(line) != vector_dimension + 1:
                continue
            v = np.array(list(map(float, line[1:])), dtype=np.float32)
            word2vec[line[0]] = v
    return word2vec


def generate_word2vec_by_character_embedding(word_list, vector_dimension):
    character_vectors = {}
    alphabets = ''
    num_chs = {}
    for word in word_list:
        for ch in word:
            n = 1
            if ch in num_chs:
                n += num_chs[ch]
            num_chs[ch] = n
    num_chs = sorted(num_chs.items(), key=lambda x: x[1], reverse=True)
    ch_sum = sum([n for (_, n) in num_chs])
    for i in range(len(num_chs)):
        if num_chs[i][1] / ch_sum >= 0.0001:
            alphabets += num_chs[i][0]
    print(alphabets)
    print("Number of alphabets:", len(alphabets))
    char_sequences = [list(word) for word in word_list]
    model = Word2Vec(char_sequences, size=vector_dimension, window=5, min_count=1)
    # model.save('char_embeddings.vec')
    for ch in alphabets:
        assert ch in model
        character_vectors[ch] = model[ch]

    word2vec = {}
    for word in word_list:
        vec = np.zeros(vector_dimension, dtype=np.float32)
        for ch in word:
            if ch in alphabets:
                vec += character_vectors[ch]
        if len(word) != 0:
            word2vec[word] = vec / len(word)
    return word2vec


def generate_unlisted_word2vec(word2vec, literal_list, vector_dimension):
    unlisted_words = []
    for literal in literal_list:
        words = literal.split(' ')
        for word in words:
            if word not in word2vec:
                unlisted_words.append(word)
    word2vec_char = generate_word2vec_by_character_embedding(unlisted_words, vector_dimension)
    word2vec.update(word2vec_char)
    return word2vec


def look_up_word2vec(id_tokens_dict, word2vec, tokens2vec_mode='add', keep_unlist=False, vector_dimension=300, tokens_max_len=5):
    if tokens2vec_mode == 'add':
        return tokens2vec_add(id_tokens_dict, word2vec, vector_dimension, keep_unlist)
    else:
        return tokens2vec_encoder(id_tokens_dict, word2vec, vector_dimension, tokens_max_len, keep_unlist)


def tokens2vec_encoder(id_tokens_dict, word2vec, vector_dimension, tokens_max_len, keep_unlist):
    tokens_vectors_dict = {}
    for v_id, tokens in id_tokens_dict.items():
        words = tokens.split(' ')
        vectors = np.zeros((tokens_max_len, vector_dimension), dtype=np.float32)
        flag = False
        for i in range(min(tokens_max_len, len(words))):
            if words[i] in word2vec:
                vectors[i] = word2vec[words[i]]
                flag = True
        if flag:
            tokens_vectors_dict[v_id] = vectors
    if keep_unlist:
        for v_id, _ in id_tokens_dict.items():
            if v_id not in tokens_vectors_dict:
                tokens_vectors_dict[v_id] = np.zeros((tokens_max_len, vector_dimension), dtype=np.float32)
    return tokens_vectors_dict


def tokens2vec_add(id_tokens_dict, word2vec, vector_dimension, keep_unlist):
    tokens_vectors_dict = {}
    cnt = 0
    for e_id, local_name in id_tokens_dict.items():
        words = local_name.split(' ')
        vec_sum = np.zeros(vector_dimension, dtype=np.float32)
        for word in words:
            if word in word2vec:
                vec_sum += word2vec[word]
        if sum(vec_sum) != 0:
            vec_sum = vec_sum / np.linalg.norm(vec_sum)
        elif not keep_unlist:
            cnt += 1
            continue
        tokens_vectors_dict[e_id] = vec_sum
    # print("clear_unlisted_value:", cnt)
    return tokens_vectors_dict


def look_up_char2vec(id_tokens_dict, character_vectors, vector_dimension=300):
    tokens_vectors_dict = {}
    for e_id, ln in id_tokens_dict.items():
        vec_sum = np.zeros(vector_dimension, dtype=np.float32)
        for ch in ln:
            if ch in character_vectors:
                vec_sum += character_vectors[ch]
        if sum(vec_sum) != 0:
            vec_sum = vec_sum / np.linalg.norm(vec_sum)
        tokens_vectors_dict[e_id] = vec_sum
    return tokens_vectors_dict


class AutoEncoder(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=None, activ='', normalize=False):
        super(AutoEncoder, self).__init__()
        self.normalize = normalize

        if hidden_dims is None:
            hidden_dims = [1024, 512]
        cfg = [input_dim] + hidden_dims + [output_dim]

        self.encoder = self._make_layers(cfg, activ)
        self.decoder = self._make_layers(cfg[::-1], activ)

        self._init_parameters()

    def _init_parameters(self):
        for param in self.parameters():
            nn.init.normal_(param, std=1.0)

    @staticmethod
    def _make_layers(cfg, activ):
        layers = []
        for i in range(len(cfg) - 1):
            layers += [nn.Linear(cfg[i], cfg[i + 1], bias=True)]
            if activ == 'sigmoid':
                layers += [nn.Sigmoid()]
            elif activ == 'tanh':
                layers += [nn.Tanh()]
        return nn.Sequential(*layers)

    def forward(self, x):
        z = self.encoder(x)
        if self.normalize is not None:
            z = l2_normalize(z)
        x = self.decoder(z)
        return x


def encode_literals(args, literal_list):
    """
    Encode literal for name embedding

    Parameters
    ----------
    args
        MultiKE model args.
    literal_list
        Literals to encode.
    """
    tokens_max_len = args.literal_len
    word2vec = read_word2vec(args.word2vec)
    word2vec_unlisted = generate_unlisted_word2vec(word2vec, literal_list, args.word2vec_dim)
    literal_vector_list = []
    for literal in literal_list:
        vectors = np.zeros((tokens_max_len, args.word2vec_dim), dtype=np.float32)
        words = literal.split(' ')
        for i in range(min(tokens_max_len, len(words))):
            if words[i] in word2vec_unlisted:
                vectors[i] = word2vec_unlisted[words[i]]
        literal_vector_list.append(vectors)
    literal_vector_list = np.stack(literal_vector_list).reshape(len(literal_vector_list), -1)
    assert len(literal_list) == len(literal_vector_list)

    word_vec_norm_list = preprocessing.normalize(literal_vector_list, norm='l2', axis=1, copy=False)
    dataset = TensorDataset(torch.from_numpy(word_vec_norm_list))
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder(literal_vector_list.shape[1], args.embed_dim, activ=args.encoder_activ, normalize=args.encoder_normalize)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = get_optimizer('adagrad', model.parameters(), 0.001)

    total = 0
    running_loss = 0.0

    model.train()
    for i in range(args.encoder_epochs):
        start_time = time.time()
        for inputs, in dataloader:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            total += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)

        end_time = time.time()
        print("epoch {} of literal encoder, loss: {:.4f}, time: {:.4f}s".format(i + 1, running_loss / total, end_time - start_time))

    print("encode literal embeddings...", len(dataset))
    dataset = TensorDataset(torch.from_numpy(literal_vector_list))
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    encoded_literal_vector = []

    model.eval()
    with torch.no_grad():
        for inputs, in dataloader:
            inputs = inputs.to(device)

            outputs = model.encoder(inputs)
            encoded_literal_vector.append(outputs.cpu())

    return torch.cat(encoded_literal_vector, dim=0).numpy()


def save_literal_vectors(path, literal_list, literal_vectors):
    np.save(os.path.join(path, 'literal_vectors.npy'), literal_vectors)
    assert len(literal_list) == len(literal_vectors)
    with open(os.path.join(path, 'literals.txt'), 'w', encoding='utf-8') as file:
        for literal in literal_list:
            file.write(literal + '\n')
    print("literals and embeddings are saved in", path)


def literal_vectors_exists(path):
    literal_vectors_path = os.path.join(path, 'literal_vectors.npy')
    literal_path = os.path.join(path, 'literals.txt')
    return os.path.exists(literal_vectors_path) and os.path.exists(literal_path)


def load_literal_vectors(path):
    print("load literal embeddings from", path)
    literal_list = []
    literal_vectors = np.load(os.path.join(path, 'literal_vectors.npy'))
    with open(os.path.join(path, 'literals.txt'), 'r', encoding='utf-8') as file:
        for line in file:
            literal = line.strip('\n')
            literal_list.append(literal)
    return literal_list, literal_vectors
