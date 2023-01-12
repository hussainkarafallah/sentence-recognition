import torch
from torch import nn
import numpy as np
import csv
import re
from nltk.metrics.distance import edit_distance
from tqdm import tqdm

def get_id_to_tokens(char_dict):
    id_to_tokens = {v:k for k, v in char_dict.items()}
    id_to_tokens.update({0:'<BLANK>'})
    return id_to_tokens


def get_vocab(csv_path):
    # loop through train examples
    # and extract words
    vocab = set()
    with open(csv_path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        labels_iter = tqdm(enumerate(reader))
        for _, (_, label) in labels_iter:
            for word in label.split():
                m = re.match("[a-zA-ZåäöáëÄÖűà]*", word)
                if m[0] == word:
                    vocab.add(word)
    return list(vocab)


def edit_distance_swap(sent, vocab, max_dist=-1):
    # loop through vocab and return first sent
    # with dist <= max_dist
    # if not found return sent
    ret_sent = []
    for word in sent.split():
        m = re.match("[a-zA-ZåäöáëÄÖűà]*", word)
        if m[0] != word:
            ret_sent.append(word)
            continue
        words_ids = dict()
        for v_word in vocab:
            thr = int(0.3*len(word))
            # 2x speedup
            if max_dist == -1 and abs( len(v_word) - len(word) ) > thr:
                continue
            if max_dist != -1 and abs( len(v_word) - len(word) ) > max_dist:
                continue
            
            dist = edit_distance(word, v_word)
            if max_dist != -1 and dist <= max_dist and dist not in words_ids:
                words_ids[dist] = v_word
            elif max_dist == -1:
                if dist <= thr and dist not in words_ids:
                    words_ids[dist] = v_word
        if words_ids.keys():
            min_key = sorted(list(words_ids.keys()))[0]
            ret_sent.append(words_ids[min_key])
        else:
            ret_sent.append(word)
    return " ".join(ret_sent)


def inference(model, x, char_dict):
    id_to_tokens = get_id_to_tokens(char_dict)
    with torch.no_grad():
        x = torch.Tensor(x[np.newaxis, ...]).to(device='cuda')
        pred = model(x)
    
    # get tokens
    tokens = []
    for i in range(pred.shape[0]):
        id = torch.argmax(pred[i,0,:]).item()
        token = id_to_tokens[id]
        tokens.append(token)
    
    # decode tokens
    sent = []
    for i in range(len(tokens)):
        if (i + 1 < len(tokens) and tokens[i] == tokens[i+1]) or tokens[i] == '<BLANK>':
            continue
        sent.append(tokens[i])
    
    return "".join(sent)


def get_sent(label, char_dict):
    id_to_tokens = get_id_to_tokens(char_dict)
    assert len(label.shape) == 1

    tokens = []
    for id in label:
        token = id_to_tokens[id]
        tokens.append(token)
    
    return "".join(tokens)
