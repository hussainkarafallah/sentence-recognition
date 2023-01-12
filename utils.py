import torch
from torch import nn
import numpy as np


def get_id_to_tokens(char_dict):
    id_to_tokens = {v:k for k, v in char_dict.items()}
    id_to_tokens.update({0:'<BLANK>'})
    return id_to_tokens


def inference(model, x, char_dict):
    id_to_tokens = get_id_to_tokens(char_dict)
    with torch.no_grad():
        x = torch.Tensor(x[np.newaxis, ...]).to(device='cuda')
        pred = model(x).permute((1, 0, 2))
    
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
