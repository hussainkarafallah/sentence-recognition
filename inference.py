import argparse
import json
import utils
from dataset import OCRDataset
from model import OCRModel
import torch
from torch import nn
import numpy as np


def get_char_dict(path):
    with open(path, 'r') as json_file:
        d = json.load(json_file)
    return d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpt', dest='cpt', help='checkpoint file', required=True)
    parser.add_argument('--val-data', dest='val_data', help='path to val data file', required=True)
    parser.add_argument('--data', dest='data', help='path to data folder', required=True)
    parser.add_argument('--vocab', dest='vocab', help='path to vocab file', required=True)
    parser.add_argument('--ngpus', dest='ngpus', help='Nummber of GPUs', default=1, type=int)
    args = parser.parse_args()

    labels_transformations = [
        lambda label: str(label),
        lambda label: [char_dict[c] if c in char_dict else -1 for c in label]
    ]

    char_dict = get_char_dict(args.vocab)
    val_data = OCRDataset(args.val_data, args.data, labels_transformations=labels_transformations)
    model = OCRModel(num_classes=len(char_dict.keys()), num_lstms=2)
    model = nn.DataParallel(model, device_ids = list(range(args.ngpus)))
    cpt_dict = torch.load(args.cpt)
    model.load_state_dict(cpt_dict['state_dict'])
    model.to(device='cuda')

    cnt = 10
    for img, label in val_data:
        cnt -= 1
        if cnt == 0:
            break
        img = torch.Tensor(img[..., np.newaxis]).cuda()
        pred_sent = utils.inference(model, img[0], char_dict)
        true_sent = utils.get_sent(label[0], char_dict)
        print('-'*20)
        print(f'model output: {pred_sent}')
        print(f'true_sent: {true_sent}')
        print('-'*20)
