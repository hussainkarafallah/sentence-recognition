from model import OCRModel
from dataset import OCRDataset
import argparse
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import augmentations as aug
import json
import os
from torchmetrics import CharErrorRate
import utils


def load_char_dict(path):
    with open(path, 'r') as json_file:
        d = json.load(json_file)
    return d

def calc_batch_cre(model, batch, labels, char_dict, vocab):
    metric = CharErrorRate()
    cre = 0
    for i in range(batch.shape[0]):
        pred_sent = utils.inference(model, batch[i], char_dict)
        # edit distance postprocessing
        pred_sent = utils.edit_distance_swap(pred_sent, vocab, max_dist=-1)
        label_sent = utils.get_sent(labels[i], char_dict)
        cre += metric([pred_sent], [label_sent]).item()
    cre = cre / batch.shape[0]
    return cre


def train(model, loss_fn, optimizer, num_epochs,
          train_dataset, val_dataset,
          start_epoch, start_iter, previous_loss,
          save_freq, save_to):
    tot_iter = 0
    for epoch in range(start_epoch, num_epochs):
        print(f'epoch: {epoch}')
        train_loss = 0
        if epoch == start_epoch:
            train_loss = previous_loss
            train_dataset.start_at = start_iter
        else:
            train_dataset.start_at = 0
        dataset_bar = tqdm(enumerate(train_dataset))
        for iter, (x, y) in dataset_bar:
            if epoch == start_epoch and iter <= start_iter:
                continue
            x = torch.Tensor(x[..., np.newaxis]).cuda()
            y = torch.Tensor(y).long().cuda()
            
            #print(f'x.shape: {x.shape}')
            pred = model(x).permute((1, 0, 2))
            #print(f'pred.shape: {pred.shape}')
            #print(f'y.shape: {y.shape}')
            loss = loss_fn(
                pred, y,
                torch.Tensor([[pred.shape[0]] * 8]).long(),
                torch.Tensor([[y.shape[1]] * 8]).long()
            )
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x
            del y
            torch.cuda.empty_cache()

            # TODO: put the losss beside the tqdm bar
            dataset_bar.set_postfix({
                'train_loss': {train_loss/(iter+1)},
            })
            


            tot_iter += 1
            if tot_iter == save_freq:
                tot_iter = 0
                cpt_dict = {
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'iter': iter,
                    'train_loss': train_loss,
                    'optimizer_state_dict': optimizer.state_dict()
                }
                torch.save(cpt_dict, save_to+os.path.sep+f'cp_{epoch}_{iter}.pth')


def eval(model, loss_fn, val_data, char_dict, vocab):
    val_metric = 0
    val_loss = 0
    num = 0
    print('start evaluating...')
    val_bar = tqdm(enumerate(val_data))
    for _, (x, y) in val_bar:
        num += 1
        with torch.no_grad():
            x = torch.Tensor(x[..., np.newaxis]).cuda()
            metric = calc_batch_cre(model, x, y, char_dict, vocab)
            val_metric += metric
            y = torch.Tensor(y).long().cuda()

            pred = model(x)
            loss = loss_fn(
                pred, y,
                input_lengths=torch.Tensor([pred.shape[0] for _ in range(1)]).long(),
                target_lengths=torch.Tensor([y.shape[1] for _ in range(1)]).long()
            )
            val_loss += loss.item()
            val_bar.set_postfix({
              'val_loss': {val_loss/(num)},
              'val_metric': {val_metric/(num)}
            })

            del x
            del y
            torch.cuda.empty_cache()
    
    print(f'val_loss: {val_loss/num}')
    print(f'val_metric: {val_metric/num}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpt', dest='cpt', help='checkpoint file', default=None)
    parser.add_argument('--epoch', dest='epoch', help='input image path', default=1, type=int)
    parser.add_argument('--save-to', dest='save_to', help='the place to save cpt in it', required=True)
    parser.add_argument('--train-data', dest='train_data', help='path to train data file', required=True)
    parser.add_argument('--val-data', dest='val_data', help='path to val data file', required=True)
    parser.add_argument('--data', dest='data', help='path to data folder', required=True)
    parser.add_argument('--vocab', dest='vocab', help='path to vocab file', required=True)
    parser.add_argument('--save-freq', dest='save_freq', help='cpt frequency', required=True, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--eval', dest='eval', help='Evaluate', default=False, type=bool)
    args = parser.parse_args()

    

    # prepare datasets
    char_dict = load_char_dict(args.vocab)
    transformations = [
        lambda img: aug.change_light(img, ratio=1.2, mn=0, mx=1),
        lambda img: aug.change_light(img, ratio=0.8, mn=0, mx=1),
        lambda img: aug.gaussian_blur(img, kernel_size=(5, 5), std=0.1),
        lambda img: aug.gaussian_noise(img, std=0.1, mn=0, mx=1)
    ] + [
        (lambda img: aug.rect_mask(img, 40, 20, mx=1, num_masks=3))
        for i in range(3)
    ]

    labels_transformations = [
        lambda label: str(label),
        lambda label: [char_dict[c] if c in char_dict else -1 for c in label]
    ]

    train_data = OCRDataset(args.train_data, args.data, transformations, labels_transformations)
    val_data = OCRDataset(args.val_data, args.data, labels_transformations=labels_transformations)

    # load train info and model
    model = OCRModel(num_classes=len(char_dict.keys()), num_lstms=2).cuda()
    if args.cpt is not None:
        cpt_dict = torch.load(args.cpt)
        start_epoch = cpt_dict['epoch']
        start_iter = cpt_dict['iter']
        previous_loss = cpt_dict['train_loss']
        model.load_state_dict(cpt_dict['state_dict'])
    else:
        start_epoch = 0
        start_iter = 0
        previous_loss = 0
    #model = nn.DataParallel(model, device_ids = list(range(4)))
    loss_fn = nn.CTCLoss(zero_infinity=True).cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    if args.cpt is not None:
        optimizer.load_state_dict(cpt_dict['optimizer_state_dict'])
    

    # load vocab for edit distance
    vocab = utils.get_vocab(args.train_data)
    print(f'vocab size: {len(vocab)}')
    if not args.eval:
        train(model, loss_fn, optimizer, args.epoch,
            train_data, val_data,
            start_epoch, start_iter, previous_loss,
            args.save_freq, args.save_to)
    else:
        eval(model, loss_fn, val_data, char_dict, vocab)
