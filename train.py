import argparse
import logging
import string
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

from model import CCAPNet
from address import typos
from address.utils import build_vocabulary
from address.proc_gen import TokenCategory

def parse_arguments():
    args = argparse.ArgumentParser(description='CCAPNet trainer script.')
    
    args.add_argument('--model-name', action='store', type=str, metavar='STR', required=True, 
                      help='model to load')         
    args.add_argument('--reset', action='store_true',
                      help='delete all checkpoints for a model')
    args.add_argument('--epochs', action='store', type=int, metavar='INT',
                      help='number of epochs to run from last checkpoint')
    args.add_argument('--batch-size', action='store', type=int, metavar='INT',
                      help='training batch size')
    
    return args.parse_args()
    
def process_arguments(args):
    model_dir = Path(args.model_name)
        
    # reset flag is used
    if args.reset == True and model_dir.is_dir():
        response = input("Confirm deletion with 'y' (otherwise exit): ")
        if response == 'y':
            print(f"Deleting '{model_dir}' directory.")
            for child in model_dir.iterdir():
                child.unlink()
            
            model_dir.rmdir()
            
        sys.exit(0)
    
    # load latest checkpoint
    elif args.reset == False and model_dir.is_dir():
        if args.epochs is None:
            print("Epochs needed for checkpointed model.")
            sys.exit(1)
            
        checkpoints = sorted(model_dir.glob('*.pt'))
        return True, model_dir, checkpoints.pop()
    
    # otherwise create a new checkpoint directory
    else:
        if args.epochs is None or args.batch_size is None:
            print("Epochs and batch size needed for new model.")
            sys.exit(1)
        
        model_dir.mkdir()
        return False, model_dir, None
        
address_chars = list(string.ascii_uppercase + string.digits + ' ')
label_chars = [str(e.value) for e in TokenCategory]

def chars_to_tensor(s, vocab):
    return torch.tensor([vocab[c] for c in s], dtype=torch.long)

class CivicAddressDataset(Dataset):
    """Dataset subclass for address data."""
    def __init__(self, csv_file, x_vocab, y_vocab, perturb_fn=None):
        self.data = pd.read_csv(
            csv_file, names=['x', 'y'], sep='|', dtype='string')
        
        self.x_vocab = x_vocab
        self.y_vocab = y_vocab
        self.perturb_fn = perturb_fn
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        address = list(self.data.iat[idx, 0])
        labels = (self.data.iat[idx, 1]).split(' ')
        
        if self.perturb_fn is not None: # perturbation should occur inplace
            self.perturb_fn(address, labels)
        
        x = chars_to_tensor(address, self.x_vocab)
        y = chars_to_tensor(labels, self.y_vocab)
        l = x.size(0)
        
        return x, y, l
        
def apply_typos(chars, clf, typo_funcs, rate, rng):
    """Randomly apply typos to an address inplace.
    
    The number of typos is Poisson distributed with parameter `rate`.
    Moreover, the typo functions are selected uniformly at random, with
    replacement.
    """
    repeats = rng.poisson(lam=rate)
    #print(f"Repeats: {repeats}")
    #print(''.join(chars))
    for i in range(repeats):
        f = rng.choice(typo_funcs)
        #print(f)
        f(chars, clf, rng)
        #print(''.join(chars))
        
def construct_batch(samples, x_pad_token, y_pad_token):
    """Collate function for DataLoader."""
    x, y, l = list(zip(*samples))
    x_padded = pad_sequence(x, batch_first=True, padding_value=x_pad_token)
    y_padded = pad_sequence(y, batch_first=True, padding_value=y_pad_token)
    lengths = torch.tensor(l, dtype=torch.long)
    return x_padded, y_padded, lengths
    
def compute_batch_accuracy(pred, targ, pad_token):
    """Compute per-character and parser accuracy."""
    correct = pred == targ
    mask = targ != pad_token
    
    char_accuracy = torch.mean(
        correct.logical_and(mask).sum(1) / mask.sum(1))
    
    parser_accuracy = torch.mean(
        correct.logical_or(mask.logical_not()).all(1).float()
    )
    
    return char_accuracy, parser_accuracy

def compute_confusion_matrix(pred, targ, pad_token, classes):
    """Compute confusion matrix over a batch of predictions."""
    pred = pred.flatten()
    targ = targ.flatten()

    mask = targ != pad_token

    pred = pred[mask].cpu()
    targ = targ[mask].cpu()

    labels = np.arange(classes)
    
    return confusion_matrix(targ, pred, labels=labels)
    
######################
# SCRIPT STARTS HERE #
######################

if __name__ == '__main__':
    args = parse_arguments()
    resume, chkpt_dir, last_chkpt_path = process_arguments(args)
    
    # define epoch-independent variables
    train_pg_path = 'datasets/train_sequences_pg.csv.gz'
    test_pg_path = 'datasets/test_sequences_pg.csv.gz'
    test_real_path = 'datasets/test_sequences_real.csv.gz'
        
    x_vocab = build_vocabulary(address_chars)
    y_vocab = build_vocabulary(label_chars)
    
    x_pad_token = len(x_vocab)
    y_pad_token = len(y_vocab)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    
    # define model, optimizer and loss
    model = CCAPNet(x_pad_token, y_pad_token)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=2.5e-5)
    
    loss = torch.nn.CrossEntropyLoss(
        ignore_index=y_pad_token, reduction='mean')
    
    if resume: # if loaded, use previous checkpoint state
        checkpoint = torch.load(last_chkpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    print(f"Model definition:\n{model}")
    
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {trainable_params}")
    
    # load datasets
    batch_size = args.batch_size if not resume else checkpoint['batch_size']
    
    collate_fn = partial(
        construct_batch,
        x_pad_token=x_pad_token, 
        y_pad_token=y_pad_token)
    
    rng = np.random.default_rng()
    perturb_fn = partial(
        apply_typos,
        typo_funcs=[typos.delete, typos.replace, typos.duplicate, typos.swaps],
        rate=1.,
        rng=rng)

    train_pg_data = CivicAddressDataset(train_pg_path, x_vocab, y_vocab, perturb_fn)
    test_pg_data = CivicAddressDataset(test_pg_path, x_vocab, y_vocab)
    test_real_data = CivicAddressDataset(test_real_path, x_vocab, y_vocab)
    
    train_pg_dataloader = DataLoader(
        train_pg_data, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        drop_last=True,
        generator=torch.Generator(device=device))
        
    test_pg_dataloader = DataLoader(
        test_pg_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=True)
        
    test_real_dataloader = DataLoader(
        test_real_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=True)
    
    # prepare logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout), 
            logging.FileHandler(chkpt_dir/'log.txt')
        ])
    
    # epoch loop
    epochs = args.epochs
    epoch_offset = 0 if not resume else checkpoint['epoch'] + 1
    
    for t in range(epochs):
        T = epoch_offset + t
        
        logging.debug("Starting loop at epoch {:d}".format(T))
        
        # training loop
        model.train()
        for batch, (x, y, l) in enumerate(train_pg_dataloader):
            logits = model(x, l)
            logits = torch.transpose(logits, 1, 2)
            error = loss(logits, y)
            
            error.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # compute per-character and parser accuracy on training batch
            pred = logits.detach().argmax(1)
            
            char_acc, pars_acc = compute_batch_accuracy(pred, y, y_pad_token)
            
            logging.debug(
                "[TRAIN] batch {:>4d} loss: {:.5f} char acc: {:.2%} parse acc: {:.2%}".format(
                    batch, error.item(), char_acc, pars_acc))
        
        # test loop
        model.eval()
        test_pg_char_acc = 0.
        test_pg_pars_acc = 0.
        test_real_char_acc = 0.
        test_real_pars_acc = 0.
        
        confusion_dims = (len(y_vocab), len(y_vocab))
        test_pg_confusion = np.zeros(confusion_dims, dtype=np.int64)
        test_real_confusion = np.zeros(confusion_dims, dtype=np.int64)

        with torch.no_grad():
            for batch, (x, y, l) in enumerate(test_pg_dataloader):
                logits = model(x, l)
                logits = torch.transpose(logits, 1, 2)
                pred = logits.argmax(1)

                test_pg_confusion += compute_confusion_matrix(pred, y, y_pad_token, len(y_vocab))

                char_acc, pars_acc = compute_batch_accuracy(pred, y, y_pad_token)
                
                test_pg_char_acc += char_acc
                test_pg_pars_acc += pars_acc
                
            test_pg_char_acc /= len(test_pg_dataloader)
            test_pg_pars_acc /= len(test_pg_dataloader)
            #test_pg_confusion /= test_pg_confusion.sum()

            for batch, (x, y, l) in enumerate(test_real_dataloader):
                logits = model(x, l)
                logits = torch.transpose(logits, 1, 2)
                pred = logits.argmax(1)

                test_real_confusion += compute_confusion_matrix(pred, y, y_pad_token, len(y_vocab))
                
                char_acc, pars_acc = compute_batch_accuracy(pred, y, y_pad_token)
                
                test_real_char_acc += char_acc
                test_real_pars_acc += pars_acc
                
            test_real_char_acc /= len(test_real_dataloader)
            test_real_pars_acc /= len(test_real_dataloader)
            #test_real_confusion /= test_real_confusion.sum()
            
            logging.debug(
                "[TEST] epoch {:d} PG ca: {:.2%} PG pa: {:.2%} real ca: {:.2%} real pa: {:.2%}".format(
                    T, test_pg_char_acc, test_pg_pars_acc, test_real_char_acc, test_real_pars_acc))
        
        logging.debug("Ending loop at epoch {:d}".format(T))
        
        # save checkpoint and confusion matrix
        confusion_path = chkpt_dir / '{:05d}.npz'.format(T)
        chkpt_path = chkpt_dir / '{:05d}.pt'.format(T)

        np.savez(confusion_path, pg=test_pg_confusion, real=test_real_confusion)
        
        torch.save({
            'epoch' : T,
            'batch_size' : batch_size,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
        }, chkpt_path)
        
        logging.debug("Saved checkpoint to '{}'".format(chkpt_path))
