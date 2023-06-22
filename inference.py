import argparse
import re
import sys
from pathlib import Path

import torch

from model import CCAPNet
from train import chars_to_tensor, address_chars, label_chars
from address.utils import normalize_text, build_vocabulary
from address.proc_gen import TokenCategory


def parse_arguments():
    args = argparse.ArgumentParser(description='CCAPNet inference script.')

    args.add_argument('--model-path', action='store', type=str, metavar='PATH', required=True,
                      help='model to load')
    args.add_argument('--input', action='store', type=str, metavar='STR',
                      help='address line to parse')
    args.add_argument('--file', action='store', type=str, metavar='PATH',
                      help='file containing address lines to parse')

    return args.parse_args()

def process_arguments(args):
    model_path = Path(args.model_path)

    try:
        checkpoint = torch.load(model_path)
    except:
        print('Failed to load model, exiting.')
        sys.exit(1)

    if args.input and args.file:
        print("Cannot use both '--input' and '--file'.")
        sys.exit(1)

    return checkpoint['model_state_dict']

def check_continuity(tc_list):
    seen = []
    for cat in tc_list:
        if cat == TokenCategory.SEPARATOR:
            continue
        elif cat not in seen:
            seen.append(cat)
        elif cat in seen[:-1]:
            return False

    return True

if __name__ == '__main__':
    args = parse_arguments()
    state_dict = process_arguments(args)

    x_vocab = build_vocabulary(address_chars)
    y_vocab = build_vocabulary(label_chars)
    y_vocab_inv = {y_vocab[c] : c for c in y_vocab}
    tc_inv = {str(e.value) : e for e in TokenCategory}

    model = CCAPNet(len(x_vocab), len(y_vocab))
    model.load_state_dict(state_dict)
    model.eval()
    
    if args.input:
        text = normalize_text(args.input)
        assert len(text) > 0, 'Normalized text has zero length'
        z = chars_to_tensor(text, x_vocab)
        z = z.unsqueeze(0)
        l = torch.tensor([z.numel()])
        
        with torch.no_grad():
            logits = model(z,l)
            pred = logits.argmax(2).squeeze(0)
            tc_list = [tc_inv[y_vocab_inv[idx]] for idx in pred.tolist()]
        
        if not check_continuity(tc_list):
            print('WARNING: parser did not categorize text in continuous blocks')

        tokens = {}

        for i, cat in enumerate(tc_list):
            if cat == TokenCategory.SEPARATOR:
                for key in tokens:
                    tokens[key].append(' ')

            if cat.name not in tokens:
                tokens[cat.name] = []
            
            tokens[cat.name].append(text[i])

        for key in tokens:
            s = ''.join(tokens[key])
            s = re.sub(r'\s+', ' ', s)
            s = s.strip()
            tokens[key] = s
       
        print(f"Normalized text: '{text}'")
        print(f"Tokens: {tokens}")


    elif args.file:
        raise NotImplementedError('Cannot handle files yet.')

    else:
        print("'--input' or '--file' were not provided!")
        sys.exit(1)

