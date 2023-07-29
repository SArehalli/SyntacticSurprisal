import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle

import argparse

import sys
sys.path.insert(0, "./CCGMultitask/")
from model import MultiTaskModel

import pandas as pd
import csv

sys.path.insert(0, "./sapbenchmark/Surprisals/")
from util import align

# Supress warnings since we fix the model loading anyway
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

from tqdm import tqdm

def indexify(word, w2idx):
    """ Convert word to an index into the embedding matrix """
    try:
        return w2idx[word] if word in w2idx else w2idx["<oov>"]
    except:
        print("error on ", word)
        raise

def tokenize(sent):
    # respect commas as a token
    sent = " ,".join(sent.split(","))

    sent = " .".join(sent.split("."))

    # split on 's
    sent = " 's".join(sent.split("\'s"))

    # split on n't
    sent = " n't".join(sent.split("n\'t"))

    return sent.split()

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--model_type", type=str, required=True)
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--aligned", action="store_true")
parser.add_argument("--uncased", action="store_true")
parser.add_argument("--progress", action="store_true")
# TODO option for selecting subword merges to compute

args = parser.parse_args()

# how can we combine subwords/punctuation to get one surprisal per word?
merge_fs = {"sum_":sum, "mean_": lambda x: sum(x)/len(x)}

# progress bar
wrap = tqdm if args.progress else (lambda x: x)

# Make it reproduceable
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# Load vocab
model_fns = args.model.split(",")
w2idxs = []
c2idxs = []
for model_fn in model_fns:
    with open(model_fn + ".w2idx", "rb") as w2idx_f:
        w2idx = pickle.load(w2idx_f)
    w2idxs.append(w2idx)

    with open(model_fn + ".c2idx", "rb") as c2idx_f:
        c2idx = pickle.load(c2idx_f)
    c2idxs.append(c2idx)


# Load experimental data csv
in_f = open(args.input, "r")
inp = list(csv.DictReader(in_f))

for model_num, model_fn in wrap(enumerate(model_fns)):

    ssurpss = []
    surpss = []

    # Load model
    model = MultiTaskModel(len(w2idx.keys()), 650, 650, 
                           [len(w2idx.keys()), len(c2idx.keys())], 2,)
    model.load_state_dict(torch.load(model_fn + ".pt", 
                                     map_location = torch.device("cuda" if args.cuda 
                                                             else "cpu")))
    if args.cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    model.eval()
    
    out_rows = []
    with torch.no_grad():
        for row in wrap(inp):
            sentence = ["<eos>"] + tokenize(row["Sentence"]) # EOS prepend
            input = torch.LongTensor([indexify(w.lower() if args.uncased else w, w2idxs[model_num]) for w in sentence])
            
            if args.cuda:
                input = input.cuda()

            h,c = model.init_hidden(1)
            surps = []
            syn_surps = []
            for token, next_token in zip(input[:-1], input[1:]):
                lm_n, ccg_n, (h,c) = model(token.view(-1, 1), (h,c))
                next_word_prob = lm_n[-1].view(-1)
                # Get normal ("lexical") surprisal
                surps.append(-next_word_prob[next_token].item())

                out_o = None
                # Get whatever kind of syntactic surprisal we specify
                vocab_size = len(w2idxs[model_num]) 
                tagset_size = len(c2idxs[model_num])

                all_vocab = torch.tensor(list(range(vocab_size))).view(1, -1)

                if args.cuda:
                    all_vocab = all_vocab.cuda()

                vocab_chunks = torch.split(all_vocab, args.batch_size, 1)

                out_os = []
                for chunk in vocab_chunks:
                    h_ = h.tile((1, chunk.shape[1], 1))
                    c_ = c.tile((1, chunk.shape[1], 1))
                    _, out, _ = model(chunk, (h_, c_))
                    out_os.append(out)

                out_o = torch.vstack(out_os)
                out_o = out_o.transpose(0,1)
                
                if args.model_type.split("_")[1] == "ambig":
                    # probably a cleaner way to do this, but this is easiest to check correctness
                    # = sum_c*( p(cn* | w1...wn) . sum_wn*( p(cn* | w1...wn-1, wn*) . p(wn* | w1...wn-1) ) 
                    _, out_gold, _ = model(next_token.view(1,1), (h,c))
                    p_predtag = torch.logsumexp(out_o + next_word_prob.tile((tagset_size, 1)), dim=1, keepdim=False) # sum_w*
                    tag_surprisal = -torch.logsumexp(p_predtag + out_gold.view(-1), dim=0, keepdim=False).item() # sum_c*
                elif args.model_type.split("_")[1] == "klambig":
                    # KL Divergence between predicted tag distribution and the tag distribution given the gold word.
                    _, out_gold, _ = model(next_token.view(1,1), (h,c)) # gold
                    p_predtag = torch.logsumexp(out_o + next_word_prob.tile((tagset_size, 1)), dim=1, keepdim=False) # predicted
                    tag_surprisal = -F.kl_div(p_predtag, out_gold.view(-1), log_target=True).item()
                else:
                    print("You have a typo in your model type")
                syn_surps.append(tag_surprisal)

            if args.aligned:
                words = row["Sentence"].split() 
                piecess, breaks = align(words, sentence[1:]) # drop EOS in sentence
                print(piecess)
                for i, (word, pieces) in enumerate(zip(words, piecess)):
                    new_row = row.copy() # new object, not a reference to the iterator
                    # Note that since the beginning-of-sentence <eos> is in out/input, but was dropped from breaks, we need to
                    # correct for misalignment (thus out[k] rather than out[k-1], input[k+1] instead of input[k]).
                    surps_ = [surps[k]
                             for k in range(breaks[i], breaks[i+1])]
                    for merge_fn, merge_f in merge_fs.items():
                        new_row[merge_fn +  "lex_surprisal"] = merge_f(surps_)
                    syn_surps_ = [syn_surps[k]
                             for k in range(breaks[i], breaks[i+1])]
                    for merge_fn, merge_f in merge_fs.items():
                        new_row[merge_fn +  "syn_surprisal"] = merge_f(syn_surps_)
                    new_row["token"] = ".".join([(w.lower() if args.uncased else w) if w in w2idxs[model_num]
                                                 else "<UNK>" for w in pieces])
                    new_row["word"] = word
                    new_row["word_pos"] = i 
                    out_rows.append(new_row)
                print(len(out_rows))
                    
            else:
                for i, (word_idx, word) in enumerate(zip(input, sentence[1:])): # drop EOS
                    new_row = row.copy() # new object, not a reference to the iterator
                    new_row["surprisal"] = -F.log_softmax(out[i], dim=-1).view(-1)[word_idx].item()
                    new_row["token"] = word if word in w2idxs[model_num] else "<UNK>"
                    new_row["word"] = word
                    new_row["word_pos"] = i 
                    out_rows.append(new_row)

    # write out to csv
    print(len(out_rows))
    print(out_rows[:10])
    with open(args.output + ".m{}".format(model_num), "w") as out_f:
        writer = csv.DictWriter(out_f, fieldnames = out_rows[0].keys())
        writer.writeheader()
        writer.writerows(out_rows)
