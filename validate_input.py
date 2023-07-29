import pandas as pd
import sys
import pickle

inp = pd.read_csv(sys.argv[1])

model_fn = sys.argv[2]

with open(model_fn + ".w2idx", "rb") as w2idx_f:
    w2idx = pickle.load(w2idx_f)

with open(model_fn + ".c2idx", "rb") as c2idx_f:
    c2idx = pickle.load(c2idx_f)

word_valid, tag_valid = [], []

for word, tag in zip(inp["word"], inp["tag_corrected"]):
    word_valid.append("-" if word.lower() in w2idx else "<oov>")
    tag_valid.append("-" if tag in c2idx else "<oov>")

inp["word_valid?"] = word_valid
inp["tag_valid?"] = tag_valid
inp.to_csv(sys.argv[1])
