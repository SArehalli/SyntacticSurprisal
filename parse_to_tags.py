import sys
from nltk.tree import Tree
import csv


parses = []
with open("".join(sys.argv[1].split(".")[:-1]) + ".ccg") as in_f:
    for line in in_f:
        if line[:2] != "ID":
            parses.append(line)

def read_label(s):
    return((s.split()[1], s.split()[4]))

taggings =  []

for parse in parses:
    t = Tree.fromstring(parse, node_pattern="<T[^>]*>", leaf_pattern="<L[^>]*>",
                               read_leaf=read_label)
    taggings.append(" ".join((x[0] for x in t.leaves())))

sents = []
with open("".join(sys.argv[1].split(".")[:-1]) + ".csv") as in_f:
    reader = csv.DictReader(in_f)
    sents = []
    type_ = []
    cond = []
    for row in reader:
        sents.append(row["sentence"])
        type_.append(row["type"])
        cond.append(row["condition"])

out_dictlist = [{"type":ty, "condition":c, "sentence":s, "supertags_auto":t} for ty,c,s,t in zip(type_, cond, sents, taggings)]

with open("".join(sys.argv[1].split(".")[:-1]) + ".full.csv", "w") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=["type", "condition", "sentence", "supertags_auto"])
    writer.writeheader()
    writer.writerows(out_dictlist)

out_byword = []

for i, (ty, c, sent, tagging) in enumerate(zip(type_, cond, sents, taggings)):
    for j, (word, tag) in enumerate(zip(sent.split(), tagging.split())):
        out_byword.append({"item":i, "type":ty, "condition":c, "full_sent":sent, "word":word, "word_pos":j, "tag_auto":tag}) 

with open("".join(sys.argv[1].split(".")[:-1]) + ".byword.csv", "w") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=["item", "type", "condition", "full_sent", "word_pos", "word", "tag_auto"])
    writer.writeheader()
    writer.writerows(out_byword)
