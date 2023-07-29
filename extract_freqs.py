"""
    extract_freqs.py
    generates a csv with frequencies/counts for each word in the training set

    Run as: 
    > python extract_freqs.py <training set name>

    saves freqs.csv out to freqs.csv in the working directory

"""
import csv
import sys

def process(token):
    """ Enforce tokenization scheme: lowercase, no whitespace, and numbers as <num>
        Change as needed.
    """

    token = token.strip().lower()
    try:
        float(token)
        return "<num>"
    except ValueError:
        return token

# count each word
counts = {}
with open(sys.argv[1]) as in_f:
    for row in in_f:
        for word in row.split():
            word = process(word)
            counts[word] = counts.get(word, 0) + 1

# Write out to file: 
# make a list of dicts
count_list = []
for word, count in counts.items():
    count_list.append({"word":word, "count":count})

# and write that out to csv with DictWriter
with open("freqs.csv", "w") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=["word", "count"])
    writer.writeheader()
    writer.writerows(count_list)
