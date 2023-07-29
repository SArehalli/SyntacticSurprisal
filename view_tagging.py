import sys
import csv

with open(sys.argv[1]) as in_f:
    reader = csv.DictReader(in_f)
    for row in reader:
        sent = row["sentence"].split()
        tags = row["supertags_auto"].split()

        sent_line = ""
        tag_line = ""
        for w, tag in zip(sent, tags):
            width = max([len(w), len(tag)]) + 2
            sent_line += ("{:" + str(width) + "}").format(w)
            tag_line += ("{:" + str(width) + "}").format(tag)
        print(tag_line)
        print(sent_line)
