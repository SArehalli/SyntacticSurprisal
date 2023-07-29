import csv
import sys
import re

with open(sys.argv[1]) as cat_file:
    reader = csv.DictReader(cat_file)
    for line in reader:
        print(" ".join(re.split('[, .]', line["sentence"])))
