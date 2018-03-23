"""
Script to produce the layout of the data; prints maximum, minimum, and average document 
lengths as well as the maximum, minimum, and average feature lengths.
"""


import glob

files = glob.glob("data/*/article.features")

doc = []
sen = []
counter = 0
for file_name in files:
    with open(file_name, 'r') as f:
        lines = f.readlines()
        doc.append(len(lines))
        for line in lines:
            line = line.strip().split()
            sen.append(len(line))
print("Documents: total=%d, min=%d, max=%d, avg=%f" % (len(doc), min(doc), max(doc), (sum(doc)/float(len(doc)))))
print("Sentences: total=%d, min=%d, max=%d, avg=%f" % (len(sen), min(sen), max(sen), (sum(sen)/float(len(sen)))))
