import glob

files = glob.glob("data/*/article.features")

doc = []
sen = []

for file_name in files:
    with open(file_name, 'r') as f:
        lines = f.readlines()
        doc.append(len(lines))
        for line in lines:
            line = line.strip().split()
            sen.append(len(line))

print("Documents: total=%d, max=%d, avg=%f" % (len(doc), max(doc), (sum(doc)/float(len(doc)))))
print("Sentences: total=%d, max=%d, avg=%f" % (len(sen), max(sen), (sum(sen)/float(len(sen)))))
