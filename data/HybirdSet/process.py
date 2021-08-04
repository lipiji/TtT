import difflib
from collections import Counter
dic = Counter()

def load_file(file1, is_train=False):
    docs = []
    with open(file1) as f1:
        k = 0
        for line in f1:
            k += 1
            if k % 10000 == 0:
                print(k)
            line = line.strip()
            if not line:
                continue
            fs = line.split("\t")
            if len(fs) != 2:
                continue
            x, y = fs
            x = x.strip()
            y = y.strip()
            if is_train:
                dic.update([w for w in x] + [w for w in y])
            a = x
            b = y
            s = difflib.SequenceMatcher(None, a, b)
            tags = ["-"] * len(x)
            for tag, i1, i2, j1, j2 in s.get_opcodes():
                if tag == "delete":
                    for i in range(i1, i2):
                        tags[i] = "D"
                elif tag == "equal":
                    for i in range(i1, i2):
                        tags[i] = "E"
                elif tag == "replace":
                    for i in range(i1, i2):
                        tags[i] = "R"
                elif tag == "insert":
                    assert i1 == i2
                    tags[i1-1] = "I"
                else:
                     print("ERROR")
            if "-" in tags:
                print("ERROR-----")
            if len(set(tags)) == 1:
                continue
            if len(set(tags)) == 2 and "E" in tags and "R" in tags:
                docs.append((x, y, " ".join(tags)))
    return docs

train_xs = load_file("../train.txt", True)
dev_xs = load_file("../dev.txt", True)
test_xs = load_file("../test.txt", False)

with open("train.txt", "w") as fo:
    for x, y, t in train_xs:
        fo.write(x + '\t' + y + "\t" + t + '\n')

with open("dev.txt", "w") as fo:
    for x, y, t in dev_xs:
        fo.write(x + '\t' + y + "\t" + t + '\n')

with open("test.txt", "w") as fo:
    for x, y, t in test_xs:
        fo.write(x + '\t' + y + "\t" + t + '\n')


with open('vocab.txt', 'w', encoding ='utf8') as fo:
    for x, y in dic.most_common():
        fo.write(x+'\t'+str(y)+'\n')
