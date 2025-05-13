train_path = "/home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined/fastalign/word_list.an-es.cognates.0.5.txt"
with open(train_path) as inf:
    lines = [tuple(line.strip().split(" ||| ")) for line in inf.readlines()]

val_path = "/home/hatch5o6/nobackup/archive/data/CogNet/arg/CogNet-v2.0.arg.spa.val-s=1420.tsv"
with open(val_path) as inf:
    lines += [tuple(line.strip().split(" ||| ")) for line in inf.readlines()]

test_path = "/home/hatch5o6/nobackup/archive/data/CogNet/arg/CogNet-v2.0.arg.spa.test-s=1420.tsv"
with open(test_path) as inf:
    lines += [tuple(line.strip().split(" ||| ")) for line in inf.readlines()]

from collections import Counter
an_chars = Counter()
es_chars = Counter()
for an, es, dist in lines:
    for char in an:
        an_chars[char] += 1
    for char in es:
        es_chars[char] += 1

print("ES CHARS", len(es_chars))
print("AN CHARS", len(an_chars))

gt_an_path = "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/Langs/es-an/an/token2count.json"
gt_es_path = "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/Langs/es-an/es/token2count.json"
import json

with open(gt_an_path) as inf:
    gt_an = json.load(inf)
with open(gt_es_path) as inf:
    gt_es = json.load(inf)

if gt_an == an_chars:
    print("AN CHARS MATCH")
else:
    print("AN CHARS FAIL :(")

if gt_es == es_chars:
    print("ES CHARS MATCH")
else:
    print("ES CHARS FAIL :(")

# with open("es_chars.json", "w") as outf:
#     outf.write(json.dumps(es_chars, ensure_ascii=False, indent=2))
# with open("an_chars.json", "w") as outf:
#     outf.write(json.dumps(an_chars, ensure_ascii=False, indent=2))

