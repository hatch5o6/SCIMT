# ar_sents = "/home/hatch5o6/nobackup/archive/data/NLLB/mt_ar_overlap/ar.txt"
# mt_sents = "/home/hatch5o6/nobackup/archive/data/NLLB/mt_ar_overlap/mt.txt"

ar_sents = "/home/hatch5o6/nobackup/archive/data/cs-hsb/WMT-CCMatrix-Overlap/val/val.ce"
mt_sents = "/home/hatch5o6/nobackup/archive/data/cs-hsb/WMT-CCMatrix-Overlap/val/val.hsb"

def check_for_duplicates(f):
    print("EXAMINING", f)
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    print("\tLEN LINES:", len(lines))
    print("\tUNIQUE LINES:", len(set(lines)))
    print("\n")

def check_for_pair_duplicates(f1, f2):
    print("-----------------")
    print("pairs")
    print(f1)
    print(f2)
    with open(f1) as inf:
        lines1 = [l.strip() for l in inf.readlines()]
    with open(f2) as inf:
        lines2 = [l.strip() for l in inf.readlines()]
    assert len(lines1) == len(lines2)
    pairs = list(zip(lines1, lines2))

    print("PAIRS:", len(pairs))
    print("UNIQUE PAIRS:", len(set(pairs)))


check_for_duplicates(ar_sents)
check_for_duplicates(mt_sents)
print("\n\n\n")
check_for_pair_duplicates(ar_sents, mt_sents)