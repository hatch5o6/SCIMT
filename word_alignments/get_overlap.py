import argparse
from tqdm import tqdm

# Here, tgt is assumed to be the language in common

def main(
    src1,
    tgt1,
    src2,
    tgt2,
    src1_out,
    src2_out
):
    src1 = read_file(src1)
    tgt1 = read_file(tgt1)
    pairs1 = list(zip(src1, tgt1))
    pairs1_dict = make_pairs_dict(pairs1)
    
    src2 = read_file(src2)
    tgt2 = read_file(tgt2)
    pairs2 = list(zip(src2, tgt2))
    pairs2_dict = make_pairs_dict(pairs2)

    new_parallel_data = []
    n_times_more_than_one_pair = 0
    print("finding overlap")
    for tgt, srcs1 in tqdm(pairs1_dict.items(), total=len(pairs1_dict)):
        srcs2 = pairs2_dict.get(tgt)
        if srcs2 is not None:
            if len(srcs1) > 1 or len(srcs2) > 1:
                n_times_more_than_one_pair += 1

            # # Add cartesian product of pairs
            # for item1 in srcs1:
            #     for item2 in srcs2:
            #         new_parallel_data.append((item1, item2))
            
            new_parallel_data.append((srcs1[0], srcs2[0]))
            
    
    with open(src1_out, "w") as f1, open(src2_out, "w") as f2:
        for item1, item2 in new_parallel_data:
            f1.write(item1.strip() + "\n")
            f2.write(item2.strip() + "\n")

    print(f"WROTE {len(new_parallel_data)} LINES OF NEW PARALLEL DATA")
    print(f"{n_times_more_than_one_pair} TIMES MORE THAN ONE PAIR FOR THE SAME SENTENCE")

def make_pairs_dict(pairs):
    print("making pairs dict")
    pairs_dict = {}
    for src, tgt in pairs:
        if tgt not in pairs_dict:
            pairs_dict[tgt] = []
        assert src not in pairs_dict[tgt]
        pairs_dict[tgt].append(src)
    return pairs_dict

def read_file(f):
    print("reading", f)
    with open(f) as inf:
        lines = [line.strip() for line in inf.readlines()]
    return lines


# lang1 and lang2 are the src and tgt we want in the end, e.g. Arabic and Maltese
# com1 and com2 is the language in common, i.e. English
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang1")
    parser.add_argument("--com1")
    parser.add_argument("--lang2")
    parser.add_argument("--com2")
    parser.add_argument("--lang1_out")
    parser.add_argument("--lang2_out")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(
        args.lang1,
        args.com1,
        args.lang2,
        args.com2,
        args.lang1_out,
        args.lang2_out
    )
