import argparse
import os

"""
I used this script to compare the djk-en cognate lists to that of en-djk.
"""

def main(
    cog1_f,
    cog2_f,
    write_to=None
):
    cog_set_1, equivalences_1 = read_cog_f(cog1_f)
    cog_set_2, equivalences_2 = read_cog_f(cog2_f)
    cog_int = cog_set_1.intersection(cog_set_2)
    unique_to_1 = cog_set_1.difference(cog_set_2)
    unique_to_2 = cog_set_2.difference(cog_set_1)

    print(cog1_f)
    print("\tLEN:", len(cog_set_1))
    print("\tPAIRS WHERE WORD 1 = WORD 2:", equivalences_1)
    print("\tUNIQUE TO IT:", len(unique_to_1))

    print("\n\n", cog2_f)
    print("\tLEN:", len(cog_set_2))
    print("\tPAIRS WHERE WORD 1 = WORD 2:", equivalences_2)
    print("\tUNIQUE TO IT:", len(unique_to_2))

    print("\n\nINTERSECTION:", len(cog_int))

    if write_to is not None:
        pass

# def write_cogs(cog_set, path):
#     with open(path, "w") as outf:
#         for word1, word2 in 

def read_cog_f(f):
    cog_sets = set()
    n_same = 0
    with open(f) as inf:
        for line in inf.readlines():
            word1, word2, score = line.strip().split(" ||| ")
            word1 = word1.strip()
            word2 = word2.strip()
            score = float(score.strip())
            pair = tuple(sorted([word1, word2]))
            if word1 == word2:
                n_same += 1
            # assert pair not in cog_sets
            cog_sets.add(pair)
    return cog_sets, n_same

def compare_parallel(
    folder_1,
    lang_pair_1,
    folder_2,
    lang_pair_2
):
    train_pairs_1 = get_pairs(folder_1, lang_pair=lang_pair_1, div="train")
    train_pairs_2 = get_pairs(folder_2, lang_pair=lang_pair_2, div="train")
    run_comparison(
        pairs1=train_pairs_1,
        folder1=folder_1,
        pairs2=train_pairs_2,
        folder2=folder_2,
        div="train"
    )

    val_pairs_1 = get_pairs(folder_1, lang_pair=lang_pair_1, div="val")
    val_pairs_2 = get_pairs(folder_2, lang_pair=lang_pair_2, div="val")
    run_comparison(
        pairs1=val_pairs_1,
        folder1=folder_1,
        pairs2=val_pairs_2,
        folder2=folder_2,
        div="val"
    )

    test_pairs_1 = get_pairs(folder_1, lang_pair=lang_pair_1, div="test")
    test_pairs_2 = get_pairs(folder_2, lang_pair=lang_pair_2, div="test")
    run_comparison(
        pairs1=test_pairs_1,
        folder1=folder_1,
        pairs2=test_pairs_2,
        folder2=folder_2,
        div="test"
    )

def get_pairs(folder, lang_pair, div="train", NG=".NG", thresh=0.5, seed=0):
    assert div in ["train", "test", "val"]
    src, tgt = tuple([l.strip() for l in lang_pair.split("-")])

    src_f = os.path.join(folder, f"word_list.{src}-{tgt}{NG}.cognates.{str(thresh)}.parallel-{src}.{div}-s={seed}.txt")
    tgt_f = os.path.join(folder, f"word_list.{src}-{tgt}{NG}.cognates.{str(thresh)}.parallel-{tgt}.{div}-s={seed}.txt")
    pairs = read_pairs(src_f, tgt_f)
    return pairs

def read_pairs(f1, f2):
    with open(f1) as inf:
        lines1 = [l.strip() for l in inf.readlines()]
    with open(f2) as inf:
        lines2 = [l.strip() for l in inf.readlines()]
    assert len(lines1) == len(lines2)
    pairs_list = list(zip(lines1, lines2))
    pairs_set = set(pairs_list)
    assert len(pairs_set) == len(pairs_list)
    return pairs_set

def run_comparison(pairs1, folder1, pairs2, folder2, div="train"):
    assert div in ["train", "val", "test"]
    intersect, unique_to_1, unique_to_2 = compare_pairs(pairs1, pairs2)
    print(f"---------------------- {div.upper()} ----------------------")
    print(folder1, ":")
    print("LEN:", len(pairs1))
    print("UNIQUE TO IT:", len(unique_to_1))
    print("\n\n", folder2, ":")
    print("LEN:", len(pairs2))
    print("UNIQUE TO IT:", len(unique_to_2))

    print("\n\nINTERSECTION:", len(intersect))

def compare_pairs(pairs1, pairs2):
    assert isinstance(pairs1, set)
    assert isinstance(pairs2, set)
    pairs2_fixed = set([
        (word2, word1)
        for word1, word2 in pairs2
    ])

    unique_to_1 = pairs1.difference(pairs2_fixed)
    unique_to_2 = pairs2_fixed.difference(pairs1)
    intersect = pairs1.intersection(pairs2_fixed)
    return intersect, unique_to_1, unique_to_2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_pair_1", help="format `xx-yy`, e.g. `djk-en` (for mode 'parallel_sets')")
    parser.add_argument("--cog1")
    parser.add_argument("--lang_pair_2", help="format `xx-yy`, e.g. `en-djk` (for mode 'parallel_sets')")
    parser.add_argument("--cog2")
    parser.add_argument("--write_to")
    parser.add_argument("--mode", choices=["pipe_delimited", "parallel_sets"], default="parallel_sets")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.mode == "pipe_delimited":
        # will pass in files of cognates in format `{word_1} ||| {word_2} ||| {score}`, as in `Agrippa ||| Agiipa ||| 0.2857142857142857`
        main(args.cog1, args.cog2, args.write_to)
    elif args.mode == "parallel_sets":
        # will pass in folder names containing train, val, and test parallel sets. Requires lang pairs be passed as well.
        compare_parallel(
            folder_1=args.cog1,
            lang_pair_1=args.lang_pair_1,
            folder_2=args.cog2,
            lang_pair_2=args.lang_pair_2
        )