import argparse
from string import punctuation
from tqdm import tqdm
from collections import Counter

def make_word_list(sent_pairs, alignments, VERBOSE=False, STOP=None):
    word_list = Counter()

    data = list(zip(sent_pairs, alignments))
    for idx, ((src_sent, tgt_sent), word_alignments) in tqdm(enumerate(data), total=len(data)):
        if idx == STOP:
            break
        if VERBOSE:
            print(f"\n\n--------------------- ({idx}) ------------------------")
            print(src_sent)
            print(tgt_sent)
            print(word_alignments)
        # should already be tokenized and joined on whitespace, so no need for word_tokenize function
        src_words = src_sent.split()
        tgt_words = tgt_sent.split()
        max_len = max(len(src_words), len(tgt_words))
        word_alignments = word_alignments.split()

        src_word_align_dict = {i:[] for i in range(len(src_words))}
        tgt_word_align_dict = {i:[] for i in range(len(tgt_words))}
        keys = set([i for i in range(max_len)])
        for word_alignment in word_alignments:
            w1, w2 = tuple(word_alignment.split("-"))
            w1, w2 = int(w1), int(w2)

            src_word_align_dict[w1].append(w2)
            tgt_word_align_dict[w2].append(w1)

            # word_list.append((src_words[w1], tgt_words[w2]))
        
        for k, v in src_word_align_dict.items():
            src_word_align_dict[k] = make_groups(v)
        for k, v in tgt_word_align_dict.items():
            tgt_word_align_dict[k] = make_groups(v)
        if VERBOSE:
            print("\n\n- final alignments -")
            print("\tsrc_word_align_dict", src_word_align_dict)
            print("\ttgt_word_align_dict", tgt_word_align_dict)
            print("")
        for w in sorted(list(keys)):
            src_word, tgt_word = "", ""
            if w < len(src_words):
                src_word = src_words[w]
            if w < len(tgt_words):
                tgt_word = tgt_words[w]

            src_word_tgts = [i for i in src_word_align_dict.get(w, [])]
            
            if VERBOSE:
                print("\t-", w, src_word, tgt_word)
                print("\t\tsrc_word_tgts groups", src_word_tgts)
            for group in src_word_tgts:
                group_words = [tgt_words[i] for i in group]
                if VERBOSE:
                    print("\t\t\tadding", f"'{src_word}'", group_words)
                word_list[(src_word, " ".join(group_words))] += 1
            
            tgt_word_srcs = [i for i in tgt_word_align_dict.get(w, [])]
            if VERBOSE:    
                print("\t\ttgt_word_srcs groups", tgt_word_srcs)
            for group in tgt_word_srcs:
                group_words = [src_words[i] for i in group]
                if VERBOSE:
                    print("\t\t\tadding", group_words, f"'{tgt_word}'")
                word_list[(" ".join(group_words), tgt_word)] += 1

    return word_list

def make_groups(a_list):
    # takes a list of ints, and groups the continuous ints together
    # e.g. [1,2,4,5] -> [[1,2], [4,5]]

    a_list.sort()
    groups = []
    group = []
    for n in a_list:
        if len(group) == 0:
            group.append(n)
        else:
            if n == group[-1] + 1:
                group.append(n)
            else:
                groups.append(group)
                group = [n]
    if len(group) > 0:
        groups.append(group)
    return groups


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alignments", help="word alignments file from fast_align")
    parser.add_argument("-s", "--sent_pairs", help="sentence pairs, delimited with ' ||| '")
    parser.add_argument("-o", "--out")
    parser.add_argument("--VERBOSE", action="store_true")
    parser.add_argument("--STOP", type=int, default=None)
    args = parser.parse_args()
    print("arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: '{v}'")
    print("\n")
    return args

if __name__ == "__main__":
    print("make_word_alignments.py")
    args = get_args()
    with open(args.alignments) as inf:
        alignments = [line.strip() for line in inf]
    sent_pairs = []
    with open(args.sent_pairs) as inf:
        for line in inf:
            line = line.strip()
            src, tgt = tuple(line.split(" ||| "))
            sent_pairs.append((src.strip(), tgt.strip()))

    word_list = make_word_list(
        sent_pairs, 
        alignments,
        VERBOSE=args.VERBOSE,
        STOP=args.STOP
    )

    with open(args.out, "w") as outf:
        for word_a, word_b in word_list.keys():
            outf.write(f"{word_a.strip()} ||| {word_b.strip()}\n")