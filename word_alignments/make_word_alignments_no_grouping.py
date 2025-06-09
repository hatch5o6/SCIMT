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

        for word_alignment in word_alignments:
            w1, w2 = tuple(word_alignment.split("-"))
            w1, w2 = int(w1), int(w2)

            word_list[(src_words[w1], tgt_words[w2])] += 1

    return word_list

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
    print("---------------------\n\n")
    return args

if __name__ == "__main__":
    print("make_word_alignments_no_grouping.py")
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
