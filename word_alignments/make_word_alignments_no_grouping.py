import argparse
from string import punctuation
from tqdm import tqdm
from collections import Counter

def make_word_list(sent_pairs, alignments, VERBOSE=False, STOP=None):
    word_list = Counter()

    data = list(zip(sent_pairs, alignments))
    ct_nbsp = 0
    for idx, ((src_sent, tgt_sent), word_alignments) in tqdm(enumerate(data), total=len(data)):
        if idx == STOP:
            break
        if VERBOSE:
            print(f"\n\n--------------------- ({idx}) ------------------------")
            print(src_sent)
            print(tgt_sent)
            print(word_alignments)
        # should already be tokenized and joined on whitespace, so no need for word_tokenize function
            
        # Found a NBSP in a fon (or ewe, but I think fon) sent, so we need to handle that.
        FOUND_NBSP = False
        if " " in src_sent:
            src_sent = src_sent.replace(" ", "<NBSP>")
            FOUND_NBSP = True
        if " " in tgt_sent:
            tgt_sent = tgt_sent.replace(" ", "<NBSP>")
            FOUND_NBSP = True

        if FOUND_NBSP:
            ct_nbsp += 1

        src_words = src_sent.split()
        src_words = replace_word_in_sent(src_words, "<NBSP>", " ")
        tgt_words = tgt_sent.split()
        tgt_words = replace_word_in_sent(tgt_words, "<NBSP>", " ")
        max_len = max(len(src_words), len(tgt_words))
        word_alignments = word_alignments.split()

        for word_alignment in word_alignments:
            w1, w2 = tuple(word_alignment.split("-"))
            w1, w2 = int(w1), int(w2)
            
            try:
                word_pair_x = (src_words[w1], tgt_words[w2])

                # don't add cognate pairs with the NBSP
                if " " not in word_pair_x:
                    word_list[word_pair_x] += 1
            except:
                print("CODE BROKEN:")
                print(f"\n\n--------------------- ({idx}) ------------------------")
                print("SRC_SENT:", src_sent)
                print("TGT_SENT:", tgt_sent)
                print(word_alignments)
                print(f"--Alignment {idx}--")
                print(word_alignment)
                print("SRC_WORDS:", len(src_words), src_words)
                print("TGT_WORDS:", len(tgt_words), tgt_words)
                exit()

    print(f"{ct_nbsp} sentence pairs have a NBSP")

    return word_list

def replace_word_in_sent(sent, word, replace_word):
    for idx, w in enumerate(sent):
        if w == word:
            sent[idx] = replace_word
    return sent

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
    print("#######################################")
    print("# make_word_alignments_no_grouping.py #")
    print("#######################################")
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

    word_list_ordered = []
    for word_pair, ct in word_list.items():
        assert isinstance(word_pair, tuple)
        assert len(word_pair) == 2
        word_list_ordered.append((ct, word_pair))
    word_list_ordered.sort(reverse=True)

    ordered_out = args.out # + ".ordered.txt"
    with open(ordered_out, "w") as outf:
        for ct, (word_a, word_b) in word_list_ordered:
            outf.write(f"{ct} ||| {word_a.strip()} ||| {word_b.strip()}\n")

    # with open(args.out, "w") as outf:
    #     for word_a, word_b in word_list.keys():
    #         outf.write(f"{word_a.strip()} ||| {word_b.strip()}\n")
