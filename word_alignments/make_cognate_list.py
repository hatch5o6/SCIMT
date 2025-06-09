import argparse
import Levenshtein
from tqdm import tqdm
from string import punctuation
punctuation += "—¡¿؟؛،٪»«›‹”“〞❮❯❛❟"

def get_cognates(word_list, theta=0.5):
    cognate_list = set()
    for word1, word2 in word_list:
        word1 = clean(word1)
        word2 = clean(word2)
        passed, distance = are_cognates(word1, word2, theta=theta)
        if passed:
            cognate_list.add((word1, word2, distance))
    return sorted(list(cognate_list))

def are_cognates(word1, word2, theta=0.5):
    if len(word1) == 0 or len(word2) == 0:
        return False, None
    max_len = max(len(word1), len(word2))
    distance = Levenshtein.distance(word1, word2) / max_len
    if distance <= theta:
        if not is_only_punct(word1) and not is_only_punct(word2):
            return True, distance
    return False, distance

def clean(word):
    while len(word) > 0 and removable_char(word[0]):
        word = word[1:]
    while len(word) > 0 and removable_char(word[-1]):
        word = word[:-1]
    return word.strip()

def removable_char(char):
    return char in punctuation or char.isspace()

def is_only_punct(word):
    global punctuation
    word = word.strip()
    for char in word:
        if char not in punctuation:
            return False
    return True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--word_list", required=True, help="word list created by make_word_alignments.py")
    parser.add_argument("-o", "--out")
    parser.add_argument("-t", "--theta", type=float, default=0.5, help="edit distance threshold for cognates")
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    args = parser.parse_args()
    print("arguments")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")
    print("-------------------\n\n")
    return args

if __name__ == "__main__":
    print("make_cognate_list.py")
    args = get_args()
    word_list = []
    with open(args.word_list) as inf:
        for line in inf:
            word1, word2 = tuple(line.strip().split(" ||| "))
            word_list.append((word1, word2))
    cognate_list = get_cognates(word_list, theta=args.theta)
    print(f"- FOUND {len(cognate_list)} COGNATES -")
    out = args.out
    if out is None:
        EXT = args.word_list.split(".")[-1]
        out = ".".join(args.word_list.split(".")[:-1]) + f".cognates.{args.theta}.{EXT}"
    
    with open(out, "w") as outf:
        for word1, word2, distance in cognate_list:
            outf.write(f"{word1.strip()} ||| {word2.strip()} ||| {distance}\n")

    # write parallel data
    EXT = out.split(".")[-1]
    src_out = out[:-len(EXT)] + f"parallel-{args.src}.{EXT}"
    tgt_out = out[:-len(EXT)] + f"parallel-{args.tgt}.{EXT}"
    with open(src_out, "w") as sf, open(tgt_out, "w") as tf:
        for word1, word2, distance in cognate_list:
            sf.write(word1.strip() + "\n")
            tf.write(word2.strip() + "\n")