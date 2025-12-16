import argparse
import Levenshtein
from tqdm import tqdm
from string import punctuation
punctuation += "—¡¿؟؛،٪»«›‹”“〞❮❯❛❟"

# --------- Written by ChatGPT for canonicalize function ----------- #
import unicodedata
# load TR39 confusables
_confusable_map = {}
# -------------------- #

def get_cognates(word_list, edit_dist_type="LED", theta=0.5):
    cognate_list = set()
    for ct, word1, word2 in word_list:
        word1 = clean(word1)
        word2 = clean(word2)
        passed, distance = are_cognates(word1, word2, edit_dist_type=edit_dist_type, theta=theta)
        if passed:
            cognate_list.add((ct, word1, word2, distance))
    return sorted(list(cognate_list), reverse=True)

def are_cognates(word1, word2, edit_dist_type="NLED", theta=0.5):
    assert edit_dist_type in ["NLED", "unicode_normalized_NLED"]
    if edit_dist_type == "NLED":
        # print("USING LEVENSHTEIN DISTANCE")
        edit_dist_func = NLED
    elif edit_dist_type == "unicode_normalized_NLED":
        # print("USING UNICODE-NORMALIZED LEVENSHTEIN DISTANCE")
        edit_dist_func = unicode_normalized_NLED

    if len(word1) == 0 or len(word2) == 0:
        return False, None
    distance = edit_dist_func(word1, word2)
    if distance <= theta:
        if not is_only_punct(word1) and not is_only_punct(word2):
            return True, distance
    return False, distance

def NLED(word1, word2):
    max_len = max(len(word1), len(word2))
    return Levenshtein.distance(word1, word2) / max_len

def unicode_normalized_NLED(word1, word2):
    norm_word1 = canonicalize(word1.lower())
    norm_word2 = canonicalize(word2.lower())
    max_len = max(len(norm_word1), len(norm_word2))
    return Levenshtein.distance(norm_word1, norm_word2) / max_len

# --------- Written by ChatGPT ----------- #

with open("word_alignments/confusables/confusables.txt", encoding="utf-8") as f:
    for line in f:
        if not line or line.startswith("#"):
            continue
        # format: <src> ; <target> ; <type> # comment
        parts = line.split(";")
        if len(parts) < 2:
            continue
        src = parts[0].strip()
        target = parts[1].strip()
        # src and target are hex codepoints space separated
        src_char = "".join(chr(int(cp, 16)) for cp in src.split())
        target_char = "".join(chr(int(cp, 16)) for cp in target.split())
        _confusable_map[src_char] = target_char

def _apply_tr39_skeleton(text: str) -> str:
    out = []
    for ch in text:
        out.append(_confusable_map.get(ch, ch))
    return "".join(out)

def canonicalize(text: str) -> str:
    # 1) NFKD normalize
    text = unicodedata.normalize("NFKD", text)

    # 2) remove combining marks
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Mn"
    )

    # 3) apply TR39 skeleton
    text = _apply_tr39_skeleton(text)

    return text
# -------------------- #

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
    parser.add_argument("-edt", "--edit_dist_type", default="NLED", choices=["unicode_normalized_NLED", "NLED"])
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
            ct, word1, word2 = tuple(line.strip().split(" ||| "))
            ct = int(ct.strip())
            word1 = word1.strip()
            word2 = word2.strip()
            word_list.append((ct, word1, word2))
    cognate_list = get_cognates(word_list, edit_dist_type=args.edit_dist_type, theta=args.theta)
    print(f"- FOUND {len(cognate_list)} COGNATES -")
    out = args.out
    if out is None:
        EXT = args.word_list.split(".")[-1]
        out = ".".join(args.word_list.split(".")[:-1]) + f".cognates.{args.theta}.{EXT}"
    
    with open(out, "w") as outf:
        for ct, word1, word2, distance in cognate_list:
            outf.write(f"{ct} ||| {word1.strip()} ||| {word2.strip()} ||| {distance}\n")
    
    cognate_list_by_distance = sorted([(distance, ct, word1, word2) for ct, word1, word2, distance in cognate_list], reverse=True)
    with open(out + ".byNED.txt", "w") as outf:
        for distance, ct, word1, word2 in cognate_list_by_distance:
            outf.write(f"{ct} ||| {word1.strip()} ||| {word2.strip()} ||| {distance}\n")

    # write parallel data
    EXT = out.split(".")[-1]
    src_out = out[:-len(EXT)] + f"parallel-{args.src}.{EXT}"
    tgt_out = out[:-len(EXT)] + f"parallel-{args.tgt}.{EXT}"
    with open(src_out, "w") as sf, open(tgt_out, "w") as tf:
        for ct, word1, word2, distance in cognate_list:
            sf.write(word1.strip() + "\n")
            tf.write(word2.strip() + "\n")