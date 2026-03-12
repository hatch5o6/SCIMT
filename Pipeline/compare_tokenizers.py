import argparse
import os
import random

def main(
    spm_model_dir,
    langs1,
    langs2
):
    assert len(langs1) == len(langs2) == 3
    tok_name_1 = tok_name(langs1)
    tok_name_2 = tok_name(langs2)
    tok1_files, tok1_vocab = get_files(spm_model_dir, tok_name_1, langs1)
    tok2_files, tok2_vocab = get_files(spm_model_dir, tok_name_2, langs2)
    file_pairs = list(zip(tok1_files, tok2_files))

    for idx, (f1, f2) in enumerate(file_pairs):
        print("#########################################################")
        assert idx in [0, 1, 2]
        print(f"Comparing `{f1}`\n\tto `{f2}`")
        if read_content(f1) == read_content(f2):
            print("Match exactly")
            print(f"\tComparing vocabs:\n\t\t-`{tok1_vocab}`\n\t\t-`{tok2_vocab}`")
            vocab_passed = read_content(tok1_vocab) == read_content(tok2_vocab)
            if vocab_passed:
                print("\t\tmatch :)")
            else:
                print("vocabs don't match :(")
        else:
            f1_lines = read_lines(f1)
            f2_lines = read_lines(f2)
            if not len(f1_lines) == len(f2_lines):
                print(f"LINES MISMATCH:\n\t-{len(f1_lines)} lines: `{f1}`\n\t-{len(f2_lines)} lines: `{f2}`")
            else:
                pairs = list(zip(f1_lines, f2_lines))
                
                samples = list(range(4)) + \
                    list(range(len(pairs) - 4, len(pairs))) + \
                    list(random.sample([p for p in range(len(pairs))], k=8))
                samples.sort()
                matches = 0
                for i in samples:
                    print(f"--------------------- {i} ------------------------")
                    f1_line, f2_line = pairs[i]
                    response = "#"
                    while response not in ["Y", "N"]:
                        response = input(f"EQUAL?\n\t-`{f1_line}`\n\t-`{f2_line}`\n(Y or N)>").strip().upper()
                    if response == "Y":
                        matches += 1
                    else:
                        print("!!!Failed!!!")
                if matches != len(samples):
                    print("FAILED")
                    print(f"ONLY {matches}/{len(samples)} MATCH")
                else:
                    print("ALL MATCH :)")


def read_lines(f):
    with open(f) as inf:
        lines = [l.rstrip() for l in inf.readlines()]
    return lines

def read_content(f):
    with open(f) as inf:
        content = inf.read()
    return content

def tok_name(langs):
    l1,l2,l3 = langs
    return f"{l1}-{l2}_{l3}"

def get_sc_tok_name(langs):
    l1, l2, l3 = langs
    l1a, l1b = l1.split("2")
    assert l1b == l2
    return f"SC_{l1a}2{l2}-{l2}_{l3}"

def get_files(spm_model_dir, tok_name, langs):
    l1,l2,l3 = langs
    
    is_sc_scen = are_sc(langs)

    if is_sc_scen:
        tok_name = get_sc_tok_name(langs)

    file_dir = os.path.join(spm_model_dir, tok_name, tok_name)

    if is_sc_scen:
        vocab = os.path.join(file_dir, f"SC_{l1}-{l2}_{l3}.vocab")
    else:
        vocab = os.path.join(file_dir, f"{l1}-{l2}_{l3}.vocab")
    
    f1 = get_data_file(file_dir, l1)
    f2 = get_data_file(file_dir, l2)
    f3 = get_data_file(file_dir, l3)
    
    return (f1, f2, f3), vocab

def are_sc(langs):
    l1, l2, l3 = langs
    assert "2" not in l2
    assert "2" not in l3
    return "2" in l1

def get_data_file(file_dir, lang):
    if "2" in lang:
        lang, l1b = lang.split("2")
        assert l1b.strip() != ""
    return os.path.join(file_dir, f"training_data.s=1500div={lang}.txt")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spm_models", default="/home/hatch5o6/nobackup/archive/CognateMT/spm_models")
    parser.add_argument("--langs1", default="es,an,en")
    parser.add_argument("--langs2", default="esx,anx,enx")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    langs1 = [l.strip() for l in args.langs1.split(",")]
    langs2 = [l.strip() for l in args.langs2.split(",")]
    main(
        spm_model_dir=args.spm_models,
        langs1=langs1,
        langs2=langs2
    )