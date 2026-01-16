import argparse

def main(ref, fairseq_generate_out):
    ref_words = sorted(read_ref(ref))
    fairseq_Ts = sorted(read_Ts_from_fairseq_generate_out(fairseq_generate_out))

    ref_words = [r.encode('utf-8') for r in ref_words]
    fairseq_Ts = [f.encode('utf-8') for f in fairseq_Ts]
    if ref_words == fairseq_Ts:
        print("The reference file, and fairseq Ts are exactly the same! :)")
    else:
        print("There are differences!!")
        print("Ref words:", len(ref_words))
        print("Fairseq Ts:", len(fairseq_Ts))

        print("NON-MATCHES:")
        n = max(len(ref_words), len(fairseq_Ts))
        n_non_matches = 0
        for i in range(n):
            r = f = None
            if i < len(ref_words):
                r = ref_words[i]
            if i < len(fairseq_Ts):
                f = fairseq_Ts[i]
            
            if r != f:
                print(f"{i}) ref: `{r}`, fairseq_T: `{f}`")
                n_non_matches += 1
        print("\nTOTAL NON-MATCHES:", n_non_matches)

        print("FINDING NON-MATCHES AGAIN")
        ref_words_set = set(ref_words)
        fairseq_Ts_set = set(fairseq_Ts)
        unique_to_ref = ref_words_set.difference(fairseq_Ts_set)
        unique_to_fairseq = fairseq_Ts_set.difference(ref_words_set)
        print("REFS w/ no match")
        for ur in unique_to_ref:
            print(f"\t`{ur}`")
        print("FAIRSEQ Ts w/ no match")
        for fr in unique_to_fairseq:
            print(f"\t`{fr}`")
    
def read_ref(f):
    with open(f) as inf:
        data = [l.strip() for l in inf.readlines()]
    return data

def read_Ts_from_fairseq_generate_out(f):
    Ts = []
    with open(f) as inf:
        for line in inf.readlines():
            line = line.rstrip()
            bar_split = line.split("|")
            if len(bar_split) > 1 and line.split("|")[1].strip() == "INFO": continue
            if line.startswith("Generate valid with beam="): continue
            if line.startswith("Generate test with beam="): continue

            assert any([
                line.startswith("S-"),
                line.startswith("T-"),
                line.startswith("H-"),
                line.startswith("D-"),
                line.startswith("P-")
            ]), f"line `{line}` does not start with S-, T-, H-, D-, P-"

            if line.startswith("T-"):
                T_tag, word = tuple(line.split("\t"))
                Ts.append(word)
    return Ts

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ref", help="path to reference file")
    parser.add_argument("-f", "--fairseq_out", help="path to fairseq-generate output", default="/home/hatch5o6/nobackup/archive/CopperMT/ES-AN-RNN-0_RNN-0_S-0/workspace/reference_models/bilingual/rnn_es-an/0/results/test_on_val_selected_checkpoint_es_an.an/generate-valid.txt")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t--{k}=`{v}`")
    return args

if __name__ == "__main__":
    print("#################################################")
    print("# check_fairseq_T_against_raw_text_reference.py #")
    print("#################################################")
    args = get_args()
    main(args.ref, args.fairseq_out)
    