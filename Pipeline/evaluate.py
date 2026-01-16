import sacrebleu
from sacrebleu import TOKENIZERS

UNK_TOK_STANDIN = "ξ"

def calc_bleu(
    hyp, # list
    refs # list of lists
):
    for ref in refs:
        if len(hyp) != len(ref):
            error = f"len hyp ({len(hyp)}) != len ref ({len(ref)})"
            error += "HYP:\n" + "\n".join(hyp[:3])
            error += "\nREF:\n" + "\n".join(ref[:3])
            raise ValueError(error)

    # # With tokenizer_13a
    # tokenizer_13a = TOKENIZERS['13a']()
    # tokenized_hyp = [tokenizer_13a(h) for h in hyp]
    # tokenized_refs = []
    # for ref_set in refs:
    #     tokenized_refs.append([tokenizer_13a(r) for r in ref_set])
    # score = sacrebleu.corpus_bleu(tokenized_hyp, tokenized_refs, tokenize="none")
    
    # # char tokenizer and unspacing text first
    # unspaced_hyp = [h.replace(" ", "") for h in hyp]
    # unspaced_refs = []
    # for ref_set in refs:
    #     unspaced_refs.append([r.replace(" ", "") for r in ref_set])
    # score = sacrebleu.corpus_bleu(unspaced_hyp, unspaced_refs, tokenize="char")

    # char tokenizer on spaced text
    score = sacrebleu.corpus_bleu(hyp, refs, tokenize="char")

    print("BLEU STUFF")
    print(score)
    return score

def read_data(f):
    with open(f) as inf:
        data = [line.strip() for line in inf.readlines()]
    return data

def read_vocab_file(f):
    vocab = set()
    with open(f) as inf:
        for line in inf.readlines():
            line = line.rstrip()
            tok, _ = tuple(line.split())
            assert tok not in vocab
            vocab.add(tok)
    return vocab

def read_unique_chars_in_ref_file(f):
    unique_chars = set()
    with open(f) as inf:
        for line in inf.readlines():
            line = line.rstrip()
            chars = line.split()
            unique_chars.update(chars)
    return unique_chars

def read_bleu_from_fairseq_hyp(f):
    with open(f) as inf:
        lines = [l for l in inf.readlines()]
    last = lines[-1].strip()
    prefix = "Generate valid with beam=10: "
    assert last.startswith(prefix)
    bleu_stuff = last[len(prefix):]
    return bleu_stuff

if __name__ == "__main__":
    print("evaluate.py")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref")
    parser.add_argument("--hyp")
    parser.add_argument("--out")
    parser.add_argument("--target_vocab", default="null")
    parser.add_argument("--hyp_out_txt", default="null", help="path to output from the CopperMT inference (made from fairseq-generate in main_nmt_bilingual_full_brendan_PREDICT.sh), should end with generate-valid.txt")
    parser.add_argument("--REPLACE_UNK", action="store_true", default=False, help="if passed, will replace unknown tokens `?` and `>` with `<unk>` in reference")
    args = parser.parse_args()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t- {k}: `{v}`")
    
    ref = read_data(args.ref)
    if args.REPLACE_UNK and args.target_vocab != "null":
        print("REPLACING OOV WITH <unk>:")
        target_vocab = read_vocab_file(args.target_vocab)
        ref_vocab = read_unique_chars_in_ref_file(args.ref)
        oov = ref_vocab.difference(target_vocab)
        assert UNK_TOK_STANDIN not in oov, f"UNK_TOK_STANDIN `{UNK_TOK_STANDIN}` is in oov!"
        print(f"OUT OF VOCAB TOKS (len: {len(oov)}): {oov}")
        for rx, r in enumerate(ref):
            # print("-----")
            # r = r.replace("?", "<<unk>>")
            # r = r.replace(">", "<<unk>>")
            new_r = r
            for tok in oov:
                # print("replacing", tok, "with <<unk>>")
                # print(f"replacing `{tok}` with <<unk>>")
                new_r = new_r.replace(tok, UNK_TOK_STANDIN)
            new_r = new_r.replace(UNK_TOK_STANDIN, "<unk>")
            if r != new_r:
                print(f"converted `{r}` to `{new_r}`")
            ref[rx] = new_r

    hyp = read_data(args.hyp)

    bleu_score = calc_bleu(hyp=hyp, refs=[ref])
    if args.hyp_out_txt != "null":
        fairseq_bleu = read_bleu_from_fairseq_hyp(args.hyp_out_txt)
        print("\n")
        if str(bleu_score) != fairseq_bleu:
            print(f"***BLEU SCORE EVALUATION DOES NOT MATCH THAT OF FAIRSEQ!!!***")
            print(f"EVAL `{str(bleu_score)}` != FAIRSEQ `{fairseq_bleu}`")
        else:
            print(f":) BLEU SCORE EVALUATION MATCHES THAT OF FAIRSEQ :)")
            print(f"EVAL `{str(bleu_score)}` == FAIRSEQ `{fairseq_bleu}`")
        print("\n")
    else:
        fairseq_bleu = None
            
    print("BLUE:", bleu_score)
    with open(args.out, "w") as outf:
        outf.write("Scores:\n")
        outf.write(f"\tREF: {args.ref}\n")
        outf.write(f"\tHYP: {args.hyp}\n")
        outf.write(f"\nBLEU_DETAILS: {bleu_score}\nBLEU_SCORE: {bleu_score.score}\n")

        if fairseq_bleu is not None:
            outf.write(f"FAIRSEQ_BLEU: {fairseq_bleu}\n")
            if str(bleu_score) != fairseq_bleu:
                outf.write("***BLEU SCORE EVALUATION DOES NOT MATCH THAT OF FAIRSEQ!!!***\n")
            else:
                outf.write(":) BLEU SCORE EVALUATION MATCHES THAT OF FAIRSEQ :)\n")

