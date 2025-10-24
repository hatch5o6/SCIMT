from sacrebleu.metrics import BLEU, CHRF

def calc_bleu(
    hyp, # list
    refs, # list of lists
    tokenize=None
):
    for ref in refs:
        if len(hyp) != len(ref):
            error = f"len hyp ({len(hyp)}) != len ref ({len(ref)})"
            error += "HYP:\n" + "\n".join(hyp[:3])
            error += "\nREF:\n" + "\n".join(ref[:3])
            raise ValueError(error)

    if tokenize is not None:
        # print("TOKENIZING")
        bleu = BLEU(tokenize="char")
    else:
        # print("NOT TOKENIZING")
        bleu = BLEU()
    score = bleu.corpus_score(hyp, refs)
    print("BLEU STUFF")
    print(score)
    return score.score

def calc_chrF(
    hyp, # list
    refs # list of lists
):
    for ref in refs:
        if len(hyp) != len(ref):
            error = f"len hyp ({len(hyp)}) != len ref ({len(ref)})"
            error += "HYP:\n" + "\n".join(hyp[:3])
            error += "\nREF:\n" + "\n".join(ref[:3])
            raise ValueError(error)

    chrf = CHRF()
    score = chrf.corpus_score(hyp, refs)
    sentence_scores = []
    for i in range(len(hyp)):
        h = hyp[i]
        r = [refs[0][i]]
        sentence_scores.append(chrf.sentence_score(h, r).score)
    return score.score, sentence_scores

def calc_comet22(
    src,
    hyp,
    ref
):
    from comet import download_model, load_from_checkpoint
    comet22_path = "/home/hatch5o6/nobackup/archive/comet/wmt22-comet-da/checkpoints/model.ckpt"
    comet22 = load_from_checkpoint(comet22_path)

    assert len(src) == len(hyp) == len(ref)
    data = [{
        "src": src[i],
        "mt": hyp[i],
        "ref": ref[i]
    } for i in range(len(src))]
    output = comet22.predict(data, batch_size=8, gpus=1)
    return output.system_score, output.scores

def calc_xcomet():
    pass

def calc_metrics(
    src, 
    ref, 
    hyp
):
    bleu_score = calc_bleu(hyp, [ref])
    print("BLEU:", bleu_score)
    chrf_system_score, chrf_sent_scores = calc_chrf(hyp, [ref])
    print("CHRF:", chrf_system_score)
    comet_system_score, comet_sent_scores = calc_comet22(src, hyp, ref)
    print("COMET-22:", comet_system_score)

    return bleu_score, chrf_system_score, chrf_sent_scores, comet_system_score, comet_sent_scores

def read_data(f):
    with open(f) as inf:
        data = [line.strip() for line in inf.readlines()]
    return data

if __name__ == "__main__":
    print("evaluate.py")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref")
    parser.add_argument("--hyp")
    parser.add_argument("--out")
    args = parser.parse_args()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t- {k}: `{v}`")
    
    ref = read_data(args.ref)
    hyp = read_data(args.hyp)

    bleu_score = calc_bleu(hyp=hyp, refs=[ref])
    chrf_score, chrf_sent_scores = calc_chrF(hyp=hyp, refs=[ref])

    print("BLUE:", bleu_score)
    print("chrF:", chrf_score)

    with open(args.out, "w") as outf:
        outf.write("Scores:\n")
        outf.write(f"\tREF: {args.ref}\n")
        outf.write(f"\tHYP: {args.hyp}\n")
        outf.write(f"\nBLEU: {bleu_score}\nchrF: {chrf_score}\n")

