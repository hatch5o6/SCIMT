from sacrebleu.metrics import BLEU, CHRF
from Levenshtein import distance

def calc_bleu(
    hyp, # list
    refs, # list of lists
    tokenize=None
):
    for ref in refs:
        assert len(hyp) == len(ref)

    if tokenize is not None:
        bleu = BLEU(tokenize="char")
    else:
        bleu = BLEU()
    score = bleu.corpus_score(hyp, refs)
    return score.score

def calc_chrF(
    hyp, # list
    refs # list of lists
):
    for ref in refs:
        assert len(hyp) == len(ref)

    chrf = CHRF()
    score = chrf.corpus_score(hyp, refs)
    sentence_scores = []
    for i in range(len(hyp)):
        h = hyp[i]
        r = [refs[0][i]]
        sentence_scores.append(chrf.sentence_score(h, r).score)
    return score.score, sentence_scores

def calc_NED(
    hyp,
    ref
):
    assert len(hyp) == len(ref)
    pairs = list(zip(hyp, ref))
    scores = []
    for hyp_seq, ref_seq in pairs:
        max_len = max(len(hyp_seq), len(req_seq))
        NED = distance(hyp_seq, ref_seq) / max_len
        scores.append(NED)
    avg = sum(scores) / len(scores)
    return avg, scores

def calc_accuracy(
    hyp,
    ref
):
    assert len(hyp) == len(ref)
    pairs = list(zip(hyp, ref))
    scores = []
    for hyp_seq, ref_seq in pairs:
        if hyp_seq.strip() == ref_seq.strip():
            scores.append(1)
        else:
            scores.append(0)
    acc = sum(scores) / len(scores)
    return acc, scores

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


if __name__ == "__main__":
    src = ["hello, my friend", "please give me a game."]
    ref = ["اهلا وسهلا يا صديقي", "عطيني لعبا لو سمحت"]
    hyp = ["مرحبا صاحبي شو أخبارك؟", "لو ارادت عطيني لعبا"]
    bleu_score = calc_bleu(hyp, [ref])
    print("BLEU:", bleu_score)
    chrf_score, chrf_sent_scores = calc_chrf(hyp, [ref])
    print("CHRF:", chrf_score)
    print(chrf_sent_scores)
    comet_score, comet_sent_scores = calc_comet22([src], [hyp], [ref])
    print("COMET-22:", comet_score)
    print(comet_sent_scores)
