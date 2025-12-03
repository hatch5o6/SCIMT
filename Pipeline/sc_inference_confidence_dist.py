import argparse
import os
import shutil
import csv
from tqdm import tqdm
import re

def main(
    inf_output,
    out_dir
):
    if os.path.exists(out_dir):
        print(f"Deleting {out_dir}")
        shutil.rmtree(out_dir)
    print(f"Creating {out_dir}")
    os.mkdir(out_dir)

    results = read_CopperMT_Results(inf_output)
    len_results = 0
    buckets = {-i / 10: [] for i in list(range(10))}
    buckets[-1] = []
    print("buckets:", buckets)
    for source_word, hyps in results.items():
        assert isinstance(hyps, tuple) or isinstance(hyps, list)
        if isinstance(hyps, tuple):
            hyps = [hyps]
        len_results += len(hyps)
        
        for hyp, h_id, h_conf in hyps:
            ADDED = False
            times_added = 0
            for b in buckets:
                # print("b", b)
                if b > -1:
                    if h_conf <= b and h_conf > b - 0.1:
                        buckets[b].append((source_word, hyp, h_id, h_conf))
                        ADDED = True
                        times_added += 1
                        break
                else:
                    assert b == (-1)
                    # print(f"bucket = -1")
                    if h_conf <= b:
                        buckets[b].append((source_word, hyp, h_id, h_conf))
                        ADDED = True
                        # print("\tadded", (source_word, hyp, h_id, h_conf))
                        times_added += 1
                        break
            assert times_added == 1, f"RNN ({hyp}, {h_id}, {h_conf}) added to more than one bucket!"
            assert ADDED, f"RNN ({hyp}, {h_id}, {h_conf}) not added to a bucket!"
    
    total = 0
    b_lengths = {}
    for b, results in buckets.items():
        total += len(results)
        b_lengths[b] = len(results)
        out_f = os.path.join(out_dir, f"{b}-results.txt")
        with open(out_f, "w", newline='') as outf:
            writer = csv.writer(outf)
            writer.writerow(["source", "hyp", "id", "confidence"])
            for source_word, hyp, h_id, h_conf in results:
                writer.writerow([source_word, hyp, h_id, h_conf])
    
    out_dist = os.path.join(out_dir, "distribution.txt")
    f = open(out_dist, "w")
    print("N RESULTS:", len_results, file=f)
    print("TOTAL: ", total, file=f)
    for b, length in b_lengths.items():
        print(f"{b}: {length}, {(length / total) * 100}%", file=f)
    f.close()


def smt_main(
    smt_src,
    smt_hyp,
    out_dir
):
    if os.path.exists(out_dir):
        print(f"Deleting {out_dir}")
        shutil.rmtree(out_dir)
    print(f"Creating {out_dir}")
    os.mkdir(out_dir)

    results = read_CopperMT_SMT_RESULTS(smt_src, smt_hyp)
    len_results = len(results)
    buckets = {-i: [] for i in list(range(20)) + [20]}
    print("buckets:", buckets)
    for source_word, hyp in results.items():
        assert isinstance(hyp, tuple)
        hyp, idx, score = hyp
        ADDED = False
        times_added = 0
        for b in buckets:
            if b == 0:
                if score > -1:
                    buckets[b].append((source_word, hyp, idx, score))
                    ADDED = True
                    times_added += 1
                    break
            elif b > -20:
                if score <= b and score > b - 1:
                    buckets[b].append((source_word, hyp, idx, score))
                    ADDED = True
                    times_added += 1
                    break
            else:
                assert b == -20
                if score <= b:
                    # print(f"score {score} <= -20")
                    buckets[b].append((source_word, hyp, idx, score))
                    ADDED = True
                    times_added += 1
                    break
        assert times_added == 1, f"SMT ({hyp}, {idx}, {score}) added to more than one bucket!"
        assert ADDED, f"SMT ({hyp}, {idx}, {score}) not added to a bucket!"
    
    total = 0
    b_lengths = {}
    for b, results in buckets.items():
        total += len(results)
        b_lengths[b] = len(results)
        out_f = os.path.join(out_dir, f"{b}-results.txt")
        with open(out_f, "w", newline='') as outf:
            writer = csv.writer(outf)
            writer.writerow(["source", "hyp", "id", "score"])
            for source_word, hyp, idx, score in results:
                writer.writerow([source_word, hyp, idx, score])
    
    out_dist = os.path.join(out_dir, "distribution.txt")
    f = open(out_dist, "w")
    print("N RESULTS: ", len_results, file=f)
    print("TOTAL: ", total, file=f)
    for b, length in b_lengths.items():
        print(f"{b}: {length}, {(length / total) * 100}%", file=f)
    f.close()
                


def read_CopperMT_Results(results_f, RETURN_SPACED=False):
    with open(results_f) as inf:
        lines = [line.strip() for line in inf.readlines()]
    
    # print("\nLAST 10 LINES IN LINES")
    # for line in lines[-10:]:
    #     print(line)
    # print("\n\n")

    data_rows = []
    for line in lines:
        split_line = line.split("|")
        if len(split_line) == 4:
            assert split_line[1].strip() == "INFO"
            continue
        elif line.startswith("Generate test with beam="):
            continue
        else:
            # should be a good line :)
            assert any([
                line.startswith("S-"),
                line.startswith("T-"),
                line.startswith("H-"),
                line.startswith("D-"),
                line.startswith("P-"),
            ])
            data_rows.append(line)
    
    # print("\nLAST 10 LINES IN DATA ROWS")
    # for line in data_rows[-10:]:
    #     print(line)
    # print("\n\n")

    print("Blocking CopperMT")
    data = []
    block = []
    for i, line in tqdm(enumerate(data_rows), total=len(data_rows)):
        if i > 0 and i % 5 == 0:
            data.append(tuple(block))
            block = []
        block.append(line.strip())
    if len(block) > 0:
        assert len(block) == 5
        data.append(tuple(block))

    # print("\nLAST 3 BLOCKS IN DATA")
    # for block in data[-3:]:
    #     print(block)
    # print("\n\n")

    # print("Data")
    # for i in data[:5]:
    #     print(type(i), i)

    print("Getting Copper MT results")
    # results = {'<unk>': []}
    results = {}
    # results_list = []
    visited_ids = set()
    for S, T, H, D, P in tqdm(data):
        assert S.startswith("S-")
        assert T.startswith("T-")
        assert H.startswith("H-")
        assert D.startswith("D-")
        assert P.startswith("P-")
        
        s_id = int(S.split()[0].strip().split("-")[1])
        S = S.split("\t")[-1].strip()
        t_id = int(T.split()[0].strip().split("-")[1])
        T = T.split("\t")[-1].strip()
        h_conf = float(H.split("\t")[-2].strip())
        h_id = int(H.split()[0].strip().split("-")[1])
        H = H.split("\t")[-1].strip()
        d_id = int(D.split()[0].strip().split("-")[1])
        D = D.split("\t")[-1].strip()

        assert s_id == t_id == h_id == d_id
        assert h_id not in visited_ids
        visited_ids.add(h_id)

        assert H == D

        if RETURN_SPACED:
            source = S
            hyp = (H, h_id, h_conf)
        else:
            source = "".join(S.split())
            hyp = ("".join(H.split()), h_id, h_conf)
        # results_list.append((source, hyp))
        # if source in results:
        #     print("SOURCE IN RESULTS")
        #     print("source:", source)
        #     print("hyp:", hyp)
        #     print(f"results['{source}']:", results[source])
        if "<unk>" not in source:
            assert source not in results
            results[source] = hyp
        else:
            if source not in results:
                results[source] = hyp
            else:
                # print(f"TRYING TO SET results['{source}'] (='{results[source]}') to {hyp}")
                # assert results[source] == hyp
                if isinstance(results[source], tuple):
                    results[source] = [results[source]]
                assert isinstance(results[source], list)
                if hyp not in results[source]:
                    results[source].append(hyp)

    # if RETURN_LIST:
    #     return results_list
    # else:
    return results


def read_smt_result(f):
    with open(f) as inf:
        data = [
            re.sub(r'\s+', '', line.strip())
            for line in inf.readlines()
        ]
    return data

def read_smt_result_scores(f):
    pred_scores = {}
    with open(f) as inf:
        for line in inf.readlines():
            idx, pred, notes, score = tuple([item.strip() for item in line.strip().split("|||")])
            idx = int(idx)
            pred = re.sub(r'\s+', '', pred)
            score = float(score)
            if idx not in pred_scores:
                pred_scores[idx] = {}
            assert pred not in pred_scores[idx]
            pred_scores[idx][pred] = score
    return pred_scores

def get_hyp_scores_f(f):
    EXT = f.split(".")[-1]
    TAG = f.split(".")[-2]
    assert TAG == "hyp"
    scores_f = ".".join(f.split(".")[:-2]) + ".10_best." + EXT
    return scores_f

def read_CopperMT_SMT_RESULTS(results_src, results_hyp):
    results_hyp_scores = get_hyp_scores_f(results_hyp)

    results_src = read_smt_result(results_src)
    results_hyp = read_smt_result(results_hyp)
    results_hyp_scores = read_smt_result_scores(results_hyp_scores)
    assert len(results_src) == len(results_hyp)
    results_list = list(zip(results_src, results_hyp))

    results = {}
    for i, (src_seg, tgt_seg) in enumerate(results_list):
        # print("idx", i, "src", src_seg, "tgt", tgt_seg)
        # print(f"results_hyp_scores[{i}]:", results_hyp_scores[i])
        score = results_hyp_scores[i][tgt_seg]

        tgt_seg = (tgt_seg, i, score)

        if src_seg not in results:
            results[src_seg] = tgt_seg
        else:
            assert results[src_seg] == tgt_seg
    return results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_output", "-i")
    parser.add_argument("--out_dir", "-o")
    parser.add_argument("--smt_src", "-s")
    parser.add_argument("--smt_hyp", "-H")
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"-{k}=`{v}`")
    print("\n\n")
    return args

if __name__ == "__main__":
    print("###################################")
    print("# sc_inference_confidence_dist.py #")
    print("###################################")
    args = get_args()
    if args.inference_output:
        main(args.inference_output, args.out_dir)
    else:
        assert args.smt_src is not None
        assert args.smt_hyp is not None
        smt_main(args.smt_src, args.smt_hyp, args.out_dir)
