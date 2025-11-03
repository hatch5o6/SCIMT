import argparse
import os
import shutil
import csv
from tqdm import tqdm

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
    buckets = {-i / 10: [] for i in range(10)}
    for source_word, hyps in results.items():
        assert isinstance(hyps, tuple) or isinstance(hyps, list)
        if isinstance(hyps, tuple):
            hyps = [hyps]
        
        for hyp, h_id, h_conf in hyps:
            for b in buckets:
                if h_conf <= b and h_conf > b - 0.1:
                    buckets[b].append((source_word, hyp, h_id, h_conf))
    
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
        if source in results:
            print("SOURCE IN RESULTS")
            print("source:", source)
            print("hyp:", hyp)
            print(f"results['{source}']:", results[source])
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_output", "-i")
    parser.add_argument("--out_dir", "-o")
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
    main(args.inference_output, args.out_dir)