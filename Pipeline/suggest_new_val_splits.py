import csv
import argparse
import json

def main(csv_log_f, ratio=0.10, min_val_size=200, max_val_size=1000):
    log = read_log(csv_log_f)
    # recs = {}
    print("RECOMMENDATIONS")
    for lang_pair, train, val, test, _ in log:
        # assert lang_pair not in recs
        total = int(train) + int(val)
        suggested_val_size = ratio * total
        suggested_val_size = max(min_val_size, suggested_val_size)
        suggested_val_size = min(max_val_size, suggested_val_size)
        suggested_val_ratio = suggested_val_size / total
        counts = {
            "total": total,
            "suggested_val_size": suggested_val_size,
            "suggested_train_ratio": 1.0 - suggested_val_ratio,
            "suggested_val_ratio": suggested_val_ratio
        }
        print("-----------------------------")
        print(lang_pair)
        print(json.dumps(counts, indent=2))
        print("\n")


def read_log(f):
    with open(f) as inf:
        lines = [l for l in csv.reader(inf)]
    assert lines[0] == ["lang_pair","train","val","test","date_of_count"]
    return lines[1:]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_log", "-l", default="/home/hatch5o6/Cognate/code/cognate_dataset_log_NG=True.csv")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args.csv_log)
