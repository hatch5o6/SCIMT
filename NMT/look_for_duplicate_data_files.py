import os
import csv

DATA_DIR = "/home/hatch5o6/Cognate/code/NMT/data/PLAIN"

def main():
    data_paths = {}
    for d in os.listdir(DATA_DIR):
        d_path = os.path.join(DATA_DIR, d)
        assert os.path.isdir(d_path)
        for f in os.listdir(d_path):
            f_path = os.path.join(d_path, f)
            f_data = read_csv(f_path)
            for src_lang, tgt_lang, src_path, tgt_path in f_data:
                if src_path not in data_paths:
                    data_paths[src_path] = []
                if tgt_path not in data_paths:
                    data_paths[tgt_path] = []

                data_paths[src_path].append(("src", src_lang, f_path))
                data_paths[tgt_path].append(("tgt", tgt_lang, f_path))
    
    print("DATA FILES APPEARING IN MULTIPLE CSVs:")
    for path, csv_list in data_paths.items():
        assert len(csv_list) >= 1
        if len(csv_list) > 1:
            print("\t-", path)
            for TAG, lang, csv_path in csv_list:
                print(f"\t\t- {TAG} - {lang} - `{csv_path}`")

def read_csv(f):
    with open(f, newline='') as inf:
        rows = [tuple(r) for r in csv.reader(inf)]
    assert rows[0] == ("src_lang", "tgt_lang", "src_path", "tgt_path")
    data = rows[1:]
    return data


if __name__ == "__main__":
    main()