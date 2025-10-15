import argparse
import os

from torch.utils.data import DataLoader
import sys
# Add NMT directory to path to import parallel_datasets
script_dir = os.path.dirname(os.path.abspath(__file__))
nmt_dir = os.path.join(os.path.dirname(script_dir), "NMT")
sys.path.append(nmt_dir)
from parallel_datasets import MultilingualDataset


def get_data(
    train_csv,
    val_csv,
    test_csv,
    src_out,
    tgt_out,
    src_lang,
    tgt_lang
):
    limit_src_langs = None
    limit_tgt_langs = None
    if src_lang != None:
        limit_src_langs = [src_lang]
    if tgt_lang != None:
        limit_tgt_langs = [tgt_lang]

    DATASETS = []
    for f in [train_csv, val_csv, test_csv]:
        dataset = get_dataset(f, limit_src_langs, limit_tgt_langs)
        if dataset is not None:
            DATASETS.append(dataset)
    
    with open(src_out, "w") as sf, open(tgt_out, "w") as tf:
        for dataset in DATASETS:
            dataloader = DataLoader(dataset, batch_size=100)
            for src_segs, tgt_segs in dataloader:
                batch = list(zip(src_segs, tgt_segs))
                for src_seg, tgt_seg in batch:
                    sf.write(src_seg.strip() + "\n")
                    tf.write(tgt_seg.strip() + "\n")

def get_dataset(csv_f, limit_src_langs, limit_tgt_langs):
    dataset = None
    if csv_f != "null":
        assert csv_f.endswith(".csv")
        if not os.path.exists(csv_f):
            print("DOES NOT EXIST", csv_f)
        assert os.path.exists(csv_f)

        dataset = MultilingualDataset(
            data_csv=csv_f,
            append_src_lang_tok=False,
            append_tgt_lang_tok=False,
            seed=None,
            upsample=False,
            shuffle=False,
            limit_src_langs=limit_src_langs,
            limit_tgt_langs=limit_tgt_langs
        )
    return dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", help="csv file or string 'null'")
    parser.add_argument("--val_csv", help="csv file or string 'null'")
    parser.add_argument("--test_csv", help="csv file or string 'null'")
    parser.add_argument("--src_out")
    parser.add_argument("--tgt_out")
    parser.add_argument("--src", help="for filtering src langs in the data csv file")
    parser.add_argument("--tgt", help="for filtering tgt langs in the data csv file")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t-{k}: {v}")
    print("------------------------------\n")
    return args

if __name__ == "__main__":
    print("############################")
    print("# make_SC_training_data.py #")
    print("############################")
    args = get_args()
    get_data(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        src_out=args.src_out, 
        tgt_out=args.tgt_out,
        src_lang=args.src,
        tgt_lang=args.tgt
    )