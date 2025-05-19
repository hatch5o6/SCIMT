import argparse
import yaml
from torch.utils.data import DataLoader

from parallel_datasets import MultilingualDataset

def test(config):
    train_dataset = MultilingualDataset(
        data_csv=config["train_data"],
        # append_src_lang_tok=config["append_src_token"],
        # append_tgt_lang_tok=config["append_tgt_token"]
        append_src_lang_tok=True,
        append_tgt_lang_tok=True
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["train_batch_size"]
    )
    print("TRAIN $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print_some(train_dataloader)

    val_dataset = MultilingualDataset(
        data_csv=config["val_data"],
        # append_src_lang_tok=config["append_src_token"],
        # append_tgt_lang_tok=config["append_tgt_token"]
        append_src_lang_tok=True,
        append_tgt_lang_tok=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
    )
    print("VAL $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print_some(val_dataloader)

def print_some(dataloader):
    b = 0
    for src_segments, tgt_segments in dataloader:
        print(f"################## {b} ###################")
        batch = list(zip(src_segments, tgt_segments))
        for i, (src, tgt) in enumerate(batch):
            print(f"--------- {i} ----------")
            print("SRC:")
            print(f"\t\"{src}\"")
            print("TGT:")
            print(f"\t\"{tgt}\"")
            if i > 7:
                break
        b += 1
        if b == 40:
            break

def read_config(f):
    with open(f) as inf:
        config = yaml.safe_load(inf)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t--{k} = '{v}'")
    return args

if __name__ == "__main__":
    print("----------------------")
    print("###### test_datasets.py ######")
    print("----------------------")
    args = get_args()
    config = read_config(args.config)
    test(config)
    
