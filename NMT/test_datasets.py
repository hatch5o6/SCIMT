import argparse
import yaml
from torch.utils.data import DataLoader

from parallel_datasets import MultilingualDataset
from train import get_multilingual_dataloaders

def definitive_test(config):
    print("definitive_test()")
    print("GETTING DATALOADERS")
    print("\n\nGetting train and val dataloaders")
    dataloaders, datasets = get_multilingual_dataloaders(
        config, 
        sections=["train", "val"],
        RETURN_DATASETS=True
    )
    
    train_dataloader = dataloaders["train"]
    train_dataset = datasets["train"]
    print("\n\nTESTING TRAINING")
    test_train(train_dataloader, train_dataset)

    val_dataloader = dataloaders["val"]
    val_dataset = datasets["val"]
    print("\n\nTESTING VALIDATION")
    test_val_test(val_dataloader, val_dataset)

    print("\n\nGetting Test dataloader")
    dataloaders, datasets = get_multilingual_dataloaders(
        config, 
        sections=["test"],
        RETURN_DATASETS=True
    )

    test_dataloader = dataloaders["test"]
    test_dataset = datasets["test"]
    print("\n\nTESTING TEST")
    test_val_test(test_dataloader, test_dataset)
    print("\n\n################## PASSED ALL TESTS ####################")


def test_train(dataloader, dataset):
    print("SRC paths:", dataset.src_paths)
    print("TGT paths:", dataset.tgt_paths)
    src_data = read_from_paths(dataset.src_paths)
    tgt_data = read_from_paths(dataset.tgt_paths)

    dataloader_src_data = []
    dataloader_tgt_data = []

    for src_lines, tgt_lines in dataloader:
        dataloader_src_data += src_lines
        dataloader_tgt_data += tgt_lines

    print("asserting data from files is not the same as data from dataloaders")
    assert src_data != dataloader_src_data
    assert tgt_data != dataloader_tgt_data
    print("however, asserting that the data from files are the same lengths as data from dataloaders")
    assert len(src_data) == len(dataloader_src_data)
    assert len(tgt_data) == len(dataloader_tgt_data)

    for i in range(10):
        print(f"---------------- {(i)} ------------------")
        print(f"SRC___________: `{src_data[i]}`")
        print(f"TGT___________: `{tgt_data[i]}`")
        print(f"DATALOADER SRC: `{dataloader_src_data[i]}`")
        print(f"DATALOADER TGT: `{dataloader_tgt_data[i]}`")

    pairs = list(zip(src_data, tgt_data))
    dataloader_pairs = list(zip(dataloader_src_data, dataloader_tgt_data))

    print("\n(pairs)")
    for i in range(10):
        print(f"---------------- {(i)} ------------------")
        print(f"PAIR___________:", pairs[i])
        print(f"DATALOADER PAIR:", dataloader_pairs[i])

    print("asserting pairs != dataloader_pairs")
    assert pairs != dataloader_pairs
    print("assert len(pairs) == len(dataloader_pairs)")
    assert len(pairs) == len(dataloader_pairs)

    pairs = sorted(pairs)
    dataloader_pairs = sorted(dataloader_pairs)
    print("asserting sorted pairs == sorted dataloader pairs")
    assert pairs == dataloader_pairs
    print("\n(sorted pairs)")
    for i in range(10):
        print(f"---------------- {(i)} ------------------")
        print(f"PAIR___________:", pairs[i])
        print(f"DATALOADER PAIR:", dataloader_pairs[i])
    print("PASSED TEST!")


def test_val_test(dataloader, dataset):
    print("asserting src_paths[] and tgt_paths[] paths are length 1")
    assert len(dataset.src_paths) == 1
    assert len(dataset.tgt_paths) == 1
    print("SRC paths:", dataset.src_paths)
    print("TGT paths:", dataset.tgt_paths)
    src_data = read_from_paths(dataset.src_paths)
    tgt_data = read_from_paths(dataset.tgt_paths)

    dataloader_src_data = []
    dataloader_tgt_data = []

    for src_lines, tgt_lines in dataloader:
        dataloader_src_data += src_lines
        dataloader_tgt_data += tgt_lines

    print("asserting data straight from files is the same as data from dataloader")
    assert src_data == dataloader_src_data
    assert tgt_data == dataloader_tgt_data

    for i in range(10):
        print(f"---------------- {(i)} ------------------")
        print(f"SRC___________: `{src_data[i]}`")
        print(f"TGT___________: `{tgt_data[i]}`")
        print(f"DATALOADER SRC: `{dataloader_src_data[i]}`")
        print(f"DATALOADER TGT: `{dataloader_tgt_data[i]}`")

    print("PASSED TEST!")

def read_from_paths(paths):
    data = []
    for p in paths:
        with open(p) as inf:
            data += [line.strip() for line in inf.readlines()]
    return data

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
    print("##############################")
    print("###### test_datasets.py ######")
    print("##############################")
    args = get_args()
    config = read_config(args.config)
    # test(config)
    definitive_test(config)
    
