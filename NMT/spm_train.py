import argparse
import sentencepiece as spm
import ast
import random
import os
import shutil
import json


def train_spm_model(
    model_name: str,
    data_file: str,
    model_type: str,
    vocab_size: int,
    character_coverage: float=1.0,
    user_defined_symbols: list=[]
):
    print("Training tokenizer")
    print("model_name:", model_name)
    print("data_file:", data_file)
    print("model_type:", model_type)
    print("vocab_size:", vocab_size)
    print("character_coverage:", character_coverage)
    print("user_defined_symbols:", user_defined_symbols)
    spm.SentencePieceTrainer.train(
        input=data_file,
        model_prefix=model_name,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        user_defined_symbols=user_defined_symbols
    )

def read_f(f):
    with open(f) as inf:
        data = [line.strip() for line in inf.readlines()]
    return data

def make_data(
    data_dict, 
    training_data_size, 
    save_file, 
    seed
):
    assert save_file.endswith(".txt")
    random.seed(seed)

    all_data = []
    all_data_divided = {}
    for data_f, ratio in data_dict.items():
        assert data_f.endswith(".txt")
        data_name = data_f.split("/")[-1][:-4]

        print("-----------------------------")
        print("data_f:", data_f)
        print("ratio:", ratio)
        data = read_f(data_f)
        random.shuffle(data)
        print("size:", len(data))

        final_data = []
        size = round(ratio * training_data_size)
        while len(final_data) < size:
            final_data += data
        assert len(final_data) >= size
        final_data = final_data[:size]
        print("After upsampling:", len(final_data))
        assert len(final_data) == size

        assert data_name not in all_data_divided # assert dataname is unique
        all_data_divided[data_name] = final_data
        all_data += final_data
    
    print("\n\nALL DATA SIZE:", len(all_data))
    
    random.shuffle(all_data)
    print(f"Writing {len(all_data)} lines to", save_file)
    with open(save_file, "w") as outf:
        outf.write("\n".join(all_data) + "\n")
    
    for data_name, data in all_data_divided.items():
        assert save_file.endswith(".txt")
        data_div_f = save_file[:-4] + f"div={data_name}.txt"
        assert data_div_f != save_file
        assert not os.path.exists(data_div_f)
        with open(data_div_f, "w") as outf:
            outf.write("\n".join(data) + "\n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--langs", help="For when --data (the data_dict) is not passed. Comma-delimited list of langs. Will search for these inside of --folder and create a data_dict.")
    parser.add_argument("-f", "--folder", help="For when --data (the data_dict) is not passed. Folder with lang data files: {lang}.txt")
    parser.add_argument("-D", "--dist_str", help="Optional argument for when --data (the data_dict) is not passed, defines distribution of data based on langs (not files). When not passed, even distribution between langs is assumed.")
    parser.add_argument("-d", "--data", required=False, help="dictionary of file paths mapping to ratio of final training data")
    parser.add_argument("-n", "--training_data_size", type=int, default=12000)
    parser.add_argument("-s", "--save_dir", required=True)
    parser.add_argument("-m", "--spm_model_name", required=True)
    parser.add_argument("-v", "--spm_vocab_size", type=int, default=8000)
    parser.add_argument("-T", "--spm_model_type", choices=["bpe", "unigram"], default="bpe")
    parser.add_argument("-c", "--character_coverage", type=float, default=1.0)
    parser.add_argument("-S", "--seed", type=int, default=1500),
    parser.add_argument("-u", "--user_defined_symbols", help="comma-delimited list of special tokens")
    parser.add_argument("--SPLIT_ON_WS", choices=['true', 'false'])
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t--{k} = '{v}'")
    return args

if __name__ == "__main__":
    print("##########################")
    print("##     spm_train.py     ##")
    print("##########################")
    args = get_args()
    if args.data != None:
        data_dict = ast.literal_eval(args.data)
        print("READ DATA_DICT:")
        print(json.dumps(data_dict, ensure_ascii=False, indent=2))
    else:
        assert args.langs is not None
        assert args.folder is not None
        assert args.dist_str is not None
        langs = [l.strip() for l in args.langs.split(",")]
        print("langs", langs)
        data_list = []
        data_dict = {}

        dists = None
        if args.dist_str is not None:
            dists = {}
            lang_dists = [lx.strip() for lx in args.dist_str.split(",")]
            for lang_dist in lang_dists:
                lx, dx = tuple(lang_dist.split(":"))
                dists[lx] = int(dx) / 100
            assert sum(dists.values()) == 1.0

        for l in langs:
            l_path = os.path.join(args.folder, f"{l}.txt")
            if os.path.exists(l_path):
                assert l_path not in data_list
                data_list.append(l_path)

                if dists is not None:
                    assert l_path not in data_dict
                    data_dict[l_path] = dists[l]

        print("DATA LIST", data_list)
        assert len(data_list) > 0
        
        if args.dist_str is None:
            assert data_dict == {}
            data_ratio = 1 / len(data_list)
            data_dict = {f: data_ratio for f in data_list}

        assert sum(data_dict.values()) == 1.0
        
        print("CREATED DATA_DICT FROM args.langs AND args.folder:")
        print(json.dumps(data_dict, ensure_ascii=False, indent=2))
    print("\n")

    if os.path.exists(args.save_dir):
        print("deleting", args.save_dir)
        shutil.rmtree(args.save_dir)
    print("creating", args.save_dir)
    os.mkdir(args.save_dir)
    save_training_data_f = os.path.join(args.save_dir, f"training_data.s={args.seed}.txt")
    model_name = os.path.join(args.save_dir, args.spm_model_name)

    make_data(
        data_dict=data_dict, 
        training_data_size=args.training_data_size,
        save_file=save_training_data_f,
        seed=args.seed
    )

    data_dict_f = os.path.join(args.save_dir, "data_dict.json")
    with open(data_dict_f, "w") as outf:
        outf.write(json.dumps(data_dict, ensure_ascii=False, indent=2))

    user_defined_symbols = []
    user_defined_symbols += [
        item.strip()
        for item in args.user_defined_symbols.split(",")
        if item.strip() != ""
    ]
    if args.SPLIT_ON_WS == 'true':
        user_defined_symbols.append('▁')

    train_spm_model(
        model_name=model_name,
        data_file=save_training_data_f,
        model_type=args.spm_model_type,
        vocab_size=args.spm_vocab_size,
        character_coverage=args.character_coverage,
        user_defined_symbols=user_defined_symbols
    )
