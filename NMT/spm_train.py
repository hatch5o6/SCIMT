import argparse
import sentencepiece as spm
import ast
import random
import os
import shutil


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
    random.seed(seed)

    all_data = []
    for data_f, ratio in data_dict.items():
        data = read_f(data_f)
        random.shuffle(data)

        final_data = []
        size = round(ratio * training_data_size)
        while len(final_data) < size:
            final_data += data
        assert len(final_data) >= size
        final_data = final_data[:size]
        assert len(final_data) == size
        all_data += final_data
    
    random.shuffle(all_data)
    print(f"Writing {len(all_data)} lines to", save_file)
    with open(save_file, "w") as outf:
        outf.write("\n".join(all_data) + "\n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, help="dictionary of file paths mapping to ratio of final training data")
    parser.add_argument("-n", "--training_data_size", type=int, default=12000)
    parser.add_argument("-s", "--save_dir", required=True)
    parser.add_argument("-m", "--spm_model_name", required=True)
    parser.add_argument("-v", "--spm_vocab_size", type=int, default=8000)
    parser.add_argument("-T", "--spm_model_type", choices=["bpe", "unigram"], default="bpe")
    parser.add_argument("-c", "--character_coverage", type=float, default=1.0)
    parser.add_argument("-S", "--seed", type=int, default=1500),
    parser.add_argument("-u", "--user_defined_symbols", help="comma-delimited list of special tokens")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t--{k} = '{v}'")
    return args

if __name__ == "__main__":
    args = get_args()
    data_dict = ast.literal_eval(args.data)

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

    user_defined_symbols = [
        item.strip()
        for item in args.user_defined_symbols.split(",")
    ]

    train_spm_model(
        model_name=model_name,
        data_file=save_training_data_f,
        model_type=args.spm_model_type,
        vocab_size=args.spm_vocab_size,
        character_coverage=args.character_coverage,
        user_defined_symbols=user_defined_symbols
    )
