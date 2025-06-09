import argparse
import ast
import random
import os
import shutil
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Whitespace, Digits

#TODO WS TOKENS? SO THAT DECODING IS RIGHT?
def train_tokenizer(
    save_model: str,
    data_file: str,
    vocab_size: int,
    special_tokens: list=["<unk>", "<s>", "</s>", "<pad>"],
    UNK: str="<unk>"
):
    print("Training tokenizer")
    print("save_model:", save_model)
    print("data_file:", data_file)
    print("vocab_size:", vocab_size)
    print("special_tokens:", special_tokens)
    
    tokenizer = Tokenizer(BPE(unk_token=UNK))
    pretokenizer = Sequence([Whitespace(), Digits(individual_digits=True)])
    tokenizer.pre_tokenizer = pretokenizer
    trainer = BpeTrainer(
        special_tokens=special_tokens,
        vocab_size=vocab_size
    )
    tokenizer.train(
        files=[data_file], 
        trainer=trainer
    )
    tokenizer.save(save_model)

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
        all_data += final_data
    
    print("\n\nALL DATA SIZE:", len(all_data))
    
    random.shuffle(all_data)
    print(f"Writing {len(all_data)} lines to", save_file)
    with open(save_file, "w") as outf:
        outf.write("\n".join(all_data) + "\n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, help="dictionary of file paths mapping to ratio of final training data")
    parser.add_argument("-n", "--training_data_size", type=int, default=12000)
    parser.add_argument("-s", "--save_dir", required=True)
    parser.add_argument("-m", "--model_name", required=True)
    parser.add_argument("-v", "--vocab_size", type=int, default=8000)
    parser.add_argument("-S", "--seed", type=int, default=1500),
    parser.add_argument("-st", "--special_tokens", default="<unk>,<s>,</s>,<pad>", help="comma-delimited list of special tokens")
    parser.add_argument("-unk", "--unknown", default="<unk>")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t--{k} = '{v}'")
    return args

if __name__ == "__main__":
    # bert = BertPreTokenizer()
    # dig = Digits(individual_digits=True)
    # seq = Sequence([bert, dig])
    # ws = Whitespace()
    # seq2 = Sequence([ws, dig])
    # wss = WhitespaceSplit()

    # sent = "I want 746 marbles, and ٣٤٥ marshmallows."
    # print(sent)
    # print("BERT:", [i for i, _ in bert.pre_tokenize_str(sent)])
    # print("DIGIT:", [i for i, _ in dig.pre_tokenize_str(sent)])
    # print("WS:", [i for i, _ in ws.pre_tokenize_str(sent)])
    # print("WSS:", [i for i, _ in wss.pre_tokenize_str(sent)])
    # print("\nSEQ:", [i for i, _ in seq.pre_tokenize_str(sent)])
    # print("SEQ2:", [i for i, _ in seq2.pre_tokenize_str(sent)])
    # exit()

    args = get_args()
    data_dict = ast.literal_eval(args.data)

    if os.path.exists(args.save_dir):
        print("deleting", args.save_dir)
        shutil.rmtree(args.save_dir)
    print("creating", args.save_dir)
    os.mkdir(args.save_dir)
    save_training_data_f = os.path.join(args.save_dir, f"training_data.s={args.seed}.txt")
    model_path = os.path.join(args.save_dir, args.model_name) + ".json"

    make_data(
        data_dict=data_dict, 
        training_data_size=args.training_data_size,
        save_file=save_training_data_f,
        seed=args.seed
    )

    special_tokens = [
        item.strip()
        for item in args.special_tokens.split(",")
    ]

    train_tokenizer(
        save_model=model_path,
        data_file=save_training_data_f,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        UNK=args.unknown
    )
