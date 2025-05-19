import argparse

def calc_overlap(
    data_f,
    spm1,
    spm2,
    out_f
):
    pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--spm1")
    parser.add_argument("--spm2")
    parser.add_argument("--out")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t--{k} = '{v}'")
    return args

if __name__ == "__main__":
    print("------------------------------")
    print("###### token_overlap.py ######")
    print("------------------------------")
    args = get_args(
        args.data,
        args.spm1,
        args.spm2,
        args.out
    )


