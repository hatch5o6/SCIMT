import argparse

from NMT.assert_no_data_overlap import assert_no_overlap

def main():
    pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test")
    args = parser.parse_args()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"--{k}=`{v}`")
    print("\n\n\n")
    return args

if __name__ == "__main__":
    print("################################")
    print("# assert_no_cognate_overlap.py #")
    print("################################")
    args = get_args()
    print("TEST:", args.test)
    main()