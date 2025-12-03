import argparse
import os
import shutil

def get_checkpoint(
    directory
):
    all_bleu_f  = os.path.join(directory, "bleu/all_bleu.csv")

    src, tgt = None, None
    best_chkpt = None
    best_score = None
    with open(all_bleu_f) as inf:
        lines = [tuple(line.strip().split()) for line in inf.readlines()]
        assert len(lines[0]) == 0
        lines = lines[1:]
        for i, (chkpt, src_lang, tgt_lang, score) in enumerate(lines):
            # find best score in list
            # # print(i, chkpt, src_lang, tgt_lang, score)
            # score = float(score)
            # if i == 0:
            #     assert src == None
            #     assert tgt == None
            #     src = src_lang
            #     tgt = tgt_lang
            
            # assert src_lang == src
            # assert tgt_lang == tgt

            # if best_score == None:
            #     assert best_chkpt == None
            #     best_score = score
            #     best_chkpt = chkpt
            # elif score > best_score:
            #     best_score = score
            #     best_chkpt = chkpt

            # get best based on checkpoint_best.pt name
            if chkpt == "checkpoint_best.pt":
                assert best_chkpt == None
                assert best_score == None
                best_chkpt = chkpt
                best_score = score
                break

    assert best_chkpt is not None
    assert best_score is not None

    selected_checkpoint_f = os.path.join(directory, f"checkpoints/{best_chkpt}")
    copy_to_f = os.path.join(directory, f"checkpoints/selected.pt")
    print("Copying Selected Checkpoint", selected_checkpoint_f)
    print("\tto", copy_to_f)
    shutil.copyfile(selected_checkpoint_f, copy_to_f)
    print("\tdone")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t-{k}: {v}")
    print("------------------------------\n")
    return args

if __name__ == "__main__":
    print("##########################")
    print("# select_checkpoint.py #")
    print("##########################")
    args = get_args()
    get_checkpoint(args.dir)