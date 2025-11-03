import argparse
import csv
import os
import matplotlib.pyplot as plt 
from tqdm import tqdm

# Example dir: /home/hatch5o6/nobackup/archive/ThesisSRE/TRIALS/trials_ccmatrix_bpe/ccmatrix_bpe-trial_0
def write_loss_graph(
    trial_dir
):
    version_dirs = os.path.join(trial_dir, "logs/lightning_logs")
    for version_dir in os.listdir(version_dirs):
        version_dir = os.path.join(version_dirs, version_dir)

        # clean out previously created .png files
        for f in os.listdir(version_dir):
            if f.endswith(".png"):
                fpath = os.path.join(version_dir, f)
                os.remove(fpath)

        metrics_csv = os.path.join(version_dir, "metrics.csv")

        epoch_train_losses = []
        epoch_val_losses = []
        step_train_losses = []
        step_val_losses = []
        with open(metrics_csv, newline='') as inf:
            r = 0
            for row in csv.reader(inf):
                # print(row)
                # print(len(row))
                if r == 0:
                    # assert row == ["epoch","lr-AdamW","step","train_loss_epoch","train_loss_step","val_loss_epoch","val_loss_step"]
                    assert row == [
                        "epoch","grad_norm","lr-AdamW","step",
                        "train_loss_epoch","train_loss_step",
                        # "train_src_max_seq_len","train_tgt_max_seq_len",
                        "val_loss_epoch","val_loss_step",
                        # "val_src_max_seq_len","val_tgt_max_seq_len"
                    ]
                else:
                    row = tuple(row)
                    # epoch, lr_adam, step, train_loss_epoch, train_loss_step, val_loss_epoch, val_loss_step = tuple(row)
                    (epoch, grad_norm, lr_adam, step,
                     train_loss_epoch, train_loss_step, 
                    #  train_src_max_seq_len, train_tgt_max_seq_len,
                     val_loss_epoch, val_loss_step, 
                    #  val_src_max_seq_len, val_tgt_max_seq_len
                     ) = tuple(row)
                    if train_loss_epoch.strip() != "":
                        assert epoch.strip() != ""
                        epoch_train_losses.append((int(step), float(train_loss_epoch)))
                    if val_loss_epoch.strip() != "":
                        assert epoch.strip() != ""
                        epoch_val_losses.append((int(step), float(val_loss_epoch)))
                    if train_loss_step.strip() != "":
                        assert step.strip() != ""
                        step_train_losses.append((int(step), float(train_loss_step)))
                    if val_loss_step.strip() != "":
                        assert step.strip() != ""
                        step_val_losses.append((int(step), float(val_loss_step)))
                r += 1
        plt.figure(0)
        a, b = zip(*epoch_train_losses)
        plt.plot(a, b, label="train_loss_epoch")
        a, b = zip(*epoch_val_losses)
        plt.plot(a, b, label="val_loss_epoch")
        plt.xlabel("steps")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(os.path.join(version_dir, "epoch_loss.png"))
        plt.clf()

        plt.figure(1)
        a, b = zip(*step_train_losses)
        plt.plot(a, b, label="train_loss_step")
        plt.xlabel("steps")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(os.path.join(version_dir, "train_step_loss.png"))
        plt.clf()

        plt.figure(2)
        a, b = zip(*step_val_losses)
        plt.plot(a, b, label="val_loss_step")
        plt.xlabel("steps")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(os.path.join(version_dir, "val_step_loss.png"))
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/mfe-en", help="language pair directory holding training results for all models")
    args = parser.parse_args()

    print("MAKING GRAPHS FOR", args.dir)
    for d in tqdm(os.listdir(args.dir)):
        d_path = os.path.join(args.dir, d)
        write_loss_graph(d_path)