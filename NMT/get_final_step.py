import os
import csv


"""
Reads the metrics.csv for each scenario and tells me the step (approximately) training stopped at.
"""

OUTPUTS="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates"


def main():
    for d in os.listdir(OUTPUTS):
        if d in ["archive"]: continue

        print(d)
        d_path = os.path.join(OUTPUTS, d)
        assert os.path.isdir(d_path)

        for model_d in os.listdir(d_path):
            model_d_path = os.path.join(d_path, model_d)
            assert os.path.isdir(model_d_path)

            assert os.listdir(os.path.join(model_d_path, "logs/lightning_logs")) == ["version_0"]
            metrics_file = os.path.join(model_d_path, "logs/lightning_logs/version_0/metrics.csv")
            with open(metrics_file, newline='') as inf:
                rows = [r for r in csv.reader(inf)]
            assert rows[0] == ["epoch", "lr-AdamW", "step", "train_loss_epoch", "train_loss_step", "val_loss_epoch", "val_loss_step"]
            final_step = rows[-1][2]
            print("\t", model_d, "FINAL_STEP:\t\t", final_step)


if __name__ == "__main__":
    main()