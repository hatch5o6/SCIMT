import argparse
import os
from datetime import datetime

FINISHED_STAMP = """
################# (Lightning) #################
#######          TRAINING ENDED         #######
###############################################
""".strip()

def get_timestamps(experiments_dir, lang_pair):
    lang_pair_dir = os.path.join(experiments_dir, lang_pair)
    print("looking in", lang_pair_dir)
    f_timestamps = []
    for d in os.listdir(lang_pair_dir):
        d_path = os.path.join(lang_pair_dir, d)
        if os.path.isdir(d_path):
            if d == "mt_eval":
                f_timestamps += get_dir_timestamps(d_path)
            else:
                for sub_d in os.listdir(d_path):
                    sub_d_path = os.path.join(d_path, sub_d)
                    assert sub_d in ["checkpoints", "logs", "predictions", "tb", f"predictions_{lang_pair}_test"], f"invalid sub_d: {sub_d_path}"
                    if sub_d == "checkpoints":
                        f_timestamps += get_dir_timestamps(sub_d_path)
                    elif sub_d in ["logs", "tb"]:
                        sub_d_path = os.path.join(sub_d_path, "lightning_logs/version_0")
                        f_timestamps += get_dir_timestamps(sub_d_path)
                    elif sub_d in ["predictions", f"predictions_{lang_pair}_test"]:
                        all_scores_f = os.path.join(sub_d_path, "all_scores.json")
                        f_timestamps.append((all_scores_f, get_timestamp(all_scores_f)))
                        for pred_sub_d in os.listdir(sub_d_path):
                            if pred_sub_d == "all_scores.json": continue
                            pred_sub_d_path = os.path.join(sub_d_path, pred_sub_d)
                            f_timestamps += get_dir_timestamps(pred_sub_d_path)
    
    for idx, (f, timestamp) in enumerate(f_timestamps):
        if is_first_of_model(idx, f_timestamps):
            _, model = get_model(f)
            print(f"\n\n\n########################################")
            print("#", model)
            print(f"########################################")
        if is_first_of_kind(idx, f_timestamps):
            kind = get_kind(f)
            print(f"\n-----------------------| {kind} |------------------------")
        print(timestamp, f)
        
def is_first_of_kind(idx, f_timestamps):
    if idx == 0:
        return True
    kind = get_kind(f_timestamps[idx][0])
    prev_kind = get_kind(f_timestamps[idx - 1][0])
    if kind != prev_kind:
        return True
    else:
        return False

def is_first_of_model(idx, f_timestamps):
    if idx == 0:
        return True
    lang_pair, model = get_model(f_timestamps[idx][0])
    prev_lang_pair, prev_model = get_model(f_timestamps[idx - 1][0])
    if model != prev_model:
        return True
    else:
        return False

def get_model(f):
    f = f.split("/PredictCognates/")[1]
    lang_pair = f.split("/")[0]
    model = f.split("/")[1]
    return lang_pair, model


def get_kind(f):
    if "/mt_eval/" in f:
        return "mt_eval"
    elif "/checkpoints/" in f:
        return "checkpoints"
    elif "/predictions/" in f:
        return "predictions"
    elif "/predictions_" in f:
        return "predictions_test"
    elif "/logs/" in f:
        return "logs"
    elif "/tb/" in f:
        return "tb"
    assert False, f"`{f}` is of an invalid kind!"

def get_timestamp(f):
    timestamp = os.path.getmtime(f)
    mt_datetime = datetime.fromtimestamp(timestamp)
    return mt_datetime.strftime('%Y-%m-%d %H:%M:%S')

def get_dir_timestamps(d):
    f_timestamps = []
    for f in os.listdir(d):
        f_path = os.path.join(d, f)
        f_timestamps.append((f_path, get_timestamp(f_path)))
    return f_timestamps

def finished_training(experiments_dir, lang_pair):
    lang_pair_dir = os.path.join(experiments_dir, lang_pair)
    print("looking in", lang_pair_dir)
    for f in os.listdir(lang_pair_dir):
        f_path = os.path.join(lang_pair_dir, f)
        if os.path.isfile(f_path):
            # print(f)
            slurm_id = f.split("_")[0]
            rest = "_".join(f.split("_")[1:])
            # print(rest)
            if rest.startswith("TRAIN."):
                passed = has_finished_stamp(f_path)
                # if not passed:
                print(f"\t-`{f_path}` {passed}")

def has_finished_stamp(f):
    with open(f) as inf:
        content = inf.read()
    return FINISHED_STAMP in content


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["finished_training", "file_timestamps"], default="file_timestamps")
    parser.add_argument("-e", "--experiments_dir", default="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates")
    parser.add_argument("-l", "--lang_pair")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.mode == "file_timestamps":
        get_timestamps(args.experiments_dir, args.lang_pair)
    else:
        finished_training(args.experiments_dir, args.lang_pair)
