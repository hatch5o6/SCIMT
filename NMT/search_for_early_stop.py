import os
from tqdm import tqdm

early_stop_signal_1 = "Monitored metric val_loss did not improve in the last 10 records."
early_stop_signal_2 = "Monitored metric val_loss did not improve in the last 5 records."
slurm_outputs_dir = "/home/hatch5o6/Cognate/code/NMT/slurm_outputs"

def main():
    DID_NOT_EARLY_STOP = []
    slurm_dirs = [d for d in os.listdir(slurm_outputs_dir) if not d.endswith(".out")]
    for i, d in enumerate(slurm_dirs):
        d_path = os.path.join(slurm_outputs_dir, d)
        if not os.path.isdir(d_path): continue
        print(f"Looking in {d} ({i}/{len(slurm_dirs)})")
        for f in tqdm(os.listdir(d_path)):
            if f == "archive" or "_TEST" in f: continue
            f_path = os.path.join(d_path, f)
            # print("\tLooking in", f)
            if not early_stop_happened(f_path):
                DID_NOT_EARLY_STOP.append(f_path)
    
    print(f"Found {len(DID_NOT_EARLY_STOP)} models that did not early stop:")
    for fd in DID_NOT_EARLY_STOP:
        print(f"\t-`{fd}`")

def early_stop_happened(f):
    f_content = read_f(f)
    if early_stop_signal_1 in f_content or early_stop_signal_2 in f_content:
        return True
    return False

def read_f(f):
    with open(f, errors="ignore") as inf:
        content = inf.read()
    return content

if __name__ == "__main__":
    main()
