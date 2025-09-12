import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--top", "-t", type=int, default=1)
args = parser.parse_args()
top = args.top


dirs = [
    "/home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/slurm_outputs", 
    "/home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/test_slurm_outputs",
    "/home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/hyper_param_search_outputs",
    "/home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/SC_smt"
]
for dir in dirs:
    fs = os.listdir(dir)
    files = {}
    for f in fs:
        if f.endswith(".py"): continue
        if f.startswith("SAVE_"): continue
        number = f.split("_")[0]
        number = int(number)
        name = "_".join(f.split("_")[1:])
        if name not in files:
            files[name] = []
        
        files[name].append((number, f))

    for name, fs in files.items():
        fs.sort(reverse=True)

        for n, f in fs[top:]:
            f_path = os.path.join(dir, f)
            os.remove(f_path)
