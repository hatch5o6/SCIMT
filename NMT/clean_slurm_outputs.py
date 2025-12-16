import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--top", "-t", type=int, default=1)
args = parser.parse_args()
top = args.top


dirs = [
    "/home/hatch5o6/Cognate/code/NMT/slurm_outputs",
    "/home/hatch5o6/Cognate/code/NMT/slurm_outputs/aeb-en",
    "/home/hatch5o6/Cognate/code/NMT/slurm_outputs/apc-en",
    "/home/hatch5o6/Cognate/code/NMT/slurm_outputs/an-en",
    "/home/hatch5o6/Cognate/code/NMT/slurm_outputs/as-hi",
    "/home/hatch5o6/Cognate/code/NMT/slurm_outputs/bem-en",
    "/home/hatch5o6/Cognate/code/NMT/slurm_outputs/bho-as",
    "/home/hatch5o6/Cognate/code/NMT/slurm_outputs/bho-hi",
    "/home/hatch5o6/Cognate/code/NMT/slurm_outputs/djk-en",
    "/home/hatch5o6/Cognate/code/NMT/slurm_outputs/ewe-en",
    "/home/hatch5o6/Cognate/code/NMT/slurm_outputs/fon-fr",
    "/home/hatch5o6/Cognate/code/NMT/slurm_outputs/hsb-de",
    "/home/hatch5o6/Cognate/code/NMT/slurm_outputs/mfe-en",

    "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/aeb-en",
    "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/apc-en",
    "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/an-en",
    "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/as-hi",
    "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/bem-en",
    "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/bho-as",
    "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/bho-hi",
    "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/djk-en",
    "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/ewe-en",
    "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/fon-fr",
    "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/hsb-de",
    "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/mfe-en",
]
for dir in dirs:
    if not os.path.exists(dir): continue
    fs = os.listdir(dir)
    files = {}
    for f in fs:
        if f.endswith(".py"): continue
        if f.endswith("scores.xlsx"): continue
        if f.startswith("SAVE_"): continue
        if f.startswith("SINGLE_GPU_"): continue
        if f.startswith("REGULAR_"): continue
        
        
        if os.path.isdir(os.path.join(dir, f)): continue
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

# clean core dumps

base_dir = "/home/hatch5o6/Cognate/code"
for f in os.listdir(base_dir):
    if f.startswith("core."):
        f_path = os.path.join(base_dir, f)
        print("removing", f_path)
        os.remove(f_path)