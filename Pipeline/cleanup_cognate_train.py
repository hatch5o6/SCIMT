import os
import shutil
from tqdm import tqdm

COGNATE_TRAIN = "/home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN"

# NOTE THIS DELETED THE ATT DATA (wooops)

to_save = {
    "ar-aeb": 195,
    "ar-apc": 283,
    "bho-hi": 249,
    "bn-as": 246,
    "djk-en": 9,
    "en-djk": 27,
    "es-an": 213,
    "ewe-fon": 120,
    "fon-ewe": 117,
    "fr-mfe": 102,
    "hi-bho": 81,
    "lua-bem": 249,
    "fr-oc": 251,
    "esx-anx": 284,
    "frx-mfx": 132,
    "fry-mfy": 192
}

print("Cleaning data:")
for d in tqdm(os.listdir(COGNATE_TRAIN)):
    pair = d.split("_")[0]
    id = to_save.get(pair)
    if id is not None:
        id = f"RNN-{id}_"

    if id is None or (id not in d and "SMT" not in d):
        d_path = os.path.join(COGNATE_TRAIN, d)
        # print("foudn d_path", d_path)
        shutil.rmtree(d_path)

print("Cleaning checkpoints")
COPPER_MT = "/home/hatch5o6/nobackup/archive/CopperMT"
for d in tqdm(os.listdir(COPPER_MT)):
    d_path = os.path.join(COPPER_MT, d)
    d_split = d.split("-")
    src = d_split[0].lower()
    tgt = d_split[1].lower()
    pair = f"{src}-{tgt}"
    # print("d_path:", d_path)
    for i in range(20):
        chkpt = os.path.join(d_path, f"workspace/reference_models/bilingual/rnn_{pair}/0/checkpoints/checkpoint{i + 1}.pt")
        if os.path.exists(chkpt):
            # print("found checkpoint", chkpt)
            os.remove(chkpt)