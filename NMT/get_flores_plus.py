import huggingface_hub
from datasets import load_dataset
import os
import shutil
from datetime import datetime
from tqdm import tqdm

out_dir = "/home/hatch5o6/nobackup/archive/data/FLORES+"
if os.path.exists(out_dir):
    print("DELETING", out_dir)
    shutil.rmtree(out_dir)
print("CREATING", out_dir)
os.mkdir(out_dir)
notes_path = os.path.join(out_dir, "notes.txt")

with open("flores_token.txt") as inf:
    flores_token = inf.read().strip()

huggingface_hub.login(token=flores_token)

starttime = datetime.now()
start = starttime.strftime("%m-%d-%Y %H:%M:%S")
ds_full = load_dataset("openlanguagedata/flores_plus")
print(ds_full)

def get_data(dataset):
    visited = set()
    data = {}
    langs = set()
    for item in tqdm(dataset):
        # print(item)
        idx = item["id"]
        if idx not in data:
            data[idx] = {}
        
        lang = item["iso_639_3"]
        script = item["iso_15924"]
        glotto = item["glottocode"]
        l = f"{lang}_{script}_{glotto}"
        assert l not in data[idx]
        text = item["text"]
        data[idx][l] = text
        langs.add(l)

        assert (idx, lang, script, glotto) not in visited
        visited.add((idx, lang, script, glotto))
    
    for idx in data:
        assert set(data[idx].keys()) == langs
    return data, langs

def write_data(dataset, langs, subdir="dev"):
    outsubdir = os.path.join(out_dir, subdir)
    assert not os.path.exists(outsubdir)
    os.mkdir(outsubdir)

    lang_data = {l: [] for l in langs}

    idxs = sorted([int(ix) for ix in dataset.keys()])
    # print("IDXS", len(idxs))
    # print(idxs[-20:])
    # print("RANGE", len(dataset))
    assert list(range(len(dataset))) == idxs

    for idx in range(len(dataset)):
        assert set(dataset[idx].keys()) == langs
        for lang in langs:
            lang_data[lang].append(dataset[idx][lang])
    
    for lang, data in lang_data.items():
        f = os.path.join(outsubdir, f"{lang}.{subdir}")
        with open(f, "w") as outf:
            for line in data:
                outf.write(line.strip() + "\n")

print("\nDEV")
dev = ds_full["dev"]
dev, dev_langs = get_data(dev)
print("\nWRITING DEV")
write_data(dev, dev_langs, subdir="dev")

print("\nDEVTEST")
devtest = ds_full["devtest"]
devtest, devtest_langs = get_data(devtest)
print("\nWRITING DEVTEST")
write_data(devtest, devtest_langs, subdir="devtest")

endtime = datetime.now()
end = endtime.strftime("%m-%d-%Y %H:%M:%S")
with open(notes_path, "w") as outf:
    outf.write("SOURCE: https://huggingface.co/datasets/openlanguagedata/flores_plus,\n\tand for Aragonese, https://huggingface.co/datasets/openlanguagedata/flores_plus/blob/main/dataset_cards/arg_Latn.md\n")
    outf.write(f"Download began {start}\n")
    outf.write(f"Download ended {end}\n")
