import os

def read_cfg(f):
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    settings = {}
    for line in lines:
        if line == "":
            continue
        if line.startswith("#"):
            continue
        split_line = line.split("=")
        key = split_line[0]
        value = "=".join(split_line[1:])
        assert key not in settings
        settings[key] = value
    
    # print("SETTINGS")
    # print(json.dumps(settings, ensure_ascii=False, indent=2))
    return settings

def add_pair(lang_pairs, pair, type="NMT_CLTL"):
    assert type in ["NMT_PLTL", "NMT_CLTL", "NMT_SLTL", "COG_PLCL", "COG_SLTL"]

    if pair not in lang_pairs:
        lang_pairs[pair] = []
    if type not in lang_pairs[pair]:
        lang_pairs[pair].append(type)
        lang_pairs[pair] = sorted(lang_pairs[pair])

    return lang_pairs

def is_SLTL_TL(pair1, pair2, pair3):
    # Determines whether the three pairs from a cfg refer to a PL/CL->TL scenario or a SL/TL->TL scenario
    # if the three pairs are the same, it's a SL/TL->TL scenario and we return True
    return pair1 == pair2 == pair3

path = "/home/hatch5o6/Cognate/code/Pipeline/cfg/SC"
lang_pairs = {}
for f in os.listdir(path):
    f_path = os.path.join(path, f)
    config = read_cfg(f_path)
    nmt_pair = (config["NMT_SRC"], config["NMT_TGT"])
    aug_pair = (config["AUG_SRC"], config["AUG_TGT"])
    cog_pair = (config["SRC"], config["TGT"])
    
    if is_SLTL_TL(nmt_pair, aug_pair, cog_pair):
        # is SL/TL->TL scenario
        lang_pairs = add_pair(lang_pairs, nmt_pair, "NMT_SLTL")
        lang_pairs = add_pair(lang_pairs, cog_pair, "COG_SLTL")
    else:
        # is PL/CL->TL scenario
        lang_pairs = add_pair(lang_pairs, nmt_pair, "NMT_CLTL")
        lang_pairs = add_pair(lang_pairs, aug_pair, "NMT_PLTL")
        lang_pairs = add_pair(lang_pairs, cog_pair, "COG_PLCL")


sorted_pairs = sorted([(pair, apps) for pair, apps in lang_pairs.items()])

for pair, apps in sorted_pairs:
    print(pair, apps)