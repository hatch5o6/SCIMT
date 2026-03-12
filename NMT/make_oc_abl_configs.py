import argparse
import yaml
import os
import shutil

def main(
   config_dir,
   new_config_dir,
   og_langs,
   new_langs     
):
    if os.path.exists(new_config_dir):
        print("DELETING", new_config_dir)
        shutil.rmtree(new_config_dir)
    print("MAKING", new_config_dir)
    os.mkdir(new_config_dir)

    for f in os.listdir(config_dir):
        if is_no_grad_clip(f) or is_augment(f) or is_char(f):
            continue

        new_f = get_new_file_name(f, og_langs, new_langs)
        
        f_path = os.path.join(config_dir, f)
        new_f_path = os.path.join(new_config_dir, new_f)
        make_new_config(f_path, new_f_path, og_langs, new_langs)

def make_new_config(config_f, new_config_f, og_langs, new_langs):
    og_pl, og_cl, og_tl = og_langs
    new_pl, new_cl, new_tl = new_langs

    config_lines = read_f(config_f)
    config_name = config_f.split("/")[-1].split(".yaml")[0]
    if len(config_name.split(".")) == 3:
        assert config_name.split(".")[-1].startswith("TEST-")
        config_name = ".".join(config_name.split(".")[:-1])
    assert len(config_name.split(".")) == 2
    new_config_name = config_name.replace(
        og_pl, new_pl
    ).replace(
        og_cl, new_cl
    ).replace(
        og_tl, new_tl
    )
    new_config_lines = []
    for l, line in enumerate(config_lines):
        ending = None
        new_ending = None
        spm_name = None
        new_spm_name = None
        if line.startswith("src:"):
            if is_pretrain(config_f):
                assert line.endswith(og_pl), f"config {config_f}, line `{line}` does not end with `{og_pl}`"
                line = line.replace(og_pl, new_pl)
            else:
                assert line.endswith(og_cl), f"config {config_f}, line `{line}` does not end with `{og_cl}`"
                line = line.replace(og_cl, new_cl)
        elif line.startswith("tgt:"):
            assert line.endswith(og_tl)
            line = line.replace(og_tl, new_tl)
        elif line.startswith("save:"):
            assert ending is None
            assert new_ending is None
            ending = f"/{og_cl}-{og_tl}/{config_name}"
            assert line.endswith(ending), f"line `{line}` does not end with `{ending}`"
            line = line[:-len(ending)] + f"/{new_cl}-{new_tl}/{new_config_name}"
        elif line.startswith("from_pretrained:") and line != "from_pretrained: null":
            assert is_finetune(config_f), f"Config is not finetune: {config_f}"
            assert ending is None
            if not is_sc_file(config_f):
                ending = f"/{og_cl}-{og_tl}/PRETRAIN.{og_pl}-{og_tl}_TRIAL_s=1000"
                new_ending = f"/{new_cl}-{new_tl}/PRETRAIN.{new_pl}-{new_tl}_TRIAL_s=1000"
            else:
                ending = f"/{og_cl}-{og_tl}/PRETRAIN.SC_{og_pl}2{og_cl}-{og_tl}_TRIAL_s=1000"
                new_ending = f"/{new_cl}-{new_tl}/PRETRAIN.SC_{new_pl}2{new_cl}-{new_tl}_TRIAL_s=1000"
            assert line.endswith(ending), f"line `{line}` does not end with `{ending}`"
            line = line[:-len(ending)] + new_ending
        elif is_data_line(line):
            nmt_dir = f"/{og_cl}-{og_tl}/"
            pre_nmt_dir = f"/{og_pl}-{og_tl}/"
            pre_sc_nmt_dir = f"/SC_{og_pl}2{og_cl}-{og_tl}/"
            if nmt_dir in line:
                line = line.replace(nmt_dir, f"/{new_cl}-{new_tl}/")
            elif pre_nmt_dir in line:
                line = line.replace(pre_nmt_dir, f"/{new_pl}-{new_tl}/")
            elif pre_sc_nmt_dir in line:
                line = line.replace(pre_sc_nmt_dir, f"/SC_{new_pl}2{new_cl}-{new_tl}/")
            else:
                assert False, f"Bad data line: {line}"
        elif line.startswith("predictions_dir: "):
            assert line == f"predictions_dir: predictions_{og_cl}-{og_tl}_test"
            line = f"predictions_dir: predictions_{new_cl}-{new_tl}_test"
        elif line.startswith("sc_model_id: "):
            if is_sc_file(config_f):
                assert og_pl.upper() in line, f"line `{line}` does not contain `{og_pl.upper()}`"
                assert og_cl.upper() in line, f"line `{line}` does not contain `{og_cl.upper()}`"
                assert og_tl.upper() not in line, f"line `{line}` line contains `{og_tl.upper()}`!"
                line = line.replace(
                    og_pl.upper(), new_pl.upper()
                ).replace(
                    og_cl.upper(), new_cl.upper()
                )
            else:
                assert line == "sc_model_id: null"
        elif line.startswith("spm: "):
            assert spm_name is None
            assert new_spm_name is None
            if is_sc_file(config_f):
                spm_name = f"SC_{og_pl}2{og_cl}-{og_cl}_{og_tl}"
                new_spm_name = f"SC_{new_pl}2{new_cl}-{new_cl}_{new_tl}"
            else:
                spm_name = f"{og_pl}-{og_cl}_{og_tl}"
                new_spm_name = f"{new_pl}-{new_cl}_{new_tl}"
            assert spm_name in line
            line = line.replace(spm_name, new_spm_name)
        new_config_lines.append(line)
    
    with open(new_config_f, "w") as outf:
        outf.write("\n".join(new_config_lines) + "\n")

def is_data_line(line):
    return line.startswith("train_data: ") or line.startswith("val_data: ") or line.startswith("test_data: ")
            
def read_f(f):
    with open(f) as inf:
        lines = [l.rstrip() for l in inf.readlines()]
    return lines

def get_new_file_name(f, og_langs, new_langs):
    og_pl, og_cl, og_tl = og_langs
    new_pl, new_cl, new_tl = new_langs

    og_scen, new_scen = None, None
    if is_sc_file(f):
        if is_finetune(f):
            og_scen = f"{og_pl}2{og_cl}-{og_tl}>>{og_cl}-{og_tl}"
            new_scen = f"{new_pl}2{new_cl}-{new_tl}>>{new_cl}-{new_tl}"
        else:
            assert is_pretrain(f)
            og_scen = f"{og_pl}2{og_cl}-{og_tl}"
            new_scen = f"{new_pl}2{new_cl}-{new_tl}"
    else:
        if is_finetune(f):
            og_scen = f"{og_pl}-{og_tl}>>{og_cl}-{og_tl}"
            new_scen = f"{new_pl}-{new_tl}>>{new_cl}-{new_tl}"
        elif is_pretrain(f):
            og_scen = f"{og_pl}-{og_tl}"
            new_scen = f"{new_pl}-{new_tl}"
        else:
            assert is_nmt(f)
            og_scen = f"{og_cl}-{og_tl}"
            new_scen = f"{new_cl}-{new_tl}"
    
    assert og_scen is not None
    assert new_scen is not None

    assert og_scen in f
    return f.replace(og_scen, new_scen)


def is_sc_file(f):
    split_f = f.split(".")
    if ".TEST-" in f:
        assert len(split_f) == 4, f"split_f {split_f} has len {len(split_f)} when it should be 4!"
    else:
        assert len(split_f) == 3, f"split_f {split_f} has len {len(split_f)} when it should be 3!"
    return split_f[1].startswith("SC_")

def is_char(f):
    f = f.split("/")[-1]
    return f.startswith("CHAR-")

def is_augment(f):
    f = f.split("/")[-1]
    return f.startswith("AUGMENT.")

def is_no_grad_clip(f):
    f = f.split("/")[-1]
    return f.endswith(".no_grad_clip.yaml")

def is_finetune(f):
    f = f.split("/")[-1]
    return f.startswith("FINETUNE.")

def is_pretrain(f):
    f = f.split("/")[-1]
    return f.startswith("PRETRAIN.")

def is_nmt(f):
    f = f.split("/")[-1]
    return f.startswith("NMT.")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_dir")
    parser.add_argument("-n", "--new_config_dir")
    parser.add_argument("-ol", "--og_langs", help="comma-delimted list pl,cl,tl")
    parser.add_argument("-nl", "--new_langs", help="comma-delimited list pl,cl,tl")
    return parser.parse_args()

def to_list(comma_str):
    return [i.strip() for i in comma_str.split(",")]

if __name__ == "__main__":
    args = get_args()
    og_langs = to_list(args.og_langs)
    new_langs = to_list(args.new_langs)
    main(
        config_dir=args.config_dir,
        new_config_dir=args.new_config_dir,
        og_langs=og_langs,
        new_langs=new_langs
    )
