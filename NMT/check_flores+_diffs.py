"""
Finds the differences between FLORES+ and flores_200 and as well as version of FLORES+ inside of PILAR
"""
import os

floresplus_dir = "/home/hatch5o6/nobackup/archive/data/FLORES+"
flores200_dir = "/home/hatch5o6/nobackup/archive/data/flores200_dataset"
pilar_floresplus_dir = "/home/hatch5o6/nobackup/archive/data/LRRomance/PILAR/FLORES+"
out_template = "FLORES+_diffs.{d}.txt"

def main():
    floresplus_dev = read_floresplus(floresplus_dir, "dev")
    floresplus_devtest = read_floresplus(floresplus_dir, "devtest")

    flores200_dev = read_flores200(flores200_dir, "dev")
    flores200_devtest = read_flores200(flores200_dir, "devtest")

    pilar_floresplus_dev = read_pilar_floresplus(pilar_floresplus_dir, "dev")
    pilar_floresplus_devtest = read_pilar_floresplus(pilar_floresplus_dir, "devtest")

    check(
        floresplus_data=floresplus_dev,
        flores200_data=flores200_dev,
        pilar_floresplus_data=pilar_floresplus_dev,
        out=out_template,
        div="dev"
    )
    check(
        floresplus_data=floresplus_devtest,
        flores200_data=flores200_devtest,
        pilar_floresplus_data=pilar_floresplus_devtest,
        out=out_template,
        div="devtest"
    )

def check(floresplus_data, flores200_data, pilar_floresplus_data, out, div="dev"):
    assert div in ["dev", "devtest"]

    flores200_leftover = set([(l, s) for l, s in flores200_data.keys()])
    pilar_floresplus_leftover = set([(l, s) for l, s in pilar_floresplus_data.keys()])

    flores200_dne = [] # langs that do not exist in flores200
    pilar_floresplus_dne = [] # langs that do not exist in pilar_floresplus

    TO_REMOVE = set()
    DIFFS = []
    for (lang, script, glotto), floresplus_f in floresplus_data.items():
        flores200_f = flores200_data.get((lang, script), "DNE")
        pilar_floresplus_f = pilar_floresplus_data.get((lang, script), "DNE")

        if flores200_f != "DNE":
            flores200_diffs = compare_files(floresplus_f, flores200_f)
        else:
            flores200_diffs = [(0, "DNE", "DNE")]
            flores200_dne.append((lang, script))
        
        if pilar_floresplus_f != "DNE":
            pilar_floresplus_diffs = compare_files(floresplus_f, pilar_floresplus_f)
        else:
            pilar_floresplus_diffs = [(0, "DNE", "DNE")]
            pilar_floresplus_dne.append((lang, script))
        
        # if len(flores200_diffs) > 0:
        DIFFS.append((floresplus_f, flores200_f, flores200_diffs))
        # if len(pilar_floresplus_diffs) > 0:
        DIFFS.append((floresplus_f, pilar_floresplus_f, pilar_floresplus_diffs))

        TO_REMOVE.add((lang, script))

    for remove_l, remove_s in TO_REMOVE:
        if (remove_l, remove_s) in flores200_leftover:
            flores200_leftover.remove((remove_l, remove_s))
        if (remove_l, remove_s) in pilar_floresplus_leftover:
            pilar_floresplus_leftover.remove((remove_l, remove_s))
    
    write_result(
        DIFFS, 
        flores200_leftover,
        pilar_floresplus_leftover,
        flores200_dne,
        pilar_floresplus_dne,
        out_f=out.replace("{d}", div)
    )

    
def write_result(
    DIFFS, 
    flores200_leftover, 
    pilar_floresplus_leftover, 
    flores200_dne,
    pilar_floresplus_dne, 
    out_f
):
    with open(out_f, "w") as outf:
        outf.write("DIFFS:-\n")
        for i, (f1, f2, diffs) in enumerate(DIFFS):
            outf.write(f"########################## FILE COMPARISON ({i}) ##########################\n")
            outf.write(f"F1: {f1}\n")
            outf.write(f"F2: {f2}\n")
            for line_n, f1line, f2line in diffs:
                outf.write(f"---------------------- ({line_n}) ----------------------\n")
                outf.write(f"F1: `{f1line}`\n")
                outf.write(f"F2: `{f2line}`\n")
            outf.write("\n")
            
        outf.write("\n\n\n")
        outf.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        outf.write("FLORES200 LEFTOVER LANGS:-\n")
        for l, s in flores200_leftover:
            outf.write(f"{l}_{s}\n")
        outf.write("\n")
        outf.write("FLORES200 DNE:-\n")
        for l, s in flores200_dne:
            outf.write(f"{l}_{s}\n")

        outf.write("\n\n\n")
        outf.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        outf.write("PILAR/FLORES+ LEFTOVER LANGS:-\n")
        for l, s in pilar_floresplus_leftover:
            outf.write(f"{l}_{s}\n")
        outf.write("\n")
        outf.write("PILAR/FLORES+ DNE:-\n")
        for l, s in pilar_floresplus_dne:
            outf.write(f"{l}_{s}\n")


def compare_files(f1, f2):
    diffs = []
    f1_lines = read_f(f1)
    f2_lines = read_f(f2)
    pairs = list(zip(f1_lines, f2_lines))
    for i, (f1line, f2line) in enumerate(pairs):
        if f1line != f2line:
            diffs.append((i, f1line, f2line))
    return diffs

def read_floresplus(path, ext_type="dev"):
    assert ext_type in ["dev", "devtest"]

    path = os.path.join(path, ext_type)

    floresplus_data = {}
    for f in os.listdir(path):
        f_path = os.path.join(path, f)
        # print("FPATH:", f_path)

        f_split = f.split(".")
        assert len(f_split) == 2
        langscriptglotto, ext = tuple(f_split)
        assert ext == ext_type
        langscriptglotto_split = langscriptglotto.split("_")
        assert len(langscriptglotto_split) == 3
        lang, script, glotto = tuple(langscriptglotto_split)
        assert (lang, script, glotto) not in floresplus_data
        floresplus_data[(lang, script, glotto)] = f_path
    return floresplus_data


def read_pilar_floresplus(path, ext_type="dev"):
    assert ext_type in ["dev", "devtest"]

    path = os.path.join(path, ext_type)

    pilar_floresplus_data = {}
    for f in os.listdir(path):
        f_path = os.path.join(path, f)

        f_split = f.split(".")
        assert len(f_split) == 2
        ext, langscript = tuple(f_split)
        assert ext == ext_type

        langscript_split = langscript.split("_")
        assert len(langscript_split) == 2
        lang, script = tuple(langscript_split)
        assert (lang, script) not in pilar_floresplus_data
        pilar_floresplus_data[(lang, script)] = f_path
    return pilar_floresplus_data

def read_flores200(path, ext_type="dev"):
    assert ext_type in ["dev", "devtest"]

    path = os.path.join(path, ext_type)

    flores200_data = {}
    for f in os.listdir(path):
        f_path = os.path.join(path, f)

        f_split = f.split(".")
        assert len(f_split) >= 2
        if len(f_split) > 2:
            continue

        langscript, ext = tuple(f_split)
        assert ext == ext_type
        split_langscript = langscript.split("_")
        assert len(split_langscript) == 2
        lang, script = tuple(split_langscript)
        assert (lang, script) not in flores200_data

        flores200_data[(lang, script)] = f_path
    return flores200_data

def read_f(f):
    with open(f) as inf:
        data = [l.strip() for l in inf.readlines()]
    return data

if __name__ == "__main__":
    main()
