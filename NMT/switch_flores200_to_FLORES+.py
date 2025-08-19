import os
import csv

PLAIN_dir = "/home/hatch5o6/Cognate/code/NMT/data/PLAIN"
FLORES_plus_dir = "/home/hatch5o6/nobackup/archive/data/FLORES+"

FLORES_plus_files = {"dev": {}, "devtest": {}}
OCCUR_MORE_THAN_ONCE = {"dev": set(), "devtest": set()}
for dataset in FLORES_plus_files.keys():
    FLORES_dataset = os.path.join(FLORES_plus_dir, dataset)
    assert os.path.isdir(FLORES_dataset)
    for f in os.listdir(FLORES_dataset):
        f_path = os.path.join(FLORES_dataset, f)
        # print("fff-", f_path)
        assert os.path.isfile(f_path)
        assert f_path.endswith(f".{dataset}")

        fsplit = f.split("_")
        assert len(fsplit) == 3
        lang, script, _ = tuple(fsplit)
        # print("lang", lang)
        # print("script", script)
        # exit()
        if (lang, script) in FLORES_plus_files[dataset].keys():
            OCCUR_MORE_THAN_ONCE[dataset].add((lang, script))
        else:
            FLORES_plus_files[dataset][(lang, script)] = f_path
print("FLORES+ contains these langs more than once:")
for k in OCCUR_MORE_THAN_ONCE:
    print(f'\t{k}')
    for l in OCCUR_MORE_THAN_ONCE[k]:
        print(f"\t\t-{l}")
print("\n\n")


def main():
    for d in os.listdir(PLAIN_dir):
        d_path = os.path.join(PLAIN_dir, d)
        assert os.path.isdir(d_path)
        
        if d.endswith("_dev_test"):
            for f in os.listdir(d_path):
                if f in ["OLD_test.csv", "OLD_val.csv"]:
                    print("------------------------------------------------------")
                    div = f.split("_")[1].split(".csv")[0]
                    assert div in ["test", "val"]
                    if div == "val":
                        tag = "dev"
                    elif div == "test":
                        tag = "devtest"

                    f_path = os.path.join(d_path, f)
                    header, src, tgt, src_path, tgt_path = read_csv(f_path)
                    print("  f:", f_path)
                    print("src:", src)
                    print('tgt:', tgt)
                    print('div:', div)

                    if is_flores200(src_path, tgt_path):
                        plus_src_path = make_floresplus(src_path, tag)
                        plus_tgt_path = make_floresplus(tgt_path, tag)
                        write_csv(
                            header=header,
                            src=src,
                            tgt=tgt,
                            src_path=plus_src_path,
                            tgt_path=plus_tgt_path,
                            out_f=os.path.join(d_path, f"{div}.csv")
                        )


def read_csv(f):
    with open(f, newline='') as inf:
        rows = [r for r in csv.reader(inf)]
    header = rows[0]
    data = rows[1:]
    assert isinstance(data, list)
    assert len(data) == 1
    
    src, tgt, src_path, tgt_path = tuple(data[0])
    return header, src, tgt, src_path, tgt_path


def is_flores200(src_path, tgt_path):

    if "/LRRomance/PILAR/FLORES+/" in src_path:
        assert "/flores200_dataset/" in tgt_path
    if "/LRRomance/PILAR/FLORES+/" in tgt_path:
        assert "/flores200_dataset/" in src_path

    if "/flores200_dataset/" in src_path:
        assert "/flores200_dataset/" in tgt_path or "/LRRomance/PILAR/FLORES+/" in tgt_path
    if "/flores200_dataset/" in tgt_path:
        assert "/flores200_dataset/" in src_path or "/LRRomance/PILAR/FLORES+/" in src_path
    
    if "/flores200_dataset/" in src_path or "/flores200_dataset/" in tgt_path:
        return True
    
def make_floresplus(path, tag):
    print("path", path)
    print("tag", tag)

    if "/flores200_dataset/" in path:
        assert "/LRRomance/PILAR/FLORES+/" not in path
        assert path.split("/")[-2] == tag
        assert path.split(".")[-1] == tag

        f_name = path.split("/")[-1]
        f_split = f_name.split(".")
        assert len(f_split) == 2
        langscript, ext = tuple(f_split)
    elif "/LRRomance/PILAR/FLORES+/" in path:
        assert "/flores200_dataset/" not in path
        assert path.split("/")[-2] == tag
        assert path.split("/")[-1].startswith(f"{tag}.")
        assert path.split("/")[-1].split(".")[0] == tag

        f_name = path.split("/")[-1]
        f_split = f_name.split(".")
        assert len(f_split) == 2
        ext, langscript = tuple(f_split)
    else:
        assert False

    assert ext == tag

    langscript_split = langscript.split("_")
    assert len(langscript_split) == 2
    lang, script = tuple(langscript_split)

    floresplus_f = FLORES_plus_files[tag][(lang, script)]
    return floresplus_f


def write_csv(header, src, tgt, src_path, tgt_path, out_f):
    content = ""
    content += ",".join(header) + "\n"
    content += ",".join([src, tgt, src_path, tgt_path])

    if os.path.exists(out_f):
        print(f"\nOUT FILE already exists: {out_f}")
        with open(out_f) as inf:
            existing_content = inf.read().strip()
            assert existing_content == content
            print("\tBut... CONTENT == EXISTING CONTENT and we don't need to write the file :)\n")
    else:
        print(f"\nWriting OUT FILE: {out_f}\n")
        with open(out_f, "w") as outf:
            outf.write(content + '\n')


if __name__ == "__main__":
    main()

