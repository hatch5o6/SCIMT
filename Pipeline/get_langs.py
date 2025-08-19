import os
path = "/home/hatch5o6/Cognate/code/NMT/data/PLAIN"

langs = set()
for f in os.listdir(path):
    f_path = os.path.join(path, f)
    assert os.path.isdir(f_path)

    lang_pair = f.split("_")[0]
    lang_pair = lang_pair.split("-")
    assert len(lang_pair) == 2

    src, tgt = tuple(lang_pair)
    langs.add(src)
    langs.add(tgt)

langs = sorted(list(langs))

for lang in langs:
    print(lang)
print(len(langs), "LANGS")