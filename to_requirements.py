

filepath = "/home/hatch5o6/Cognate/code/sound.reqs.txt"

with open(filepath) as inf:
    lines = [l.strip() for l in inf.readlines()]
libs = {}
for line in lines:
    if line.startswith("#"):
        continue
    split = line.split()
    assert len(split) in [3, 4]
    lib = split[0]
    v = split[1]
    assert lib not in libs.keys()
    libs[lib] = v

with open("sound.requirements.txt", "w") as outf:
    for lib, v in libs.items():
        outf.write(f"{lib}=={v}\n")
