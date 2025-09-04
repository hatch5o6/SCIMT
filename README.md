## Clone CopperMT and add new/updated scripts
From root directory, run these:
```
cd CopperMT
git clone https://github.com/clefourrier/CopperMT.git
cd ../CopperMTfiles
python move_files.py
```

## Pipeline/train_SC.sh
**Pipeline/train_SC.sh** trains the character correspondence (SC) models.
We call it SC, which stands for "sound correspondence", but more accurately, what we're determining are character correspondences.

**Pipeline/train_SC.sh** is run from /Cognate/code, and takes a single positional argument, a *.cfg* config file, e.g.:
```
bash Pipeline/train_SC.sh /home/hatch5o6/Cognate/code/Pipeline/cfg/SC/fr-mfe.cfg
```

See *Pipeline/cfg/SC* for the .cfg files for all 10 scenarios of these experiments. They contain the following parameters:
- **MODULE_HOME_DIR** asdf