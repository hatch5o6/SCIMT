path = "/home/hatch5o6/nobackup/archive/ThesisSRE/data"
import os
import shutil
from tqdm import tqdm
for dir in tqdm(os.listdir(path)):
    if dir.startswith("CCMatrix_train_"):
        dir_path = os.path.join(path, dir)
        shutil.rmtree(dir_path)