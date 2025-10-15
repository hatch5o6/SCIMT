import os
import shutil

for f in os.listdir("."):
    if os.path.isdir(f) or f == "move_files.py": continue
    shutil.copy(f, "../CopperMT/CopperMT/pipeline")

for f in os.listdir("neural_translation"):
    shutil.copy(os.path.join("neural_translation", f), "../CopperMT/CopperMT/pipeline/neural_translation")

for f in os.listdir("statistical_translation"):
    shutil.copy(os.path.join("statistical_translation", f), "../CopperMT/CopperMT/pipeline/statistical_translation")