#!/usr/bin/env python
"""
Test that core dependencies are installed correctly.
Run from project root: python test_imports.py
"""

def test_imports():
    print("Testing core imports...")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        import lightning
        print(f"✓ Lightning {lightning.__version__}")
    except ImportError as e:
        print(f"✗ Lightning import failed: {e}")
        return False

    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False

    try:
        import sentencepiece
        print(f"✓ SentencePiece")
    except ImportError as e:
        print(f"✗ SentencePiece import failed: {e}")
        return False

    try:
        from NMT.parallel_datasets import MultilingualDataset
        from NMT.evaluate import calc_bleu
        print(f"✓ NMT modules")
    except ImportError as e:
        print(f"✗ NMT modules import failed: {e}")
        return False

    print("\n✓ All core dependencies are installed correctly!")
    return True

if __name__ == "__main__":
    import sys
    import os

    # Add NMT to path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NMT'))

    success = test_imports()
    sys.exit(0 if success else 1)
