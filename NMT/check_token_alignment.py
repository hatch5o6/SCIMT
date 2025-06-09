import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

from spm_tokenizers import SPMTokenizer

fr_file = "/home/hatch5o6/nobackup/archive/data/CCMatrix_fr_en/fixed/stitched/cleaned/tgt.100k.txt"
SC_fr2mfe_file = "/home/hatch5o6/nobackup/archive/data/CCMatrix_fr_en/fixed/stitched/cleaned/tgt.100k.SC_NGfr2NGmfe.txt"
TOTAL = 3
tok_lang = "french"

fr_spm_name = "/home/hatch5o6/nobackup/archive/CognateMT/spm_models_ws/fr.mfe.WS/fr.mfe.WS"
fr_tokenizer = SPMTokenizer(spm_name=fr_spm_name)

sc_spm_name = "/home/hatch5o6/nobackup/archive/CognateMT/spm_models_ws/SC_fr2mfe.mfe.WS/SC_fr2mfe.mfe.WS"
sc_tokenizer = SPMTokenizer(spm_name=sc_spm_name)


def read_data(f, TOTAL):
    lines = []
    with open(f) as inf:
        while len(lines) < TOTAL:
            lines.append(inf.readline().strip())
    return lines

fr_data = read_data(fr_file, TOTAL=TOTAL)
SC_fr2mfe_data = read_data(SC_fr2mfe_file, TOTAL=TOTAL)
pairs = list(zip(fr_data, SC_fr2mfe_data))

for fr_seg, SC_fr2mfe_seg in pairs:
    _, fr_seg_toks = fr_tokenizer.tokenize(fr_seg)
    _, SC_fr2mfe_seg_toks = sc_tokenizer.tokenize(SC_fr2mfe_seg)
    fr_words = word_tokenize(fr_seg, language=tok_lang)
    SC_fr2mfe_words = word_tokenize(SC_fr2mfe_seg, language=tok_lang)
    
    print("\n\n############################")
    print("FR SEG:", fr_seg_toks)
    print("SC SEG:", SC_fr2mfe_seg_toks)
    print("-")
    if len(fr_words) != len(SC_fr2mfe_words):
        print("----------")
        print("fr:", len(fr_words), fr_words)
        print("SC_fr2mfe:", len(SC_fr2mfe_words), SC_fr2mfe_words)
        print("NOT EQUAL")
    else:
        word_pairs = list(zip(fr_words, SC_fr2mfe_words))
        print("\nFR --> SC_FR2MFE")
        for fr_word, SC_fr2mfe_word in word_pairs:
            _, fr_word_toks = fr_tokenizer.tokenize(fr_word)
            _, SC_fr2mfe_word_toks = sc_tokenizer.tokenize(SC_fr2mfe_word)
            print(fr_word_toks, "-->", SC_fr2mfe_word_toks)



