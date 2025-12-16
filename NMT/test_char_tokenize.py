from char_tokenizers import CharacterTokenizer

tokenizer = CharacterTokenizer(
    vocab_path="NMT/char_vocab.csv", 
    data_paths=["/home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr-mfe_en/fr-mfe_en/training_data.s=1500.txt"],
    lang_toks=["<mfe>", "<fr>", "<en>"]
)
