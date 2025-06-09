from tokenizers import Tokenizer

tokenizer_path = "/home/hatch5o6/nobackup/archive/CognateMT/hf_tokenizers/SC_fr2mfe.mfe/SC_fr2mfe.mfe.json"
tokenizer = Tokenizer.from_file(tokenizer_path)

sents = [
            "Des soz qui pevan rand me paran fier: nou travayon avek les Eagl.",
            "ze vwa que davantaz de politisian, an Norvez egalman, on komanse so prand sela au serye."
        ]

pad_token = "<pad>"
pad_id = tokenizer.token_to_id(pad_token)
print("pad id", pad_id)
tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
output = tokenizer.encode_batch(sents)

for seq in output:
    print("--------")
    print(seq.tokens)
    print(seq.ids)
    print("DECODED:", tokenizer.decode(seq.ids, skip_special_tokens=True))
