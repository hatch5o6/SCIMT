from align_tokens import fr_tokenize
from spm_tokenizers import SPMTokenizer

fr_line = "Faire partie de la fonction publique donne des privilèges qui ne sont pas accessibles à tous au Canada."
sc_line = "Fan parti de la fonksion piblik donn de privilez qui nenn son pa aksesib sa tou al Kanada."
tokenizer = SPMTokenizer(spm_name="/home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr-mfe_en.NWS/fr-mfe_en/fr-mfe_en")

fr_tok_seq, sc_tok_seq, fr_toks = fr_tokenize(
    tokenizer,
    fr_line,
    sc_line
)

print("FR TOK SEQ")
print(fr_tok_seq)
print("\nSC TOK SEQ")
print(sc_tok_seq)
print("\nFR TOKS")
print(fr_toks)
