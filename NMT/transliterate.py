import epitran


epi = epitran.Epitran("ara-Arab")

sentence = "مرحبا يا أصدقائي"
result = epi.transliterate(sentence)
print(result)