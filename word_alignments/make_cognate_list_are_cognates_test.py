from make_cognate_list import are_cognates as new_are_cognates
from make_cognate_list_OLD import are_cognates as old_are_cognates

test_file = "/home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/es-an_ES-AN-RNN-0_RNN-285_S-0/fastalign/word_list.es-an.NG.txt"

def main():
    pairs = read_word_pairs(test_file)
    for i, (word1, word2) in enumerate(pairs):
        new_passed, new_dist = new_are_cognates(word1, word2, edit_dist_type="NLED")
        old_passed, old_dist = old_are_cognates(word1, word2)

        print(f"{i}) `{word1}`, `{word2}`, new_dist: {new_dist}, old_dist: {old_dist}")
        assert new_passed == old_passed, f"\t{i}) `{word1}`, `{word2}` -- NEW_PASSED: {new_passed}, OLD_PASSED: {old_passed}"
        assert new_dist == old_dist, f"\t{i}) `{word1}`, `{word2}` -- NEW_DIST: {new_dist}, OLD_DIST: {old_dist}"
    print("\nALL PASSED :)")

def read_word_pairs(f):
    pairs = []
    with open(f) as inf:
        for line in inf.readlines():
            line = line.strip()
            n, word1, word2 = tuple(line.split(" ||| "))
            assert " " not in n
            assert " " not in word1
            assert " " not in word2
            pairs.append((word1, word2))
    return pairs
            

if __name__ == "__main__":
    main()