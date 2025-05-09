from make_word_alignments import merge_group
import json

src_word_align_dict = {(0,): [(0, 1)], (1,): [(2,)], (2,): [(2,)], (3,): [(3,), (8,)], (4,): [(6, 7), (9, 10)], (5,): [(5,), (11,)]}
tgt_word_align_dict = {(0,): [(0,)], (1,): [(0,)], (2,): [(1, 2)], (3,): [(3,)], (4,): [], (5,): [(5,)], (6,): [(4,), (3,)], (7,): [(4,), (3,)], (8,): [(4,), (3,)], (9,): [(4,), (3,)], (10,): [(4,), (3,)], (11,): [(4,5), (7,8), (3,)], (12,): [(6,)]}

print("BEFORE", tgt_word_align_dict)

tgt_word_align_dict = merge_group((0, 1), tgt_word_align_dict)
print("\nAFTER:",tgt_word_align_dict)

print("\n\n\n doing it again")
tgt_word_align_dict = merge_group((6, 7, 8, 9, 10, 11), tgt_word_align_dict)
print("\nAFTER:",tgt_word_align_dict)

