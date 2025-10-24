import argparse
from string import punctuation
from tqdm import tqdm
from collections import Counter

def make_word_list(sent_pairs, alignments, VERBOSE=False, START=None, STOP=None):
    word_list = Counter()

    data = list(zip(sent_pairs, alignments))
    for idx, ((src_sent, tgt_sent), word_alignments) in tqdm(enumerate(data), total=len(data)):
        if START and idx < START:
            continue
        if idx == STOP:
            break
        if VERBOSE:
            print(f"\n\n--------------------- ({idx}) ------------------------")
            print(src_sent)
            print(tgt_sent)
            print(word_alignments)
        # should already be tokenized and joined on whitespace, so no need for word_tokenize function
        #TODO if ever use this again, may need to account for non-breaking spaces. see make_word_alignments_no_grouping
        src_words = src_sent.split()
        tgt_words = tgt_sent.split()
        max_len = max(len(src_words), len(tgt_words))
        word_alignments = word_alignments.split()

        src_word_align_dict = {(i,):[] for i in range(len(src_words))}
        tgt_word_align_dict = {(i,):[] for i in range(len(tgt_words))}
        # keys = set([i for i in range(max_len)])
        for word_alignment in word_alignments:
            w1, w2 = tuple(word_alignment.split("-"))
            w1, w2 = int(w1), int(w2)

            src_word_align_dict[(w1,)].append(w2)
            tgt_word_align_dict[(w2,)].append(w1)

            # word_list.append((src_words[w1], tgt_words[w2]))
        
        # print("BEFORE")
        # print("SRC GRP -> TGT GRPS:", src_word_align_dict)
        # print("TGT GRP -> SRC GRPS:", tgt_word_align_dict)

        for k, v in src_word_align_dict.items():
            src_word_align_dict[k] = make_groups(v)
        for k, v in tgt_word_align_dict.items():
            tgt_word_align_dict[k] = make_groups(v)

        # print("AFTER")
        # print("SRC GRP -> TGT GRPS:", src_word_align_dict)
        # print("TGT GRP -> SRC GRPS:", tgt_word_align_dict)

        for src_group, tgt_groups in src_word_align_dict.items():
            for tgt_group in tgt_groups:
                tgt_word_align_dict = merge_group(tgt_group, tgt_word_align_dict)
                # print("SRC GRP:", src_group)
                # print("\t-TGT GRPS:", tgt_groups)
                # print("tgt_word_align_dict[tgt_group]:", tgt_word_align_dict[tgt_group])
                # TODO - Not sure we can actually make this assert. Should we assert the tgt group has a pointer back to the src group? This should happen, but is it necessary to assert it?
                # assert tgt_word_align_dict[tgt_group] == [src_group]
        
        for tgt_group, src_groups, in tgt_word_align_dict.items():
            for src_group in src_groups:
                src_word_align_dict = merge_group(src_group, src_word_align_dict)
                # assert src_word_align_dict[src_group] == [tgt_group]

        if VERBOSE:
            print("\n\n- final alignments -")
            print("\tsrc_word_align_dict", src_word_align_dict)
            print("\ttgt_word_align_dict", tgt_word_align_dict)
            print("")
        
        keys = set(
            list(src_word_align_dict.keys()) + list(tgt_word_align_dict.keys())
        )
        for w in sorted(list(keys)):
            src_word_group, tgt_word_group = "", ""
            if w in src_word_align_dict:
                for idx in w:
                    assert idx < len(src_words)
                src_word_group = " ".join([src_words[idx] for idx in w])
            
            if w in tgt_word_align_dict:
                for idx in w:
                    assert idx < len(tgt_words)
                tgt_word_group = " ".join(tgt_words[idx] for idx in w)

            src_word_tgts = [i for i in src_word_align_dict.get(w, [])]
            
            if VERBOSE:
                print("\t-", w, f"'{src_word_group}'", f"'{tgt_word_group}'")
                print("\t\tsrc_word_tgts groups", src_word_tgts)
            for group in src_word_tgts:
                group_words = " ".join([tgt_words[i] for i in group])
                if VERBOSE:
                    print("\t\t\tadding", f"src: '{src_word_group}'", f"tgt group: '{group_words}'")
                word_list[(src_word_group, group_words)] += 1
            
            tgt_word_srcs = [i for i in tgt_word_align_dict.get(w, [])]
            if VERBOSE:    
                print("\t\ttgt_word_srcs groups", tgt_word_srcs)
            for group in tgt_word_srcs:
                group_words = " ".join([src_words[i] for i in group])
                if VERBOSE:
                    print("\t\t\tadding", f"src group: '{group_words}'", f"tgt: '{tgt_word_group}'")
                word_list[(group_words, tgt_word_group)] += 1

    return word_list

def merge_group(group, align_dict):
    # print("merging group", group)
    # print("BEFORE", align_dict)

    if len(group) < 2:
        return align_dict

    if group not in align_dict:
        align_dict[group] = []
    
    idx_values = {}
    for idx in group:
        # print("idx", idx)
        if (idx,) in align_dict:
            v = align_dict[(idx,)]
            idx_values[idx] = v
            # print("\tv:", v)
            # I THINK WE SHOULD ONLY MERGE ON WHAT THEY HAVE IN COMMON
            # Will need to add to align_dict outside the loop. In the loop, we need to just gather all the values.
            # combined_v = sorted(list(set(align_dict[group] + v)))
            # print("\tcombined_v:", v)
            # align_dict[group] = combined_v
            # to_pop.append((idx,))
    
    intersected_v = None
    for idx, v in idx_values.items():
        if intersected_v is None:
            intersected_v = set(v)
        else:
            intersected_v = intersected_v.intersection(set(v))
    
    to_pop = []
    align_dict[group] = sorted(list(intersected_v))
    for idx in group:
        if (idx,) in align_dict:
            align_dict[(idx,)] = sorted(list(
                set(align_dict[(idx,)]).difference(intersected_v)
            ))
        if len(align_dict[(idx,)]) == 0:
            to_pop.append((idx,))

    for k in to_pop:
        align_dict.pop(k)
    return align_dict

def make_groups(a_list):
    # takes a list of ints, and groups the continuous ints together
    # e.g. [1,2,4,5] -> [[1,2], [4,5]]

    a_list.sort()
    groups = []
    group = []
    for n in a_list:
        if len(group) == 0:
            group.append(n)
        else:
            if n == group[-1] + 1:
                group.append(n)
            else:
                groups.append(tuple(group))
                group = [n]
    if len(group) > 0:
        groups.append(tuple(group))
    return groups


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alignments", help="word alignments file from fast_align")
    parser.add_argument("-s", "--sent_pairs", help="sentence pairs, delimited with ' ||| '")
    parser.add_argument("-o", "--out")
    parser.add_argument("--VERBOSE", action="store_true")
    parser.add_argument("--START", type=int, default=None)
    parser.add_argument("--STOP", type=int, default=None)
    args = parser.parse_args()
    print("arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: '{v}'")
    print("---------------------\n\n")
    return args

if __name__ == "__main__":
    print("make_word_alignments.py")
    args = get_args()
    with open(args.alignments) as inf:
        alignments = [line.strip() for line in inf]
    sent_pairs = []
    with open(args.sent_pairs) as inf:
        for line in inf:
            line = line.strip()
            src, tgt = tuple(line.split(" ||| "))
            sent_pairs.append((src.strip(), tgt.strip()))

    word_list = make_word_list(
        sent_pairs, 
        alignments,
        VERBOSE=args.VERBOSE,
        START=args.START,
        STOP=args.STOP
    )

    with open(args.out, "w") as outf:
        for word_a, word_b in word_list.keys():
            outf.write(f"{word_a.strip()} ||| {word_b.strip()}\n")