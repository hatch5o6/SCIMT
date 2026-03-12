import argparse
import csv

def main(
    test_f,
    hyp1_f,
    hyp2_f,
    mt_eval_f
):
    mt_eval = read_mt_eval_file(mt_eval_f)
    hyp1 = read_f(hyp1_f)
    hyp1_name = hyp_name(hyp1_f)

    hyp2 = read_f(hyp2_f)
    hyp2_name = hyp_name(hyp2_f)

    test = read_f(test_f)

    print(f"\n\nmt eval ref vs `{test_f}`")
    assert sample_in_og(mt_eval["ref"], test)

    print(f"\n\nmt eval {hyp1_name} vs `{hyp1_f}`")
    assert sample_in_og(mt_eval[hyp1_name], hyp1)

    print(f"\n\nmt eval {hyp2_name} vs `{hyp2_f}`")
    assert sample_in_og(mt_eval[hyp2_name], hyp2)

    print("all tests passed :)")

def sample_in_og(subsample, og):
    print("subsample:", len(subsample))
    print("og:", len(og))

    for s, seq in enumerate(subsample):
        if seq not in og:
            print(f"seq not in og: ({s}) `{seq}`")
            return False
    return True
    

def hyp_name(hyp_f):
    return hyp_f.split("/")[-4]

def read_f(f):
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    return lines

def read_mt_eval_file(f):
    with open(f, newline='') as inf:
        rows = [r for r in csv.reader(inf, delimiter="\t")]
    header = rows[0]
    ref_head, m1_head, m2_head = header
    data = rows[1:]
    for r in data:
        assert len(r) == 3

    ref_items = [r[0] for r in data]
    m1_items = [r[1] for r in data]
    m2_items = [r[2] for r in data]

    data_dict = {
        ref_head: ref_items,
        m1_head: m1_items,
        m2_head: m2_items
    }

    return data_dict
    


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test")
    parser.add_argument("--hyp1")
    parser.add_argument("--hyp2")
    parser.add_argument("--mt_eval")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(
        test_f=args.test,
        hyp1_f=args.hyp1,
        hyp2_f=args.hyp2,
        mt_eval_f=args.mt_eval
    )