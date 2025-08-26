import argparse
from torch.utils.data import DataLoader
import csv
from tqdm import tqdm

from parallel_datasets_sc import SCAlignedMultilingualDataset



def main(
    data_csv,
    sc_data_csv,
    sc_model_id,
    pipeline
):
    for step in pipeline:
        print("TEST", step)
        globals()[step](
            data_csv=data_csv,
            sc_data_csv=sc_data_csv,
            sc_model_id=sc_model_id
        )
    print("PASSED THESE STEPS:")
    for step in pipeline:
        print("\npassed", step)


def test_ordered(
    data_csv,
    sc_data_csv,
    sc_model_id
):
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!! RUNNING test_ordered !!!!!!!!!!!!!!!!!!!!!!!!!")
    data_csv_content, data_csv_content_len = read_csv_f(data_csv, sc_model_id=sc_model_id)
    sc_data_csv_content, sc_data_csv_content_len = read_csv_f(sc_data_csv, sc_model_id=sc_model_id)

    print("\n\n#################################")
    # get_sc_dataloader(data_csv=data_csv, PRINT=100)
    plain_dataloader = get_sc_dataloader(data_csv=data_csv)
    print("getting plain_dataloader data")
    plain_data = get_dataloader_data(plain_dataloader)
    print("\nPLAIN DATALOADER (TOP 100)")
    print_dataloader_data(plain_data, n_lines=100)

    print("\n\n")
    # get_sc_dataloader(
    #     data_csv=data_csv,
    #     sc_data_csv=sc_data_csv,
    #     sc_model_id=sc_model_id,
    #     PRINT=100
    # )
    sc_dataloader = get_sc_dataloader(
        data_csv=data_csv,
        sc_data_csv=sc_data_csv,
        sc_model_id=sc_model_id
    )
    print("getting sc_dataloader data")
    sc_data = get_dataloader_data(sc_dataloader)
    print("\nSC DATALOADER (TOP 100)")
    print_dataloader_data(sc_data, n_lines=100)
    print("\n\n")

    assert len(plain_data) == len(sc_data) == data_csv_content_len == sc_data_csv_content_len

    # tests lengths of both plain_data and sc_data are same, 
    #   that plain_sc_lines are <NONE>, 
    #   and that the other tags and lines (not sc_line) are equal
    test_sc_vs_plain_data(plain_data, sc_data)
    # Should not need to examine plain_data because we asserted 
    #   that its sc_lines are null and that everything else is 
    #   the same as sc_data.


    csv_content = list(zip(data_csv_content, sc_data_csv_content))
    i = 0
    for z, (data_content_item, sc_content_item) in enumerate(csv_content):
        # content_items are from content (i.e.read from csv directly)

        # SC_CONTENT_ITEM_HAS_SC_PATH - whether the src_path in sc_content_item is an SC PATH (contains `SC_{SC_MODEL_ID}`)
        #   if this is False, then data_content_item and sc_content_item are exactly the same.
        #   if this is True, then data_content_item and sc_content_item only differ in src_path and src_lines
        SC_CONTENT_ITEM_HAS_SC_PATH = test_items_are_same(data_content_item, sc_content_item)
        assert len(data_content_item["src_lines"]) == data_content_item["len"]
        for n in range(data_content_item["len"]):

            # items are from content (i.e.read from csv directly)
            data_content_item_src_line = data_content_item["src_lines"][n]
            data_content_item_tgt_line = data_content_item["tgt_lines"][n]
            data_content_item_src_tag = data_content_item["src_lang"]
            data_content_item_tgt_tag = data_content_item["tgt_lang"]

            sc_content_item_src_line = sc_content_item["src_lines"][n]
            sc_content_item_tgt_line = sc_content_item["tgt_lines"][n]
            sc_content_item_src_tag = sc_content_item["src_lang"]
            sc_content_item_tgt_tag = sc_content_item["tgt_lang"]

            (sc_dataloader_src_tag, 
             sc_dataloader_src_line, 
             sc_dataloader_sc_line, 
             sc_dataloader_tgt_tag, 
             sc_dataloader_tgt_line) = sc_data[i]

            # test tags
            assert sc_dataloader_src_tag == data_content_item_src_tag == sc_content_item_src_tag # good
            assert sc_dataloader_tgt_tag == data_content_item_tgt_tag == sc_content_item_tgt_tag # good
            assert sc_dataloader_src_tag != sc_dataloader_tgt_tag # good

            # test tgt_line
            assert sc_dataloader_tgt_line == data_content_item_tgt_line == sc_content_item_tgt_line # good
            

            # test src_line
            if SC_CONTENT_ITEM_HAS_SC_PATH:
                # assert data_content_item_src_line != sc_content_item_src_line # can't necessarily assert this because the sc prediction could be the same as the original source segment
                if not all([
                    sc_dataloader_src_line == data_content_item_src_line, # good
                    sc_dataloader_sc_line != "<NONE>", # good 
                    sc_dataloader_sc_line == sc_content_item_src_line # good
                ]):
                    print("ERROR: SC CSV CONTENT ITEM HAS SC PATH assertions")
                    print(f"---------------- i ({i}), n ({z}-{n}) ----------------")
                    print("data csv src path:", data_content_item["src_path"])
                    print("data csv tgt path:", data_content_item["tgt_path"])
                    print("  sc csv src path:", sc_content_item["src_path"])
                    print("  sc csv tgt path:", sc_content_item["tgt_path"])
                    print("~")
                    print(f"     data_csv src line: `{data_content_item_src_line}`")
                    print(f"sc dataloader src line: `{sc_dataloader_src_line}`\n")
                    
                    print(f"       sc_csv src line: `{sc_content_item_src_line}`")
                    print(f" sc dataloader sc line: `{sc_dataloader_sc_line}`")
                    assert False
                # assert sc_dataloader_src_line != sc_dataloader_sc_line # can't necessarily assert this because the sc prediction could be the same as the original source segment
            else:
                if not all([
                    sc_dataloader_sc_line == "<NONE>", # good
                    sc_dataloader_src_line == data_content_item_src_line == sc_content_item_src_line # good
                ]):
                    print("ERROR: SC CSV CONTENT ITEM DOES NOT HAVE SC PATH assertions")
                    print(f"---------------- i ({i}), n ({z}-{n}) ----------------")
                    print("data csv src path:", data_content_item["src_path"])
                    print("data csv tgt path:", data_content_item["tgt_path"])
                    print("  sc csv src path:", sc_content_item["src_path"])
                    print("  sc csv tgt path:", sc_content_item["tgt_path"])
                    print("~")
                    print(f"     data_csv src line: `{data_content_item_src_line}`")
                    print(f"sc dataloader src line: `{sc_dataloader_src_line}`\n")
                    
                    print(f"       sc_csv src line: `{sc_content_item_src_line}`")
                    print(f" sc dataloader sc line: `{sc_dataloader_sc_line}`")
                    assert False

            i += 1
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@ PASSED test_ordered @@@@@@@@@@@@@@@@@@@@@@@@@")



def test_shuffled(
    data_csv,
    sc_data_csv,
    sc_model_id
):
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!! RUNNING test_shuffled !!!!!!!!!!!!!!!!!!!!!!!!!")
    data_csv_content, data_csv_content_len = read_csv_f(data_csv, sc_model_id=sc_model_id)
    sc_data_csv_content, sc_data_csv_content_len = read_csv_f(sc_data_csv, sc_model_id=sc_model_id)

    print("\n\n#################################")
    # get_sc_dataloader(data_csv=data_csv, PRINT=100)
    plain_dataloader = get_sc_dataloader(data_csv=data_csv, shuffle=True)
    print("getting plain_dataloader data")
    plain_data = get_dataloader_data(plain_dataloader)
    print("\nPLAIN DATALOADER (TOP 100)")
    print_dataloader_data(plain_data, n_lines=100)

    print("\n\n")
    # get_sc_dataloader(
    #     data_csv=data_csv,
    #     sc_data_csv=sc_data_csv,
    #     sc_model_id=sc_model_id,
    #     PRINT=100
    # )
    sc_dataloader = get_sc_dataloader(
        data_csv=data_csv,
        sc_data_csv=sc_data_csv,
        sc_model_id=sc_model_id,
        shuffle=True
    )
    print("getting sc_dataloader data")
    sc_data = get_dataloader_data(sc_dataloader)
    print("\nSC DATALOADER (TOP 100)")
    print_dataloader_data(sc_data, n_lines=100)
    print("\n\n")

    assert len(plain_data) == len(sc_data) == data_csv_content_len == sc_data_csv_content_len

    # tests lengths of both plain_data and sc_data are same, 
    #   that plain_sc_lines are <NONE>, 
    #   and that the other tags and lines (not sc_line) are equal
    test_sc_vs_plain_data(plain_data, sc_data)
    # Should not need to examine plain_data because we asserted 
    #   that its sc_lines are null and that everything else is 
    #   the same as sc_data.

    sc_data_set = get_sc_data_set(sc_data=sc_data)
    sc_data_set_len = len(sc_data_set)
    csv_content = list(zip(data_csv_content, sc_data_csv_content))
    REMOVED_FROM_SC_DATA_SET = set()
    for z, (data_content_item, sc_content_item) in enumerate(csv_content):
        # content_items are from content (i.e.read from csv directly)

        # SC_CONTENT_ITEM_HAS_SC_PATH - whether the src_path in sc_content_item is an SC PATH (contains `SC_{SC_MODEL_ID}`)
        #   if this is False, then data_content_item and sc_content_item are exactly the same.
        #   if this is True, then data_content_item and sc_content_item only differ in src_path and src_lines
        SC_CONTENT_ITEM_HAS_SC_PATH = test_items_are_same(data_content_item, sc_content_item)
        assert len(data_content_item["src_lines"]) == data_content_item["len"]
        for n in range(data_content_item["len"]):

            # items are from content (i.e.read from csv directly)
            data_content_item_src_line = data_content_item["src_lines"][n]
            data_content_item_tgt_line = data_content_item["tgt_lines"][n]
            data_content_item_src_tag = data_content_item["src_lang"]
            data_content_item_tgt_tag = data_content_item["tgt_lang"]

            sc_content_item_src_line = sc_content_item["src_lines"][n]
            sc_content_item_tgt_line = sc_content_item["tgt_lines"][n]
            sc_content_item_src_tag = sc_content_item["src_lang"]
            sc_content_item_tgt_tag = sc_content_item["tgt_lang"]

            # test tags
            assert data_content_item_src_tag == sc_content_item_src_tag # good
            assert data_content_item_tgt_tag == sc_content_item_tgt_tag # good
            assert data_content_item_src_tag != data_content_item_tgt_tag # good

            # test tgt_line
            assert data_content_item_tgt_line == sc_content_item_tgt_line # good

            # test lines
            if SC_CONTENT_ITEM_HAS_SC_PATH:
                sc_line = sc_content_item_src_line # good
                assert sc_line != "<NONE>" # shouldn't actually need this, probably # good
            else:
                assert sc_content_item_src_line == data_content_item_src_line # good
                sc_line = "<NONE>" # good

            item = (
                data_content_item_src_tag, # src_tag
                data_content_item_src_line, # src_line
                sc_line, # sc_line
                data_content_item_tgt_tag, # tgt_tag
                data_content_item_tgt_line # tgt_line
            )
            assert item in sc_data_set
            assert item not in REMOVED_FROM_SC_DATA_SET

            sc_data_set.remove(item)
            REMOVED_FROM_SC_DATA_SET.add(item)

    assert len(REMOVED_FROM_SC_DATA_SET) == sc_data_set_len
    assert len(sc_data_set) == 0
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@ PASSED test_shuffled @@@@@@@@@@@@@@@@@@@@@@@@@")




            

def test_sc_vs_plain_data(plain_data, sc_data):
    # tests lengths of both plain_data and sc_data are same (data from dataloaders), 
    #   that plain_sc_lines are None, 
    #   and that the other tags and lines (not sc_line) are equal
    assert len(plain_data) == len(sc_data)

    for i in range(len(plain_data)):
        plain_src_tag,  plain_src_line,     plain_sc_line,  plain_tgt_tag,  plain_tgt_line = plain_data[i]
        sc_src_tag,     sc_src_line,        sc_sc_line,     sc_tgt_tag,     sc_tgt_line = sc_data[i]

        assert plain_sc_line == "<NONE>"
        # assert sc_sc_line != "<NONE>" # <NONE> can occur in this, actually, for data from non {SC_MODEL_ID} filepaths

        assert plain_src_tag == sc_src_tag
        assert plain_src_line == sc_src_line
        assert plain_tgt_tag == sc_tgt_tag
        assert plain_tgt_line == sc_tgt_line

def print_item(label, item):
    new_item = {k: v for k, v in item.items() if not k.endswith("_lines")}
    print(label, new_item)

def test_items_are_same(data_item, sc_item):
    print("TESTING THAT DATA and SC CONTENT ITEMS (directly from csv) ARE THE SAME")
    """
    item = {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,

        "src_path": src_path,
        "tgt_path": tgt_path,

        "src_lines": src_lines,
        "tgt_lines": tgt_lines,

        "src_has_sc_path": True / False
        "len": len(src_lines)
    }
    """
    keys = {"src_lang", "tgt_lang", "src_path", "tgt_path", "src_lines", "tgt_lines", "src_has_sc_path", "len"}
    assert set(data_item.keys()) == keys
    assert set(sc_item.keys()) == keys

    assert "SC_{SC_MODEL_ID}" not in data_item["src_path"]
    assert "SC_{SC_MODEL_ID}" not in data_item["tgt_path"]
    assert "SC_{SC_MODEL_ID}" not in sc_item["tgt_path"]

    assert len(data_item["src_lines"]) == len(sc_item["src_lines"]) == len(data_item["tgt_lines"]) == len(sc_item["tgt_lines"])

    assert data_item["src_has_sc_path"] == False
    SC_ITEM_HAS_SC_PATH = sc_item["src_has_sc_path"]

    for key in keys:
        if key == "src_has_sc_path":
            assert data_item[key] == False
        elif key in ["src_path", "src_lines"]:
            if SC_ITEM_HAS_SC_PATH:
                if sc_item[key] == data_item[key]:
                    print("ERROR:")
                    print_item("sc_item:", sc_item)
                    print_item("data_item:", data_item)
                assert sc_item[key] != data_item[key]
            else:
                if sc_item[key] != data_item[key]:
                    print("ERROR:")
                    print_item("sc_item:", sc_item)
                    print_item("data_item:", data_item)
                assert sc_item[key] == data_item[key]
        else:
            if sc_item[key] != data_item[key]:
                print("ERROR:")
                print_item("sc_item:", sc_item)
                print_item("data_item:", data_item)
            assert sc_item[key] == data_item[key]

    if SC_ITEM_HAS_SC_PATH == False:
        # assert sc_item == data_item
        for key in keys:
            if key == "src_has_sc_path":
                assert data_item[key] == False
            else:
                assert sc_item[key] == data_item[key]

    return SC_ITEM_HAS_SC_PATH


def get_dataloader_data(dataloader):
    data = []
    for src_tags, src_lines, sc_lines, tgt_tags, tgt_lines in tqdm(dataloader):
        batch = list(zip(src_tags, src_lines, sc_lines, tgt_tags, tgt_lines))
        for src_tag, src_line, sc_line, tgt_tag, tgt_line in batch:
            data.append((src_tag, src_line, sc_line, tgt_tag, tgt_line))
    return data

def read_csv_f(f, sc_model_id):
    print("#######################################")
    print("READING CSV CONTENT:", f)
    print("\tSC_MODEL_ID=", sc_model_id)
    csv_data = []

    with open(f, newline='') as inf:
        rows = [row for row in csv.reader(inf)]
    header = rows[0]
    assert header == ["src_lang","tgt_lang","src_path","tgt_path"]
    data = [tuple(row) for row in rows[1:]]
    TOTAL_LEN = 0
    for src_lang, tgt_lang, src_path, tgt_path in data:
        print("--")
        print("src_lang:", src_lang, "src_path:", src_path)
        HAS_SC_PATH = False
        if "SC_{SC_MODEL_ID}" in src_path:
            print("\treplacing SC_MODEL_ID in src_path")
            src_path = src_path.replace("{SC_MODEL_ID}", sc_model_id)
            print("\t->", src_path)
            HAS_SC_PATH = True
        assert "SC_{SC_MODEL_ID}" not in tgt_path
        print("tgt_lang:", tgt_lang, "tgt_path:", tgt_path)

        print(f"\n\tsrc_lines read from", src_path)
        src_lines = read_f_lines(src_path)
        print(f"\ttgt_lines read from", tgt_path)
        tgt_lines = read_f_lines(tgt_path)
        assert len(src_lines) == len(tgt_lines)
        item = {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,

            "src_path": src_path,
            "tgt_path": tgt_path,

            "src_lines": src_lines,
            "tgt_lines": tgt_lines,

            "src_has_sc_path": HAS_SC_PATH,
            "len": len(src_lines)
        }
        TOTAL_LEN += len(src_lines)
        csv_data.append(item)
    return csv_data, TOTAL_LEN

def read_whole_f(f):
    with open(f) as inf:
        whole_data = inf.read()
    return whole_data

def read_f_lines(f):
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    return lines

def print_dataloader_data(data, n_lines=100):
    for i, (src_tag, src_line, sc_line, tgt_tag, tgt_line) in enumerate(data):
        print(f"--------------- {i} -----------------")
        print(f"(src) {src_tag}: `{src_line}`")
        print(f"(sc)  {src_tag}: `{sc_line}`")
        print(f"(tgt) {tgt_tag}: `{tgt_line}`")
        if i == n_lines:
            break

def get_sc_dataloader(
    data_csv,
    sc_data_csv=None,
    sc_model_id=None,
    upsample=False,
    shuffle=False,
    PRINT:int=None
):
    assert isinstance(PRINT, int) or PRINT == None
    print("SC DATA:")
    # SHOULD ONLY DO THIS ON TRAINING PROBABLY
    # Upsample=False and shuffle=True used for training data in pretrain -> finetune scenarios.
    sc_dataset = SCAlignedMultilingualDataset(
        data_csv=data_csv,
        sc_data_csv=sc_data_csv,
        sc_model_id=sc_model_id,
        append_src_lang_tok=False,
        append_tgt_lang_tok=False,
        append_tgt_to_src=False,
        upsample=upsample,
        shuffle=shuffle
    )
    sc_dataloader = DataLoader(
        sc_dataset,
        batch_size=1024,
        shuffle=False
    )

    if isinstance(PRINT, int):
        i = 0
        for src_tags, src_lines, sc_lines, tgt_tags, tgt_lines in sc_dataloader:
            batch = list(zip(src_tags, src_lines, sc_lines, tgt_tags, tgt_lines))
            for src_tag, src_line, sc_line, tgt_tag, tgt_line in batch:
                print(f"--------------- {i} -----------------")
                print(f"(src) {src_tag}: `{src_line}`")
                print(f"(sc)  {src_tag}: `{sc_line}`")
                print(f"(tgt) {tgt_tag}: `{tgt_line}`")
                i += 1
            
            if i >= PRINT:
                break
        return None
    else:
        return sc_dataloader

def get_sc_data_set(sc_data):
    sc_data_set = set()
    for src_tag, src_line, sc_line, tgt_tag, tgt_line in sc_data:
        item = (src_tag, src_line, sc_line, tgt_tag, tgt_line)
        assert item not in sc_data_set
        sc_data_set.add(item)
    assert len(sc_data) == len(sc_data_set)
    return sc_data_set
            
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv")
    parser.add_argument("--sc_data_csv")
    parser.add_argument("--sc_model_id")
    parser.add_argument("--pipeline", default="test_ordered,test_shuffled", help="comma-delimited list of tests")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t--{k} = `{v}`")
    print("\n\n")
    return args

if __name__ == "__main__":
    print("-------------------------------------------")
    print("###### tests_parallel_datasets_sc.py ######")
    print("-------------------------------------------")
    args = get_args()
    pipeline = [p.strip() for p in args.pipeline.split(",")]
    main(
        data_csv=args.data_csv,
        sc_data_csv=args.sc_data_csv,
        sc_model_id=args.sc_model_id,
        pipeline=pipeline
    )