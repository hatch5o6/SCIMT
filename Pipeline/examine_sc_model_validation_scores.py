import argparse
import xlsxwriter
import os
import re
import copy
from tqdm import tqdm

SCORES_HEADER = "######## 4.3 Calculate scores ########"
PREAMBLE_REPLACE_UNK_TRUE = "Calculating Scores"
PREAMBLE_REPLACE_UNK_FALSE = "Calculating Scores w/o replacing <unk> in the reference"
FINISHED_LINE = "Finished-----------------------"
FINISHED_TIMESTAMP = r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) [0-9]{1,2} [0-9]{2}:[0-9]{2}:[0-9]{2}(\s(AM|PM))? MST 202[0-9]'
FINISHED_LINE_B = "-------------------------------"
MATCH_MESSAGE = ":) BLEU SCORE EVALUATION MATCHES THAT OF FAIRSEQ :)"
MISMATCH_MESSAGE = "***BLEU SCORE EVALUATION DOES NOT MATCH THAT OF FAIRSEQ!!!***"

WKSHEET_HEADER = ["f", 
                  "RUNK=True eval BLEU", "RUNK=True fairseq BLEU", "RUNK=True match?", 
                  "RUNK=False eval BLEU", "RUNK=False fairseq BLEU", "RUNK=False match?"]

def main(
    slurm_dir,
    # smt_slurm_dir,
    out_path
):
    if not out_path.endswith(".xlsx"):
        raise ValueError(f"out_path `{out_path}` does not end with .xlsx!")
    
    workbook = xlsxwriter.Workbook(out_path)
    format_file = workbook.add_format({"bold": True})
    format_runk_true_header = workbook.add_format({"bg_color": "#6666ff", "bold": True})
    format_runk_true = workbook.add_format({"bg_color": "#ccccff"})
    format_runk_false_header = workbook.add_format({"bg_color": "#cc66ff", "bold": True})
    format_runk_false = workbook.add_format({"bg_color": "#eeccff"})
    format_false = workbook.add_format({"bg_color": "#ff9999"})

    langpair_worksheets = make_worksheets(workbook, slurm_dir, format_file, format_runk_true_header, format_runk_false_header)
    visited_all_jobs = {}
    for f in tqdm(os.listdir(slurm_dir)):
        if f in [".claude", ".git", ".gitignore"]: continue
        if f.endswith(".xlsx"): continue
        # if "_SC_smt" in f:
        #     f_path = os.path.join(smt_slurm_dir, f)
        # else:
        #     
        f_path = os.path.join(slurm_dir, f)
        print("\n\n-----------------------------------------------")
        print("Examining", f_path)
        lang_pair = get_lang_pair(f)

        if lang_pair not in visited_all_jobs:
            visited_all_jobs[lang_pair] = {job_id: False for job_id in range(288)}
            # visited_all_jobs[lang_pair]["SMT"] = False
        
        f_split = f.split(".")
        # if "_SC_smt" in f:
        #     assert f_split[-1] == "out", f"file: {f}, {f_path}"
        #     assert f_split[-2] == lang_pair
        #     job_id == "SMT"
        # else:
        assert f_split[-1] == "out", f"file: {f}, {f_path}"
        assert f_split[-2] == "cfg", f"file: {f}, {f_path}"
        job_id = int(f_split[-3])

        assert job_id in visited_all_jobs[lang_pair]
        visited_all_jobs[lang_pair][job_id] = True

        worksheet = langpair_worksheets[lang_pair]
        row = [f] + get_scores(f_path)
        for c, value in enumerate(row):
            if WKSHEET_HEADER[c] == "f":
                format = format_file
            elif WKSHEET_HEADER[c].startswith("RUNK=True"):
                format = format_runk_true
            elif WKSHEET_HEADER[c].startswith("RUNK=False"):
                format = format_runk_false
            
            if WKSHEET_HEADER[c].endswith("match?") and value == False:
                format = format_false

            worksheet["sheet"].write(worksheet["r"], c, value, format)
        langpair_worksheets[lang_pair]["r"] += 1
    
    for pair in langpair_worksheets:
        langpair_worksheets[pair]["sheet"].autofit()
    workbook.close()

    print("Jobs not visited:")
    for lang_pair, jobs_visted in visited_all_jobs.items():
        fails = [job_id for job_id, passed in jobs_visted.items() if passed == False]
        print(f"\t- {lang_pair}:", fails)
        passes = [job_id for job_id, passed in jobs_visted.items() if passed == True]
        for i in range(288):
            assert i in passes, f"lang {lang_pair} did not pass job {i}"
        # assert "SMT" in passes, f"lang {lang_pair} did not pass job SMT"
        assert len(passes) == 288, f"lang {lang_pair} does not have 288 passes"
        print("\t\tpassed :)")

    
def make_worksheets(workbook, slurm_dir, format_file, format_runk_true_header, format_runk_false_header):
    worksheets = {}
    lang_pairs = get_all_lang_pairs(slurm_dir)
    for pair in lang_pairs:
        worksheets[pair] = {"sheet": workbook.add_worksheet(pair), "r": 1}
        for c, col_heading in enumerate(WKSHEET_HEADER):
            if col_heading == "f":
                format = format_file
            elif col_heading.startswith("RUNK=True"):
                format = format_runk_true_header
            else:
                assert col_heading.startswith("RUNK=False")
                format = format_runk_false_header
            worksheets[pair]["sheet"].write(0, c, col_heading, format)
    return worksheets

def get_all_lang_pairs(slurm_dir):
    lang_pairs = set()
    for f in os.listdir(slurm_dir):
        lang_pairs.add(get_lang_pair(f))
    return lang_pairs

def get_lang_pair(file_name):
    return file_name.split(".")[1]

def job_finished(file_path):
    is_finished = False
    with open(file_path) as inf:
        lines = [l.rstrip() for l in inf.readlines()]
    l = 0
    while l < len(lines):
        line = lines[l]
        if line == FINISHED_LINE:
            l += 1
            line = lines[l]
            assert re.fullmatch(FINISHED_TIMESTAMP, line) != None, f"`{file_path}` line {l} (`{line}`) does not match timestamp regex!"
            l += 1
            line = lines[l]
            assert line == FINISHED_LINE_B
            l += 1
            line = lines[l]
            if file_path in ["/home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/hyper_param_search_outputs/9341031_hyper_param_search.bho-hi.174.cfg.out",
                             "/home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/hyper_param_search_outputs_FAKE/9341031_hyper_param_search.bho-hi.174.cfg.out"]:
                # This file had an issue with clean_slurm_outputs.py -- I'm guessing that it found a file path to delete but before it deleted it, another parallel job got to it.
                l += 4
                line = lines[l]
            assert line == "rm: cannot remove '/home/hatch5o6/Cognate/code/core*': No such file or directory", f"`{file_path}`line {l} is not removing core failure"
            assert l == len(lines) - 1
            is_finished = True
        l += 1
    assert l == len(lines)
    return is_finished

def get_scores(file_path):
    if not job_finished(file_path):
        return ["NOT_FINISHED" for i in range(6)]
    score_lines_replace_unk_true, score_lines_replace_unk_false = read_score_lines_from_file(file_path)
    print("parse_score_lines runk=true: ", file_path)
    (eval_bleu_score_runktrue, 
     eval_bleu_runktrue, 
     fairseq_bleu_score_runktrue, 
     fairseq_bleu_runktrue,
     matches_fairseq_runktrue) = parse_score_lines(score_lines_replace_unk_true)
    if eval_bleu_score_runktrue == fairseq_bleu_score_runktrue:
        # assert matches_fairseq_runktrue == True, f"{file_path} get_scores failed: assert matches_fairseq_runktrue == True"
        pass
    else:
        assert matches_fairseq_runktrue == False, f"{file_path} get_scores failed: assert matches_fairseq_runktrue == False"

    print("parse_score_lines runk=false: ", file_path)
    (eval_bleu_score_runkfalse, 
     eval_bleu_runkfalse, 
     fairseq_bleu_score_runkfalse, 
     fairseq_bleu_runkfalse, 
     matches_fairseq_runkfalse) = parse_score_lines(score_lines_replace_unk_false, replace_unk=False)
    if eval_bleu_score_runkfalse == fairseq_bleu_score_runkfalse:
        # assert matches_fairseq_runkfalse == True, f"{file_path} get_scores failed: assert matches_fairseq_runkfalse == True"
        pass
    else:
        assert matches_fairseq_runkfalse == False, f"{file_path} get_scores failed: assert matches_fairseq_runkfalse == False"
    
    return [eval_bleu_score_runktrue, fairseq_bleu_score_runktrue, matches_fairseq_runktrue, 
            eval_bleu_score_runkfalse, fairseq_bleu_score_runkfalse, matches_fairseq_runkfalse]
    
    

def parse_score_lines(score_lines, replace_unk=True):
    if replace_unk:
        preamble = PREAMBLE_REPLACE_UNK_TRUE
    else:
        preamble = PREAMBLE_REPLACE_UNK_FALSE

    message = None
    eval_bleu = None
    fairseq_bleu = None
    matches_fairseq = None
    n_times_default_case_after_bleu_stuff = 0
    i = 0
    finished_getting_scores_at = None
    while i < len(score_lines):
        line = score_lines[i]
        match i:
            case 0:
                assert line == preamble
            case 1:
                assert line.startswith("    python Pipeline/evaluate.py --ref ")
            case 2:
                assert line == "evaluate.py"
            case 3:
                assert line == "Arguments:"
            case 4:
                assert line.startswith("\t- ref: `")
            case 5:
                assert line.startswith("\t- hyp: `")
            case 6:
                assert line.startswith("\t- out: `")
            case 7:
                assert line.startswith("\t- target_vocab: `")
            case 8:
                assert line.startswith("\t- hyp_out_txt: `")
            case 9:
                assert line.startswith(f"\t- REPLACE_UNK: `{replace_unk}`")
            case _:
                print("DEFAULT CASE", i)
                assert i > 9
                if line == "BLEU STUFF":
                    print("FOUND BLEU STUFF", i)
                    i += 1
                    line = score_lines[i]
                    assert line.startswith("BLEU = ")
                    assert eval_bleu == None
                    eval_bleu = line
                    i += 3
                    line = score_lines[i]
                    assert line in [MATCH_MESSAGE, MISMATCH_MESSAGE], print(f"line {i} `{line}` is not a MATCH or MISMATCH MESSAGE")
                    assert message == None
                    message = line
                    i += 10
                    line = score_lines[i]
                    assert line == f"BLEU_DETAILS: {eval_bleu}"
                    i += 1
                    line = score_lines[i]
                    assert line.startswith("BLEU_SCORE: ")
                    i += 1
                    line = score_lines[i]
                    assert line.startswith("FAIRSEQ_BLEU: ")
                    assert fairseq_bleu == None
                    fairseq_bleu = line.split("FAIRSEQ_BLEU: ")[1]
                    i += 1
                    line = score_lines[i]
                    assert line == message
                    assert matches_fairseq == None
                    matches_fairseq = fairseq_bleu == eval_bleu
                    # print("___eval_bleu:", eval_bleu)
                    # print("fairseq_bleu:", fairseq_bleu)
                    # print("_______match:", fairseq_bleu == eval_bleu)
                    # print("_____message:", message)
                    if matches_fairseq:
                        assert message == MATCH_MESSAGE
                    else:
                        assert message == MISMATCH_MESSAGE
                    n_times_default_case_after_bleu_stuff += 1
                    print(f"GOT SCORES, breaking at i = {i}")
                    assert finished_getting_scores_at == None
                    finished_getting_scores_at = copy.deepcopy(i)
                    break
        i += 1
    assert i == finished_getting_scores_at
    print(f"i is still {i}")

    assert n_times_default_case_after_bleu_stuff == 1
    eval_bleu_score = parse_bleu(eval_bleu)
    fairseq_bleu_score = parse_bleu(fairseq_bleu)

    return eval_bleu_score, eval_bleu, fairseq_bleu_score, fairseq_bleu, matches_fairseq

def parse_bleu(bleu_string):
    bleu, equals, score = bleu_string.split()[:3]
    assert bleu == "BLEU"
    assert equals == "="
    score = float(score)
    return score

def read_score_lines_from_file(f):
    score_lines = []
    with open(f) as inf:
        IN_SCORE_LINES = False
        line = inf.readline()
        while line:
            line = line.rstrip()
            if line == SCORES_HEADER:
                assert IN_SCORE_LINES == False
                IN_SCORE_LINES = True
            if line == FINISHED_LINE:
                assert IN_SCORE_LINES == True
                IN_SCORE_LINES = False
            
            if IN_SCORE_LINES == True:
                score_lines.append(line)
            line = inf.readline()
    assert score_lines[0] == SCORES_HEADER
    score_lines = score_lines[1:]
    dividing_idx = get_score_lines_dividing_index(score_lines)
    scores_replace_unk_true = score_lines[:dividing_idx]
    scores_replace_unk_false = score_lines[dividing_idx:]

    assert scores_replace_unk_true[0] == PREAMBLE_REPLACE_UNK_TRUE
    assert scores_replace_unk_false[0] == PREAMBLE_REPLACE_UNK_FALSE

    return scores_replace_unk_true, scores_replace_unk_false

def get_score_lines_dividing_index(score_lines):
    dividing_idx = None
    for i, line in enumerate(score_lines):
        if line == PREAMBLE_REPLACE_UNK_FALSE:
            assert dividing_idx is None
            dividing_idx = i
            break
    assert dividing_idx is not None
    return dividing_idx

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slurm_outputs", 
        help="path to folder containing slurm outputs of the hyperparameter search", 
        default="/home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/hyper_param_search_outputs")
    # parser.add_argument("-S", "--SMT_slurm_outputs",
    #     default="/home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/SC_smt")
    parser.add_argument("-o", "--out", help="path to write excel sheet")
    args = parser.parse_args()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"--{k}=`{v}`")
    print("\n\n")
    return args

if __name__ == "__main__":
    print("#########################################")
    print("# examine_sc_model_validation_scores.py #")
    print("#########################################")
    args = get_args()
    main(
        args.slurm_outputs, 
        # args.SMT_slurm_outputs, 
        args.out)