import argparse
import os
import shutil
import yaml
from transformers import BartForConditionalGeneration, BartConfig
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_info
import json
import re

from parallel_datasets import MultilingualDataset
from spm_tokenizers import SPMTokenizer
from char_tokenizers import CharacterTokenizer
from LightningBART import LBART
from evaluate import calc_bleu, calc_chrF
from build_loss_graph import write_loss_graph


SAVE_DIR_SUBDIRS = ["checkpoints", "logs", "tb", "predictions", "character_vocab"]

def train_model(config, REVERSE_SRC_TGT=False):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print("WORLD_SIZE:", world_size)

    rank_zero_info(f"REVERSE_SRC_TGT: {REVERSE_SRC_TGT}")

    if REVERSE_SRC_TGT:
        src_lang = config["src"]
        tgt_lang = config["tgt"]
        config["src"] = tgt_lang
        config["tgt"] = src_lang
        rank_zero_info("Reversed source lang '{src_lang}' and target lang '{tgt_lang}'.\n")

    rank_zero_info("train_model")
    rank_zero_info("TRAINING CONFIG:")
    for k, v in config.items():
        rank_zero_info(f"\t-{k}: {v}, {type(v)}")

    L.seed_everything(config["seed"], workers=True)

    parent_dir = "/".join(config["save"].split("/")[:-1])
    # if not os.path.exists(parent_dir):
    if int(os.environ.get("RANK", 0)) == 0:
        rank_zero_info(f"Creating parent dir: {parent_dir}")
        os.makedirs(parent_dir, exist_ok=True)

    save_dir = os.path.join(config["save"] + f"_TRIAL_s={config['seed']}")
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    logs_dir = os.path.join(save_dir, "logs")
    tb_dir = os.path.join(save_dir, "tb")
    rank_zero_info(f"SAVE_DIR {save_dir}")
    if os.path.exists(save_dir):
        for item in os.listdir(save_dir):
            if item not in SAVE_DIR_SUBDIRS:
                raise ValueError(f"Invalid subdir/file '{item}' in {save_dir}!")
            item_dir = os.path.join(save_dir, item)
            if not os.path.isdir(item_dir) or list(os.listdir(item_dir)) != []:
                raise ValueError(f"Subdir/file '{item}' must be an empty directory in order to begin training this model!")
    
    if int(os.environ.get("RANK", 0)) == 0:
        rank_zero_info(f"Creating {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        rank_zero_info(f"Creating {checkpoints_dir}")
        os.makedirs(checkpoints_dir, exist_ok=True)
        rank_zero_info(f"Creating {logs_dir}")
        os.makedirs(logs_dir, exist_ok=True)
        rank_zero_info(f"Creating {tb_dir}")
        os.makedirs(tb_dir, exist_ok=True)

    dataloaders = get_multilingual_dataloaders(config, REVERSE_SRC_TGT=REVERSE_SRC_TGT, sections=["train", "val"])
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]

    # rank_zero_info("TESTING TRAIN DATALOADER")
    # examine_dataloader(train_dataloader)
    # rank_zero_info("\n\nTESTING VAL DATALOADER")
    # examine_dataloader(val_dataloader)
    # exit()

    if config["do_char"] == True:
        char_vocab_dir = os.path.join(save_dir, "character_vocab")
        if int(os.environ.get("RANK", 0)) == 0:
            os.makedirs(char_vocab_dir, exist_ok=True)
        char_vocab_path = os.path.join(char_vocab_dir, "vocab.csv")
        src_tokenizer, tgt_tokenizer = get_char_tokenizers(config, char_vocab_path)
    else:
        src_tokenizer, tgt_tokenizer = get_spm_tokenizers(config)

    # rank_zero_info("EXAMINING TOKENIZERS ON TRAIN DATALOADER")
    # examine_tokenizers(src_tokenizer, tgt_tokenizer, train_dataloader)
    # exit()

    model = get_bart_model(src_tokenizer, tgt_tokenizer, config)

    if config["from_pretrained"] not in [None, "None"]:

        if not config["from_pretrained"].endswith(".ckpt"):
            pretrain_predictions_dir = os.path.join(config["from_pretrained"], "predictions")
            print("FROM PRETRAINED PREDICTIONS:", pretrain_predictions_dir)
            assert os.path.exists(pretrain_predictions_dir)
            assert os.path.isdir(pretrain_predictions_dir)
            print("Choosing pretrained model to fine-tune.")
            chosen_pretrained_checkpoint = choose_checkpoint(pretrain_predictions_dir)
        else:
            print("Model to fine-tune is set to", config["from_pretrained"])
            chosen_pretrained_checkpoint = config["from_pretrained"]
        assert chosen_pretrained_checkpoint.endswith('.ckpt')
        
        print("WILL FINETUNE MODEL:", chosen_pretrained_checkpoint)

        lightning_model = LBART.load_from_checkpoint(
            checkpoint_path=chosen_pretrained_checkpoint,
            model=model,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            config=config
        )
    else:
        print("WILL TRAIN FROM SCRATCH")
        lightning_model = LBART(
            model=model,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            config=config
        )

    # callbacks
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=config["early_stop"], verbose=True)
    top_k_model_checkpoint = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="{epoch}-{step}-{val_loss:.4f}",
        save_top_k=config["save_top_k"],
        monitor="val_loss",
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    print_callback = PrintCallback()
    train_callbacks = [
        early_stopping,
        top_k_model_checkpoint,
        lr_monitor,
        print_callback
    ]
    logger = CSVLogger(save_dir=logs_dir)
    tb_logger = TensorBoardLogger(save_dir=tb_dir)

    if config["n_gpus"] >= 1:
        strategy = "ddp"
    else:
        strategy = "auto"

    trainer = L.Trainer(
        max_steps=config["max_steps"],
        val_check_interval=config["val_interval"],
        accelerator=config["device"],
        default_root_dir=save_dir,
        callbacks=train_callbacks,
        logger=[logger, tb_logger],
        deterministic=True,
        strategy=strategy,
        # log_every_n_steps=1,
        gradient_clip_val=config["gradient_clip_val"]
    )

    # if config["from_pretrained"] is not None:
    #     print("RESUMING TRAINING FROM", config["from_pretrained"])
    #     trainer.fit(
    #         lightning_model,
    #         train_dataloader,
    #         val_dataloader,
    #         ckpt_path=config["from_pretrained"]
    #     )
    # else:
    trainer.fit(
        lightning_model,
        train_dataloader,
        val_dataloader
    )

def examine_tokenizers(src_tokenizer, tgt_tokenizer, dataloader):
    rank_zero_info(f"SRC TOKENIZER: {src_tokenizer.spm_name}")
    rank_zero_info(f"TGT TOKENIZER: {tgt_tokenizer.spm_name}")

    printed = 0
    for b, (src_sents, tgt_sents) in enumerate(dataloader):
        if b % 20 == 0:
            batch = list(zip(src_sents, tgt_sents))
            rank_zero_info(f"####### BATCH {b} #######")
            for i, (src_sent, tgt_sent) in enumerate(batch):
                if i % 20 == 0:
                    src_tok_ids, src_toks = src_tokenizer.tokenize(src_sent)
                    tgt_tok_ids, tgt_toks = tgt_tokenizer.tokenize(tgt_sent)
                    rank_zero_info(f"----- ({i}) -----")
                    rank_zero_info(f"SRC______: {src_sent}")
                    rank_zero_info(f"SRCTOKS__: {src_toks}")
                    rank_zero_info(f"SRCTOKIDS: {src_tok_ids}")
                    rank_zero_info(f"TGT______: {tgt_sent}")
                    rank_zero_info(f"TGTTOKS__: {tgt_toks}")
                    rank_zero_info(f"TGTTOKIDS: {tgt_tok_ids}")
                    printed += 1
                if printed == 20:
                    break
        if printed == 20:
            break


def examine_dataloader(dataloader):
    printed = 0
    for b, (src_sents, tgt_sents) in enumerate(dataloader):
        if b % 20 == 0:
            batch = list(zip(src_sents, tgt_sents))
            rank_zero_info(f"####### BATCH {b} #######")
            for i, (src_sent, tgt_sent) in enumerate(batch):
                if i % 20 == 0:
                    rank_zero_info(f"----- ({i}) -----")
                    rank_zero_info(f"SRC: {src_sent}")
                    rank_zero_info(f"TGT: {tgt_sent}")
                    printed += 1
                if printed == 20:
                    break
        if printed == 20:
            break


def get_predictions(dataloader, src_tokenizer, tgt_tokenizer, config):
    model = get_bart_model(src_tokenizer, tgt_tokenizer, config)

    lightning_model = LBART.load_from_checkpoint(
        checkpoint_path=config["test_checkpoint"],
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        config=config
    )
    lightning_model.eval()

    trainer = L.Trainer(
        accelerator=config["device"]
    )

    batch_predictions = trainer.predict(
        lightning_model,
        dataloaders=dataloader
    )

    predictions = []
    for b, batch in enumerate(batch_predictions):
        predictions += batch

    return predictions


def test_model(config, REVERSE_SRC_TGT=False):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print("WORLD_SIZE:", world_size)
    assert world_size == 1, f"Expected 1 rank, but got {world_size}"
    rank = int(os.environ.get("RANK", 0))
    assert rank == 0

    rank_zero_info(f"REVERSE_SRC_TGT: {REVERSE_SRC_TGT}")

    if REVERSE_SRC_TGT:
        src_lang = config["src"]
        tgt_lang = config["tgt"]
        config["src"] = tgt_lang
        config["tgt"] = src_lang
        rank_zero_info("Reversed source lang '{src_lang}' and target lang '{tgt_lang}'.\n")

    print("test_model")
    print("config:")
    for k, v in config.items():
        print(f"{k}: {v}")

    L.seed_everything(config["seed"], workers=True)

    save_dir = os.path.join(config["save"] + f"_TRIAL_s={config['seed']}")
    # Create loss graphs
    write_loss_graph(save_dir)

    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    logs_dir = os.path.join(save_dir, "logs")
    if "predictions_dir" in config:
        predictions_dir = os.path.join(save_dir, config["predictions_dir"])
    else:
        predictions_dir = os.path.join(save_dir, "predictions")
    # if not os.path.exists(predictions_dir):
    if int(os.environ.get("RANK", 0)) == 0:
        os.makedirs(predictions_dir, exist_ok=True)

    for d in [save_dir, checkpoints_dir, logs_dir, predictions_dir]:
        if not os.path.exists(d):
            raise ValueError(f"Expected to find directory {d}!")
    
    dataloaders = get_multilingual_dataloaders(config, REVERSE_SRC_TGT=REVERSE_SRC_TGT, sections=["val", "test"])
    val_dataloader = dataloaders["val"]
    test_dataloader = dataloaders["test"]

    # print("TESTING TEST DATALOADER")
    # examine_dataloader(test_dataloader)
    # exit()

    src_tokenizer, tgt_tokenizer = get_spm_tokenizers(config)

    if config["test_checkpoint"] in [None, "None", "null"]:
        assert os.path.exists(checkpoints_dir)
        print("No checkpoint provided for testing. Will test all and choose.")
        checkpoints_to_test = [
            os.path.join(checkpoints_dir, ckpt_file_path)
            for ckpt_file_path in os.listdir(checkpoints_dir)
        ]
    else:
        print("Testing checkpoint provided:", config["test_checkpoint"])
        checkpoints_to_test = [config["test_checkpoint"]]

    all_ckpt_scores = {}
    for proposed_ckpt_file in checkpoints_to_test:
        config["test_checkpoint"] = proposed_ckpt_file
        print("## Evaluating", config["test_checkpoint"])

        val_results = get_predictions(
            dataloader=val_dataloader,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            config=config
        )
        test_results = get_predictions(
            dataloader=test_dataloader,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            config=config
        )

        val_metrics, val_chkpt_predictions_dir = get_metrics(results=val_results, predictions_dir=predictions_dir, config=config, metrics_set="val")
        test_metrics, test_chkpt_predictions_dir = get_metrics(results=test_results, predictions_dir=predictions_dir, config=config, metrics_set="test")
        assert val_chkpt_predictions_dir == test_chkpt_predictions_dir
        chkpt_predictions_dir = test_chkpt_predictions_dir
        
        metrics = {"val": val_metrics, "test": test_metrics}

        print("METRICS")
        print(metrics)
        save_metrics = os.path.join(chkpt_predictions_dir, "metrics.json")
        with open(save_metrics, "w") as outf:
            outf.write(json.dumps(metrics, indent=2))
        
        assert config["test_checkpoint"] not in all_ckpt_scores
        all_ckpt_scores[config["test_checkpoint"]] = metrics
    
    best_bleu_ckpt = None
    best_bleu_score = None
    for ckpt_path, ckpt_metrics in all_ckpt_scores.items():
        if best_bleu_ckpt is None:
            assert best_bleu_score is None
            best_bleu_ckpt = ckpt_path
            best_bleu_score = ckpt_metrics["val"]["BLEU"]
        else:
            if ckpt_metrics["val"]["BLEU"] > best_bleu_score:
                best_bleu_score = ckpt_metrics["val"]["BLEU"]
                best_bleu_ckpt = ckpt_path
    assert "BEST_VAL_BLEU_CHECKPOINT" not in all_ckpt_scores
    assert all_ckpt_scores[best_bleu_ckpt]["val"]["BLEU"] == best_bleu_score
    all_ckpt_scores["BEST_VAL_BLEU_CHECKPOINT"] = {
        "checkpoint": best_bleu_ckpt,
        "val_BLEU": best_bleu_score,
        "val_chrF": all_ckpt_scores[best_bleu_ckpt]["val"]["chrF"],
        "test_BLEU": all_ckpt_scores[best_bleu_ckpt]["test"]["BLEU"],
        "test_chrF": all_ckpt_scores[best_bleu_ckpt]["test"]["chrF"]
    }
    all_ckpt_scores["TEST_DATA"] = config["test_data"]
    all_ckpt_scores["VAL_DATA"] = config["val_data"]

    save_all_metrics = os.path.join(predictions_dir, "all_scores.json")
    with open(save_all_metrics, "w") as outf:
        outf.write(json.dumps(all_ckpt_scores, ensure_ascii=False, indent=2))

# def choose_checkpoint(checkpoints_dir):
#     print("Choosing checkpoint with lowest validation loss")
#     chosen_val_loss = None
#     chosen_f = None
#     for f in os.listdir(checkpoints_dir):
#         val_loss = float(f.split("-val_loss=")[-1].split(".ckpt")[0])
#         if chosen_f == None:
#             assert chosen_val_loss == None
#             chosen_f = f
#             chosen_val_loss = val_loss
#         else:
#             assert isinstance(chosen_f, str)
#             assert isinstance(chosen_val_loss, float)
#             if val_loss < chosen_val_loss:
#                 chosen_f = f
#                 chosen_val_loss = val_loss
#     assert chosen_f != None
#     chosen_f_path = os.path.join(checkpoints_dir, chosen_f)
#     print("CHOSE CHECKPOINT", chosen_f_path)
#     return chosen_f_path

def get_metrics(results, predictions_dir, config, metrics_set="test"):
    assert metrics_set in ["test", "val"]
    src_data = []
    ref_data = []
    preds = []
    for generated_ids, src_segments_text, prediction, tgt_segments_text in results:
        src_data.append(src_segments_text)
        ref_data.append(tgt_segments_text)
        preds.append(prediction)    
    
    assert len(src_data) == len(ref_data) == len(preds)
    
    chkpt_name = config["test_checkpoint"].split("/")[-1]
    chkpt_predictions_dir = os.path.join(predictions_dir, chkpt_name)
    if os.path.exists(chkpt_predictions_dir):
        shutil.rmtree(chkpt_predictions_dir)
    if int(os.environ.get("RANK", 0)) == 0:
        os.makedirs(chkpt_predictions_dir, exist_ok=True)

    # Predictions
    save_preds = os.path.join(chkpt_predictions_dir, f"{metrics_set}_predictions.txt")
    with open(save_preds, "w") as outf:
        outf.write("\n".join(preds) + "\n")
    
    # Metrics
    bleu_score = calc_bleu(preds, [ref_data])
    chrf_score, chrf_sents = calc_chrF(preds, [ref_data])

    metrics = {
        "BLEU": bleu_score,
        "chrF": chrf_score,
        # "chrF_sents": chrf_sents
        # "test_data": config["test_data"]
    }
    return metrics, chkpt_predictions_dir

def choose_checkpoint(predictions_dir):
    all_metrics_f = os.path.join(predictions_dir, "all_scores.json")
    print(f"all_metrics_f: {all_metrics_f}")
    with open(all_metrics_f) as inf:
        all_metrics = json.load(inf)
    return all_metrics["BEST_VAL_BLEU_CHECKPOINT"]["checkpoint"]

def inference(
    predict_config,
    checkpoint_path,
    name,
    VERBOSE=True,
    div="test",
    REVERSE_SRC_TGT=False
):
    pass
    # TODO Write this function


    # pred_dataset = ParallelDataset(
    #     src_file_path=config[f"{div}_src_data"],
    #     tgt_file_path=config[f"{div}_tgt_data"]
    # )
    # pred_dataloader = DataLoader(
    #     pred_dataset, 
    #     batch_size=config["pred_batch_size"]
    # )
    # tokenizer = CharacterTokenizer(
    #     vocab_path=predict_config["vocab_path"]
    # )

    # model_config = BartConfig()
    # model_config.vocab_size = 300
    # model_config.encoder_layers = config["encoder_layers"]
    # model_config.decoder_layers = config["decoder_layers"]
    # model_config.encoder_attention_heads = config["encoder_attention_heads"]
    # model_config.decoder_attention_heads = config["decoder_attention_heads"]
    # model_config.pad_token_id=tokenizer.token2idx[tokenizer.pad]
    # model_config.bos_token_id=tokenizer.token2idx[tokenizer.bos]
    # model_config.eos_token_id=tokenizer.token2idx[tokenizer.eos]
    # model_config.forced_eos_token_id=tokenizer.token2idx[tokenizer.eos]
    # model_config.decoder_start_token_id=tokenizer.token2idx[tokenizer.bos]

    # model = BartForConditionalGeneration(model_config)
    # model.eval()

    # print("CONFIG")
    # for key, value in config.items():
    #     print(f"{key}:{value}")

    # print("MODEL", type(model), model)
    # print("CHKPT", checkpoint_path)

    # lightning_model = LightningMSABART.load_from_checkpoint(
    #     checkpoint_path,
    #     model=model,
    #     tokenizer=tokenizer,
    #     config=config
    # )
    # lightning_model.eval()
    # trainer = L.Trainer(
    #     accelerator=config["device"],
    #     devices=config["num_gpus"]
    # )
    # predictions = trainer.predict(lightning_model, dataloaders=pred_dataloader)
    # pred_dir = os.path.join(checkpoint_path.split("checkpoints")[0], "predictions")
    # if not os.path.exists(pred_dir):
    #     os.mkdir(pred_dir)
    # pred_file = os.path.join(pred_dir, f"{checkpoint_path.split('/')[-1].split('.txt')[0]}_{div}_preds.txt")
    # verbose_file = os.path.join(pred_dir, f"{checkpoint_path.split('/')[-1].split('.txt')[0]}_{div}_verbose.txt")
    # hyp_sents = []
    # ref_sents = []
    # with open(pred_file, 'w') as predf, open(verbose_file, 'w') as verbf:
    #     for b, batch in enumerate(predictions):
    #         print("BATCH", b, "SIZE", len(batch))
    #         for i, (generated_ids, src_text, prediction, tgt_text) in enumerate(batch):
    #             predf.write(prediction.strip() + "\n")
    #             print(prediction)

    #             if VERBOSE:
    #                 verbf.write(f"####################### {b}-{i} ########################\n")
    #                 verbf.write("SOURCE\n")
    #                 verbf.write(src_text.strip() + "\n")
    #                 verbf.write(f"IDS ({len(generated_ids)})\n")
    #                 verbf.write(f"{generated_ids}\n")
    #                 verbf.write("PREDICTION\n")
    #                 verbf.write(prediction.strip() + "\n")
    #                 verbf.write("TARGET\n")
    #                 verbf.write(tgt_text.strip() + "\n")
    #                 hyp_sents.append(prediction)
    #                 ref_sents.append(tgt_text)
    #     correct, total, accuracy = calc_accuracy(ref_sents, hyp_sents)
    #     verbf.write(f"\nCORRECT: {correct}\tTOTAL: {total}\tACCURACY: {accuracy}")
    
    # return hyp_sents, correct, total, accuracy


def get_multilingual_dataloaders(config, REVERSE_SRC_TGT=False, sections=["train", "val", "test", "inference"], RETURN_DATASETS=False):
    sections = sorted(list(set(sections)))
    for section in sections:
        if section not in ["train", "val", "test", "inference"]:
            raise ValueError(f"Dataloader sections must be in ['train', 'val', 'test', 'inference']")
        # if section in ["train", "val"]:
        #     assert config[f"{section}_data"].endswith(f"/{section}.no_overlap_v1.csv")
        # else:
        #     assert config[f"{section}_data"].endswith(f"/{section}.csv")
        assert config[f"{section}_data"].endswith(f"/{section}.csv")

    dataloaders = {}
    datasets = {}
    for section in sections:
        print("\nGETTING DATALOADER FOR SECTION", section)
        SHUFFLE = False
        UPSAMPLE = False
        if section == "train":
            SHUFFLE = True
            UPSAMPLE = config["upsample"] # Only upsample for train set (and if it's indicated in the config)
        a_dataset = MultilingualDataset(
            data_csv=config[f"{section}_data"],
            sc_model_id=config["sc_model_id"],
            append_src_lang_tok=config["append_src_token"],
            append_tgt_lang_tok=config["append_tgt_token"],
            upsample=UPSAMPLE,
            shuffle=SHUFFLE,
            REVERSE_SRC_TGT=REVERSE_SRC_TGT
        )
        datasets[section] = a_dataset
        print(f"{section} Dataloader, shuffle={SHUFFLE}")
        a_dataloader = DataLoader(
            a_dataset, 
            batch_size=config[f"{section}_batch_size"],
            shuffle=SHUFFLE
        )
        dataloaders[section] = a_dataloader

    if RETURN_DATASETS:
        return dataloaders, datasets
    else:
        return dataloaders

def get_spm_tokenizers(config):
    # TODO src and tgt or just one? Probs two.
    # I changed my mind. Just one. Though I already implemented for two.
        # So we'll just set both src and tgt tokenizers to the same one :)
        # They'll be duplicates of each other.
    src_tokenizer = SPMTokenizer(
        # spm_name=config["src_spm"]
        spm_name=config["spm"]
    )
    tgt_tokenizer = SPMTokenizer(
        # spm_name=config["tgt_spm"]
        spm_name=config["spm"]
    )
    tokenizer_asserts(src_tokenizer, tgt_tokenizer)

    rank_zero_info("\nSpecial Toks")
    for i, (tok, idx) in enumerate(src_tokenizer.token2idx.items()):
        rank_zero_info(f"{tok}:{idx}")
        rank_zero_info(f"{idx}: {src_tokenizer.idx2token[idx]}")
        rank_zero_info("-------------------------")
        if i == 5:
            break

    return src_tokenizer, tgt_tokenizer

def get_char_tokenizers(config, vocab_path):
    data_dir = "/".join(config["spm"].split("/")[:-1])
    text_files = []
    for f in os.listdir(data_dir):
        if re.fullmatch(r'training_data\.s=\d{1,4}\.txt', f):
            text_files.append(f)
    # make sure we only found 1 file
    assert len(text_files) == 1
    data_file = os.path.join(data_dir, text_files[0])
    assert "div=" not in data_file # extra check to make sure we didn't get any of lang-specific files

    # src and tgt tokenizers should be the exact same
    src_tokenizer = CharacterTokenizer(vocab_path=vocab_path, data_paths=[data_file])
    tgt_tokenizer = CharacterTokenizer(vocab_path=vocab_path, data_paths=[data_file])

    return src_tokenizer, tgt_tokenizer



def get_bart_model(src_tokenizer, tgt_tokenizer, config):
    model_vocab_size = max(src_tokenizer.vocab_size, tgt_tokenizer.vocab_size)
    rank_zero_info(f"MODEL VOCAB SIZE: {model_vocab_size}")
    
    # TODO Set all BartConfigs :)
    rank_zero_info("\nget_bart_model() SETTING CONFIGS:")
    for k,v in config.items():
        rank_zero_info(f"{k}:`{v}`")
    rank_zero_info("\n")
    model_config = BartConfig()
    model_config.vocab_size = model_vocab_size
    model_config.pad_token_id=src_tokenizer.token2idx[src_tokenizer.pad]
    model_config.bos_token_id=src_tokenizer.token2idx[src_tokenizer.bos]
    model_config.eos_token_id=src_tokenizer.token2idx[src_tokenizer.eos]
    model_config.forced_eos_token_id=src_tokenizer.token2idx[src_tokenizer.eos]
    model_config.decoder_start_token_id=src_tokenizer.token2idx[src_tokenizer.bos]
    
    # Model Config
    model_config.encoder_layers = config["encoder_layers"]
    model_config.decoder_layers = config["decoder_layers"]
    model_config.encoder_attention_heads = config["encoder_attention_heads"]
    model_config.decoder_attention_heads = config["decoder_attention_heads"]
    model_config.encoder_ffn_dim = config["encoder_ffn_dim"]
    model_config.decoder_ffn_dim = config["decoder_ffn_dim"]
    model_config.encoder_layerdrop = config["encoder_layerdrop"]
    model_config.decoder_layerdrop = config["decoder_layerdrop"]

    model_config.max_position_embeddings = config["max_position_embeddings"]
    model_config.d_model = config["d_model"]
    model_config.dropout = config["dropout"]

    model = BartForConditionalGeneration(model_config)

    return model


def tokenizer_asserts(src_tokenizer, tgt_tokenizer):
    assert src_tokenizer.pad == tgt_tokenizer.pad
    assert src_tokenizer.bos == tgt_tokenizer.bos
    assert src_tokenizer.eos == tgt_tokenizer.eos
    assert src_tokenizer.unk == tgt_tokenizer.unk

    assert src_tokenizer.token2idx[src_tokenizer.pad] == tgt_tokenizer.token2idx[tgt_tokenizer.pad]
    assert src_tokenizer.token2idx[src_tokenizer.bos] == tgt_tokenizer.token2idx[tgt_tokenizer.bos]
    assert src_tokenizer.token2idx[src_tokenizer.eos] == tgt_tokenizer.token2idx[tgt_tokenizer.eos]
    assert src_tokenizer.token2idx[src_tokenizer.unk] == tgt_tokenizer.token2idx[tgt_tokenizer.unk]


class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        rank_zero_info("################# (Lightning) #################")
        rank_zero_info("######          TRAINING STARTED         ######")
        rank_zero_info("###############################################")
    def on_train_end(self, trainer, pl_module):
        rank_zero_info("################# (Lightning) #################")
        rank_zero_info("#######          TRAINING ENDED         #######")
        rank_zero_info("###############################################")

def read_config(f):
    with open(f) as inf:
        config = yaml.safe_load(inf)
    config["learning_rate"] = float(config["learning_rate"])
    config["warmup_steps"] = round(0.05 * config["max_steps"])

    # asserts
    assert config["val_interval"] in [0.25, 0.5]
    assert config["early_stop"] in [5, 10]
    if config["val_interval"] == 0.25:
        assert config["early_stop"] == 10
    if config["val_interval"] == 0.5:
        assert config["early_stop"] == 5


    return config


def test_dataloaders(config, REVERSE_SRC_TGT=False):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print("WORLD_SIZE:", world_size)

    rank_zero_info(f"REVERSE_SRC_TGT: {REVERSE_SRC_TGT}")

    if REVERSE_SRC_TGT:
        src_lang = config["src"]
        tgt_lang = config["tgt"]
        config["src"] = tgt_lang
        config["tgt"] = src_lang
        rank_zero_info("Reversed source lang '{src_lang}' and target lang '{tgt_lang}'.\n")

    rank_zero_info("train_model")
    rank_zero_info("TRAINING CONFIG:")
    for k, v in config.items():
        rank_zero_info(f"\t-{k}: {v}, {type(v)}")

    L.seed_everything(config["seed"], workers=True)

    # parent_dir = "/".join(config["save"].split("/")[:-1])
    # if not os.path.exists(parent_dir):
    # if int(os.environ.get("RANK", 0)) == 0:
    #     rank_zero_info(f"Creating parent dir: {parent_dir}")
    #     os.makedirs(parent_dir, exist_ok=True)

    # save_dir = os.path.join(config["save"] + f"_TRIAL_s={config['seed']}")
    # checkpoints_dir = os.path.join(save_dir, "checkpoints")
    # logs_dir = os.path.join(save_dir, "logs")
    # tb_dir = os.path.join(save_dir, "tb")
    # rank_zero_info(f"SAVE_DIR {save_dir}")
    # if os.path.exists(save_dir):
    #     for item in os.listdir(save_dir):
    #         if item not in SAVE_DIR_SUBDIRS:
    #             raise ValueError(f"Invalid subdir/file '{item}' in {save_dir}!")
    #         item_dir = os.path.join(save_dir, item)
    #         if not os.path.isdir(item_dir) or list(os.listdir(item_dir)) != []:
    #             raise ValueError(f"Subdir/file '{item}' must be an empty directory in order to begin training this model!")
    
    # if int(os.environ.get("RANK", 0)) == 0:
    #     rank_zero_info(f"Creating {save_dir}")
    #     os.makedirs(save_dir, exist_ok=True)
    #     rank_zero_info(f"Creating {checkpoints_dir}")
    #     os.makedirs(checkpoints_dir, exist_ok=True)
    #     rank_zero_info(f"Creating {logs_dir}")
    #     os.makedirs(logs_dir, exist_ok=True)
    #     rank_zero_info(f"Creating {tb_dir}")
    #     os.makedirs(tb_dir, exist_ok=True)

    dataloaders = get_multilingual_dataloaders(config, REVERSE_SRC_TGT=REVERSE_SRC_TGT, sections=["train", "val"])
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-m", "--mode", choices=["TRAIN", "TEST", "INFERENCE"], default="TRAIN")
    parser.add_argument("-R", "--REVERSE_SRC_TGT", action="store_true", default=False, help="If passed, will create a translation model for tgt->src instead of src->tgt")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t--{k} = `{v}`")
    return args

if __name__ == "__main__":
    print("######################")
    print("###### train.py ######")
    print("######################")
    args = get_args()
    config = read_config(args.config)
    if args.mode == "TRAIN":
        train_model(config=config, REVERSE_SRC_TGT=args.REVERSE_SRC_TGT)
    elif args.mode == "TEST":
        test_model(config=config, REVERSE_SRC_TGT=args.REVERSE_SRC_TGT)
    elif args.mode == "INFERENCE":
        inference(config=config, REVERSE_SRC_TGT=args.REVERSE_SRC_TGT)
