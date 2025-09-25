import argparse
import os
import shutil
import yaml
from transformers import BartForConditionalGeneration, BartConfig
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger
import json

from parallel_datasets import MultilingualDataset
from spm_tokenizers import SPMTokenizer
from LightningBART import LBART
from evaluate import calc_bleu, calc_chrF

SAVE_DIR_SUBDIRS = ["checkpoints", "logs"]

def train_model(config):
    print("train_model")
    print("TRAINING CONFIG:")
    for k, v in config.items():
        print(f"\t-{k}: {v}")

    L.seed_everything(config["seed"], workers=True)

    save_dir = os.path.join(config["save"] + f"_TRIAL_s={config['seed']}")
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    logs_dir = os.path.join(save_dir, "logs")
    print("SAVE_DIR", save_dir)
    if os.path.exists(save_dir):
        for item in os.listdir(save_dir):
            if item not in SAVE_DIR_SUBDIRS:
                raise ValueError(f"Invalid subdir/file '{item}' in {save_dir}!")
            item_dir = os.path.join(save_dir, item)
            if not os.path.isdir(item_dir) or list(os.listdir(item_dir)) != []:
                raise ValueError(f"Subdir/file '{item}' must be an empty directory in order to begin training this model!")
    
    if not os.path.exists(save_dir):
        print("Creating", save_dir)
        os.mkdir(save_dir)
    if not os.path.exists(checkpoints_dir):
        print("Creating", checkpoints_dir)
        os.mkdir(checkpoints_dir)
    if not os.path.exists(logs_dir):
        print("Creating", logs_dir)
        os.mkdir(logs_dir)

    dataloaders = get_multilingual_dataloaders(config, sections=["train", "val"])
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]

    # print("TESTING TRAIN DATALOADER")
    # examine_dataloader(train_dataloader)
    # print("\n\nTESTING VAL DATALOADER")
    # examine_dataloader(val_dataloader)
    # exit()

    src_tokenizer, tgt_tokenizer = get_spm_tokenizers(config)

    # print("EXAMINING TOKENIZERS ON TRAIN DATALOADER")
    # examine_tokenizers(src_tokenizer, tgt_tokenizer, train_dataloader)
    # exit()

    model = get_bart_model(src_tokenizer, tgt_tokenizer, config)

    if config["from_pretrained"] not in [None, "None"]:

        if not config["from_pretrained"].endswith(".ckpt"):
            pretrain_checkpoints_dir = os.path.join(config["from_pretrained"], "checkpoints")
            assert os.path.exists(pretrain_checkpoints_dir)
            assert os.path.isdir(pretrain_checkpoints_dir)
            print("Choosing pretrained model to fine-tune.")
            chosen_pretrained_checkpoint = choose_checkpoint(pretrain_checkpoints_dir)
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

    trainer = L.Trainer(
        max_epochs=config["max_epochs"],
        val_check_interval=config["val_interval"],
        accelerator=config["device"],
        default_root_dir=save_dir,
        callbacks=train_callbacks,
        logger=logger,
        deterministic=True
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
    print("SRC TOKENIZER:", src_tokenizer.spm_name)
    print("TGT TOKENIZER:", tgt_tokenizer.spm_name)

    printed = 0
    for b, (src_sents, tgt_sents) in enumerate(dataloader):
        if b % 20 == 0:
            batch = list(zip(src_sents, tgt_sents))
            print(f"####### BATCH {b} #######")
            for i, (src_sent, tgt_sent) in enumerate(batch):
                if i % 20 == 0:
                    src_tok_ids, src_toks = src_tokenizer.tokenize(src_sent)
                    tgt_tok_ids, tgt_toks = tgt_tokenizer.tokenize(tgt_sent)
                    print(f"----- ({i}) -----")
                    print("SRC______:", src_sent)
                    print("SRCTOKS__:", src_toks)
                    print("SRCTOKIDS:", src_tok_ids)
                    print("TGT______:", tgt_sent)
                    print("TGTTOKS__:", tgt_toks)
                    print("TGTTOKIDS:", tgt_tok_ids)
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
            print(f"####### BATCH {b} #######")
            for i, (src_sent, tgt_sent) in enumerate(batch):
                if i % 20 == 0:
                    print(f"----- ({i}) -----")
                    print("SRC:", src_sent)
                    print("TGT:", tgt_sent)
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


def test_model(config):
    print("test_model")
    print("config:")
    for k, v in config.items():
        print(f"{k}: {v}")

    L.seed_everything(config["seed"], workers=True)

    save_dir = os.path.join(config["save"] + f"_TRIAL_s={config['seed']}")
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    logs_dir = os.path.join(save_dir, "logs")
    predictions_dir = os.path.join(save_dir, "predictions")
    if not os.path.exists(predictions_dir):
        os.mkdir(predictions_dir)

    for d in [save_dir, checkpoints_dir, logs_dir, predictions_dir]:
        if not os.path.exists(d):
            raise ValueError(f"Expected to find directory {d}!")
    
    dataloaders = get_multilingual_dataloaders(config, sections=["test"])
    test_dataloader = dataloaders["test"]

    # print("TESTING TEST DATALOADER")
    # examine_dataloader(test_dataloader)
    # exit()

    src_tokenizer, tgt_tokenizer = get_spm_tokenizers(config)

    if config["test_checkpoint"] in [None, "None", "null"]:
        assert os.path.exists(checkpoints_dir)
        print("No checkpoint provided for testing. Will choose.")
        config["test_checkpoint"] = choose_checkpoint(checkpoints_dir)
    else:
        print("Testing checkpoint provided:", config["test_checkpoint"])

    results = get_predictions(
        dataloader=test_dataloader,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        config=config
    )

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
    os.mkdir(chkpt_predictions_dir)

    # Predictions
    save_preds = os.path.join(chkpt_predictions_dir, "predictions.txt")
    with open(save_preds, "w") as outf:
        outf.write("\n".join(preds) + "\n")
    
    # Metrics
    bleu_score = calc_bleu(preds, [ref_data])
    chrf_score, chrf_sents = calc_chrF(preds, [ref_data])

    metrics = {
        "BLEU": bleu_score,
        "chrF": chrf_score,
        # "chrF_sents": chrf_sents
    }
    print("METRICS")
    print(metrics)
    save_metrics = os.path.join(chkpt_predictions_dir, "metrics.json")
    with open(save_metrics, "w") as outf:
        outf.write(json.dumps(metrics, indent=2))

    return preds, metrics  

def choose_checkpoint(checkpoints_dir):
    print("Choosing checkpoint with lowest validation loss")
    chosen_val_loss = None
    chosen_f = None
    for f in os.listdir(checkpoints_dir):
        val_loss = float(f.split("-val_loss=")[-1].split(".ckpt")[0])
        if chosen_f == None:
            assert chosen_val_loss == None
            chosen_f = f
            chosen_val_loss = val_loss
        else:
            assert isinstance(chosen_f, str)
            assert isinstance(chosen_val_loss, float)
            if val_loss < chosen_val_loss:
                chosen_f = f
                chosen_val_loss = val_loss
    assert chosen_f != None
    chosen_f_path = os.path.join(checkpoints_dir, chosen_f)
    print("CHOSE CHECKPOINT", chosen_f_path)
    return chosen_f_path

def inference(
    predict_config,
    checkpoint_path,
    name,
    VERBOSE=True,
    div="test"
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


def get_multilingual_dataloaders(config, sections=["train", "val", "test", "inference"], RETURN_DATASETS=False):
    sections = sorted(list(set(sections)))
    for section in sections:
        if section not in ["train", "val", "test", "inference"]:
            raise ValueError(f"Dataloader sections must be in ['train', 'val', 'test', 'inference']")
        if section in ["train", "val"]:
            assert config[f"{section}_data"].endswith(f"/{section}.no_overlap_v1.csv")
        else:
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
            shuffle=SHUFFLE
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
    src_tokenizer = SPMTokenizer(
        spm_name=config["src_spm"]
    )
    tgt_tokenizer = SPMTokenizer(
        spm_name=config["tgt_spm"]
    )
    tokenizer_asserts(src_tokenizer, tgt_tokenizer)

    print("\nSpecial Toks")
    for i, (tok, idx) in enumerate(src_tokenizer.token2idx.items()):
        print(f"{tok}:{idx}")
        print(f"{idx}: {src_tokenizer.idx2token[idx]}")
        print("-------------------------")
        if i == 5:
            break

    return src_tokenizer, tgt_tokenizer

def get_bart_model(src_tokenizer, tgt_tokenizer, config):
    model_vocab_size = max(src_tokenizer.vocab_size, tgt_tokenizer.vocab_size)
    print("MODEL VOCAB SIZE:", model_vocab_size)
    
    # TODO Set all BartConfigs :)
    print("\nget_bart_model() SETTING CONFIGS:")
    for k,v in config.items():
        print(f"{k}:`{v}`")
    print("\n")
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
        print("\n\n##############################")
        print("###### TRAINING STARTED ######")
        print("##############################\n\n")
    def on_train_end(self, trainer, pl_module):
        print("\n\n##############################")
        print("####### TRAINING ENDED #######")
        print("##############################\n\n")

def read_config(f):
    with open(f) as inf:
        config = yaml.safe_load(inf)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-m", "--mode", choices=["TRAIN", "TEST", "INFERENCE"], default="TRAIN")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t--{k} = `{v}`")
    return args

if __name__ == "__main__":
    print("----------------------")
    print("###### train.py ######")
    print("----------------------")
    args = get_args()
    config = read_config(args.config)
    if args.mode == "TRAIN":
        train_model(config=config)
    elif args.mode == "TEST":
        test_model(config=config)
    elif args.mode == "INFERENCE":
        inference(config=config)
