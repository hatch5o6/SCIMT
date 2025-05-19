import argparse
import os
import yaml
from transformers import BartForConditionalGeneration, BartConfig
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger

from parallel_datasets import MultilingualDataset
from spm_tokenizers import SPMTokenizer
from LightningBART import LBART

def train_model(config):
    print("train_model")
    print("TRAINING CONFIG:")
    for k, v in config.items():
        print(f"\t-{k}: {v}")

    L.seed_everything(config["seed"], workers=True)

    save_dir = os.path.join(config["save"] + f"_TRIAL_s={config['seed']}")
    print("SAVE_DIR", save_dir)
    if os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} already exists!")
    else:
        print("Creating", save_dir)
        os.mkdir(save_dir)
        checkpoints_dir = os.path.join(save_dir, "checkpoints")
        print("Creating", checkpoints_dir)
        os.mkdir(checkpoints_dir)
        logs_dir = os.path.join(save_dir, "logs")
        print("Creating", logs_dir)
        os.mkdir(logs_dir)

    dataloaders = get_multilingual_dataloaders(config, sections=["train", "val"])
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]
    src_tokenizer, tgt_tokenizer = get_spm_tokenizers(config)
    model = get_bart_model(src_tokenizer, tgt_tokenizer)

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

    trainer.fit(
        lightning_model,
        train_dataloader,
        val_dataloader
    )

def get_multilingual_dataloaders(config, sections=["train", "val", "test", "inference"]):
    sections = sorted(list(set(sections)))
    for section in sections:
        if section not in ["train", "val", "test", "inference"]:
            raise ValueError(f"Dataloader sections must be in ['train', 'val', 'test', 'inference']")

    dataloaders = {}
    for section in sections:
        a_dataset = MultilingualDataset(
            data_csv=config[f"{section}_data"],
            append_src_lang_tok=config["append_src_token"],
            append_tgt_lang_tok=config["append_tgt_token"]
        )
        a_dataloader = DataLoader(
            a_dataset, 
            batch_size=config[f"{section}_batch_size"],
            shuffle=True
        )
        dataloaders[section] = a_dataloader

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

def get_bart_model(src_tokenizer, tgt_tokenizer):
    model_vocab_size = max(src_tokenizer.vocab_size, tgt_tokenizer.vocab_size)
    print("MODEL VOCAB SIZE:", model_vocab_size)
    
    # TODO Set all BartConfigs :)
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


def get_predictions(dataloader, config):
    #TODO
    pass

def test_model(config):
    print("test_model")
    print("config:")
    for k, v in config.items():
        print(f"{k}: {v}")

    L.seed_everything(config["seed"], workers=True)

    # TODO

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
    parser.add_argument("--config")
    parser.add_argument("--mode", choices=["TRAIN", "TEST", "INFERENCE"], default="TRAIN")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t--{k} = '{v}'")
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
