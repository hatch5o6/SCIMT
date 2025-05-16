import argparse
import os
import shutil
import yaml
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartConfig
import torch
from torch import optim
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger

from parallel_datasets import ParallelDataset
from tokenizers import SPMTokenizer
from LightningBART import LBART

def train_model(config):
    print("train_model")
    print("training config:")
    for k, v in config.items():
        print(f"{k}: {v}")

    L.seed_everything(config["seed"], workers=True)

    save_dir = config["save"]
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

    
    train_dataset = ParallelDataset(
        src_file_path=config["train_src_data"],
        tgt_file_path=config["train_tgt_data"]
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["train_batch_size"],
        shuffle=True
    )
    val_dataset = ParallelDataset(
        src_file_path=config["val_src_data"],
        tgt_file_path=config["val_tgt_data"]
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
    )

    # TODO src and tgt or just one? Probs two.
    tokenizer = SPMTokenizer(
        spm_dir=config["tokenizer"]
    )

    # TODO Set all BartConfigs :)
    model_config = BartConfig()
    model_config.vocab_size = 300
    model_config.pad_token_id=tokenizer.token2idx[tokenizer.pad]
    model_config.bos_token_id=tokenizer.token2idx[tokenizer.bos]
    model_config.eos_token_id=tokenizer.token2idx[tokenizer.eos]
    model_config.forced_eos_token_id=tokenizer.token2idx[tokenizer.eos]
    model_config.decoder_start_token_id=tokenizer.token2idx[tokenizer.bos]
    # Model Hyperparameters
    model_config.encoder_layers = config["encoder_layers"]
    model_config.decoder_layers = config["decoder_layers"]
    model_config.encoder_attention_heads = config["encoder_attention_heads"]
    model_config.decoder_attention_heads = config["decoder_attention_heads"]
    model_config.max_position_embeddings = config["max_position_embeddings"]
    model_config.encoder_ffn_dim = config["encoder_ffn_dim"]
    model_config.encoder_layerdrop = config["encoder_layerdrop"]
    model_config.decoder_ffn_dim = config["decoder_ffn_dim"]
    model_config.decoder_layerdrop = config["decoder_layerdrop"]
    model_config.d_model = config["d_model"]
    model_config.dropout = config["dropout"]


    print("\nSpecial Toks")
    for i, (tok, idx) in enumerate(tokenizer.token2idx.items()):
        print(f"{tok}:{idx}")
        print(f"{idx}: {tokenizer.idx2token[idx]}")
        print("-------------------------")
        if i == 5:
            break
    
    print("\nMODEL SPECIAL TOKS (and tokenizer)")
    print(f"BOS: {model_config.bos_token_id}")
    print(f"EOS: {model_config.eos_token_id}")
    print(f"Forced EOS: {model_config.forced_eos_token_id}")
    print(f"Decoder Start Token: {model_config.decoder_start_token_id}")
    print(f"PAD: {model_config.pad_token_id}")

    model = BartForConditionalGeneration(model_config)

    lightning_model = LBART(
        model=model,
        # TODO
        src_tokenizer=tokenizer,
        tgt_tokenizer=tokenizer
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
    # TODO Write this function


    pred_dataset = ParallelDataset(
        src_file_path=config[f"{div}_src_data"],
        tgt_file_path=config[f"{div}_tgt_data"]
    )
    pred_dataloader = DataLoader(
        pred_dataset, 
        batch_size=config["pred_batch_size"]
    )
    tokenizer = CharacterTokenizer(
        vocab_path=predict_config["vocab_path"]
    )

    model_config = BartConfig()
    model_config.vocab_size = 300
    model_config.encoder_layers = config["encoder_layers"]
    model_config.decoder_layers = config["decoder_layers"]
    model_config.encoder_attention_heads = config["encoder_attention_heads"]
    model_config.decoder_attention_heads = config["decoder_attention_heads"]
    model_config.pad_token_id=tokenizer.token2idx[tokenizer.pad]
    model_config.bos_token_id=tokenizer.token2idx[tokenizer.bos]
    model_config.eos_token_id=tokenizer.token2idx[tokenizer.eos]
    model_config.forced_eos_token_id=tokenizer.token2idx[tokenizer.eos]
    model_config.decoder_start_token_id=tokenizer.token2idx[tokenizer.bos]

    model = BartForConditionalGeneration(model_config)
    model.eval()

    print("CONFIG")
    for key, value in config.items():
        print(f"{key}:{value}")

    print("MODEL", type(model), model)
    print("CHKPT", checkpoint_path)

    lightning_model = LightningMSABART.load_from_checkpoint(
        checkpoint_path,
        model=model,
        tokenizer=tokenizer,
        config=config
    )
    lightning_model.eval()
    trainer = L.Trainer(
        accelerator=config["device"],
        devices=config["num_gpus"]
    )
    predictions = trainer.predict(lightning_model, dataloaders=pred_dataloader)
    pred_dir = os.path.join(checkpoint_path.split("checkpoints")[0], "predictions")
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    pred_file = os.path.join(pred_dir, f"{checkpoint_path.split('/')[-1].split('.txt')[0]}_{div}_preds.txt")
    verbose_file = os.path.join(pred_dir, f"{checkpoint_path.split('/')[-1].split('.txt')[0]}_{div}_verbose.txt")
    hyp_sents = []
    ref_sents = []
    with open(pred_file, 'w') as predf, open(verbose_file, 'w') as verbf:
        for b, batch in enumerate(predictions):
            print("BATCH", b, "SIZE", len(batch))
            for i, (generated_ids, src_text, prediction, tgt_text) in enumerate(batch):
                predf.write(prediction.strip() + "\n")
                print(prediction)

                if VERBOSE:
                    verbf.write(f"####################### {b}-{i} ########################\n")
                    verbf.write("SOURCE\n")
                    verbf.write(src_text.strip() + "\n")
                    verbf.write(f"IDS ({len(generated_ids)})\n")
                    verbf.write(f"{generated_ids}\n")
                    verbf.write("PREDICTION\n")
                    verbf.write(prediction.strip() + "\n")
                    verbf.write("TARGET\n")
                    verbf.write(tgt_text.strip() + "\n")
                    hyp_sents.append(prediction)
                    ref_sents.append(tgt_text)
        correct, total, accuracy = calc_accuracy(ref_sents, hyp_sents)
        verbf.write(f"\nCORRECT: {correct}\tTOTAL: {total}\tACCURACY: {accuracy}")
    
    return hyp_sents, correct, total, accuracy




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
