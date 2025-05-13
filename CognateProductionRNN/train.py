import argparse
import yaml
import os
from torch import optim
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger
from evaluate import calc_chrF, calc_bleu, calc_NED, calc_accuracy

from data import *
from model import *

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
        
    # get train dataloader and langs
    input_lang, output_lang, train_dataloader = get_dataloader(
        file_path=config["train"],
        lang1=config["src"],
        lang2=config["tgt"],
        input_lang_dir=config["src_lang"],
        output_lang_dir=config["tgt_lang"],
        reverse=config["reverse"], # if the data is ```lang1 (tgt) ||| lang2 (src) ||| 0.0```, then set reverse=True
        batch_size=config["batch_size"],
        device=config["device"],
        LIMIT=config["LIMIT"],
        shuffle=True
    )

    # get val dataloader
    _, _, val_dataloader = get_dataloader(
        file_path=config["val"],
        lang1=config["src"],
        lang2=config["tgt"],
        input_lang_dir=config["src_lang"],
        output_lang_dir=config["tgt_lang"],
        reverse=config["reverse"], # if the data is ```lang1 (tgt) ||| lang2 (src) ||| 0.0```, then set reverse=True
        batch_size=config["batch_size"],
        device=config["device"],
        shuffle=False
    )

    model = Seq2Seq(
        encoder_layers=config["encoder_layers"],
        decoder_layers=config["decoder_layers"],
        encoder_input_size=input_lang.n_tokens,
        encoder_hidden_size=config["encoder_hidden_size"],
        encoder_dropout=config["encoder_dropout"],
        decoder_output_size=output_lang.n_tokens,
        decoder_hidden_size=config["decoder_hidden_size"],
        decoder_dropout=config["decoder_dropout"],
        device=config["device"],
    )

    lightning_model = LightningRNN(
        model=model,
        input_lang=input_lang,
        output_lang=output_lang,
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
        val_dataloader,
    )


def test_model(config):
    print("train_model")
    print("training config:")
    for k, v in config.items():
        print(f"{k}: {v}")

    L.seed_everything(config["seed"], workers=True)

    # read data
    pairs = readData(
        file_path = config["test"],
        reverse=config["reverse"]
    )
    src_data = [src_seq for src_seq, _ in pairs]
    tgt_data = [tgt_seq for _, tgt_seq in pairs]

    # get test dataloader and langs
    input_lang, output_lang, test_dataloader = get_dataloader(
        file_path=config["test"],
        lang1=config["src"],
        lang2=config["tgt"],
        input_lang_dir=config["src_lang"],
        output_lang_dir=config["tgt_lang"],
        reverse=config["reverse"], # if the data is ```lang1 (tgt) ||| lang2 (src) ||| 0.0```, then set reverse=True
        batch_size=config["batch_size"],
        device=config["device"],
        shuffle=False
    )

    model = Seq2Seq(
        encoder_layers=config["encoder_layers"],
        decoder_layers=config["decoder_layers"],
        encoder_input_size=input_lang.n_tokens,
        encoder_hidden_size=config["encoder_hidden_size"],
        encoder_dropout=config["encoder_dropout"],
        decoder_output_size=output_lang.n_tokens,
        decoder_hidden_size=config["decoder_hidden_size"],
        decoder_dropout=config["decoder_dropout"],
        device=config["device"]
    )

    lightning_model = LightningRNN.load_from_checkpoint(
        checkpoint_path=config["test_checkpoint"],
        model=model,
        input_lang=input_lang,
        output_lang=output_lang,
        config=config
    )
    lightning_model.eval()

    trainer = L.Trainer(
        accelerator=config["device"]
    )

    batch_predictions = trainer.predict(
        lightning_model,
        dataloaders=test_dataloader
    )

    predictions = []
    for b, batch in batch_predictions:
        predictions += batch
    assert len(src_data) == len(tgt_data) == len(predictions)

    bleu_score = calc_bleu(predictions, [tgt_data])
    chrf_score, chrf_sents = calc_chrF(predictions, [tgt_data])
    ned, ned_sents = calc_NED(predictions, tgt_data)
    acc, acc_sents = calc_accuracy(predictions, tgt_data)

    metrics = {
        "BLEU": bleu_score,
        "chrF": chrf_score,
        "chrF_sents": chrf_sents,
        "ned": ned,
        "ned_sents": ned_sents,
        "accuracy": acc,
        "accuracy_sents": acc_sents 
    }

    return predictions, metrics


class LightningRNN(L.LightningModule):
    def __init__(
        self,
        # encoder,
        # decoder,
        model,
        input_lang,
        output_lang,
        config
    ):
        super().__init__()
        # self.encoder = encoder,
        # self.decoder = decoder
        self.model = model
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.config = config
        self.criterion = nn.NLLLoss()

    def training_step(self, batch, batch_idx):
        input_tensor, target_tensor = batch

        # TODO what does this do?
        # encoder_optimizer.zero_grad()
        # decoder_optimizer.zero_grad()

        # encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        # decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, target_tensor)

        decoder_outputs, _, _ = self.model(input_tensor, target_tensor)

        loss = self.criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )

        # TODO optimizers
        # encoder_optimizer.step()
        # decoder_optimizer.step()

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["batch_size"]
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input_tensor, target_tensor = batch

        # TODO what does this do?
        # encoder_optimizer.zero_grad()
        # decoder_optimizer.zero_grad()

        # encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        # decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, target_tensor)

        decoder_outputs, _, _ = self.model(input_tensor, target_tensor)

        loss = self.criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )

        # TODO optimizers
        # encoder_optimizer.step()
        # decoder_optimizer.step()

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["batch_size"]
        )

        return loss

    def test_step(self, batch, batch_idx):
        # with torch.no_grad(): # DO I NEED THIS IN LIGHTING?
        input_tensor, _ = batch
        decoder_outputs, decoder_hidden, decoder_attn = self.model(input_tensor)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(self.output_lang.index2word[idx.item()])

        # TODO calculate metrics
        metrics = None
        
        return decoded_words, decoder_attn, metrics

    def predict_step(self, batch, batch_idx):
        # with torch.no_grad(): # DO I NEED THIS IN LIGHTING?
        input_tensor, _ = batch
        decoder_outputs, decoder_hidden, decoder_attn = self.model(input_tensor)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(self.output_lang.index2word[idx.item()])
        
        return decoded_words, decoder_attn

    def configure_optimizers(self):
        # TODO How to do two optimizers?
        # If I combine them to a seq2seq, then I may only need one, like this
        optimizer = optim.Adam(
            self.parameters(), 
            lr=self.config["learning_rate"]
        )
        return optimizer

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
    parser.add_argument("--TEST", action="store_true")
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
    if args.TEST:
        test_model(config=config)
    else:
        train_model(config=config)
