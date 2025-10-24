from torch import optim
import lightning as L
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.utilities import rank_zero_info

class LBART(L.LightningModule):
    def __init__(
        self,
        model,
        src_tokenizer,
        tgt_tokenizer,
        config
    ):
        super().__init__()
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.config = config

        device = self.config["device"]
        print("DEVICE:", device)
        self.model = self.model.to(device)

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        src_segments, tgt_segments = batch
        VERBOSE = False
        if self.config["verbose"]:
            VERBOSE = True
            freq = 500
            break_after = 4
        elif "little_verbose" in self.config and self.config["little_verbose"]:
            VERBOSE = True
            freq = 10000
            break_after = 2

        if VERBOSE:
            if batch_idx % freq == 0:
                rank_zero_info(f"\n########## train batch {batch_idx} ##########")
                for s, srcseg in enumerate(src_segments):
                    tgtseg = tgt_segments[s]

                    srctok_ids, srctoks = self.src_tokenizer.tokenize(srcseg)
                    tgttok_ids, tgttoks = self.tgt_tokenizer.tokenize(tgtseg)
                    # src_toks = self.src_tokenizer.encode(srcseg).tokens

                    rank_zero_info("----------------")
                    rank_zero_info(f"{s} - SRC) '{srcseg}'")
                    rank_zero_info(f"{s} - SRC TOKS) {srctoks}")
                    rank_zero_info(f"{s} - TGT) '{tgtseg}'")
                    rank_zero_info(f"{s} - TGT TOKS) {tgttoks}")
                    if s == break_after:
                        break
        
        src_segments_tensor = self.src_tokenizer.batch_tokenize(
            src_segments,
            return_tensor=True
        )
        src_segments_tensor = src_segments_tensor.to(self.config["device"])

        tgt_segments_tensor = self.tgt_tokenizer.batch_tokenize(
            tgt_segments,
            return_tensor=True
        )
        tgt_segments_tensor = tgt_segments_tensor.to(self.config["device"])

        loss = self.model(
            input_ids=src_segments_tensor,
            labels=tgt_segments_tensor
        ).loss

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["train_batch_size"]
        )

        return loss

    def validation_step(self, batch, batch_idx):
        src_segments, tgt_segments = batch
        VERBOSE = False
        if self.config["verbose"]:
            VERBOSE = True
            freq = 1
            break_after = 4
        elif "little_verbose" in self.config and self.config["little_verbose"]:
            VERBOSE = True
            freq = 1
            break_after = 1

        if VERBOSE:
            if batch_idx % freq == 0:
                rank_zero_info(f"\n########## val batch {batch_idx} ##########")
                for s, srcseg in enumerate(src_segments):
                    tgtseg = tgt_segments[s]

                    srctok_ids, srctoks = self.src_tokenizer.tokenize(srcseg)
                    tgttok_ids, tgttoks = self.tgt_tokenizer.tokenize(tgtseg)

                    rank_zero_info("----------------")
                    rank_zero_info(f"{s} - SRC) '{srcseg}'")
                    rank_zero_info(f"{s} - SRC TOKS) {srctoks}")
                    rank_zero_info(f"{s} - TGT) '{tgtseg}'")
                    rank_zero_info(f"{s} - TGT TOKS) {tgttoks}")
                    if s == break_after:
                        break
        
        src_segments_tensor = self.src_tokenizer.batch_tokenize(
            src_segments,
            return_tensor=True
        )
        src_segments_tensor = src_segments_tensor.to(self.config["device"])

        tgt_segments_tensor = self.tgt_tokenizer.batch_tokenize(
            tgt_segments,
            return_tensor=True
        )
        tgt_segments_tensor = tgt_segments_tensor.to(self.config["device"])

        loss = self.model(
            input_ids=src_segments_tensor,
            labels=tgt_segments_tensor
        ).loss

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["val_batch_size"]
        )
        return loss
    
    def predict_step(self, batch, batch_idx):
        src_segments_text, tgt_segments_text = batch
        
        src_segments = self.src_tokenizer.batch_tokenize(
            src_segments_text,
            return_tensor=True
        )
        src_segments = src_segments.to(self.config["device"])

        generated_ids = [
            t.tolist()
            for t in self.model.generate(
                input_ids=src_segments, 
                max_length=self.config["max_length"]
            )
        ]

        prediction = self.tgt_tokenizer.batch_detokenize(generated_ids, remove_special_toks=self.config["remove_special_toks"])
        results = list(zip(generated_ids, src_segments_text, prediction, tgt_segments_text))
        return results
    
    # Basic optimizer
    # def configure_optimizers(self):
    #     optimizer = optim.Adam(
    #         self.parameters(), 
    #         lr=float(self.config["learning_rate"])
    #     )
    #     return optimizer

    # With scheduler
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=float(self.config["learning_rate"]),
            weight_decay=self.config["weight_decay"]
        )

        lr_lambda = get_linear_schedule_with_warmup(
            num_warmup_steps=self.config["warmup_steps"],
            num_training_steps=self.config["max_steps"]
        )
        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

def get_linear_schedule_with_warmup(num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),
        )
    return lr_lambda
