import torch
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

    def on_before_optimizer_step(self, optimizer): 
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1e9) 
        if self.trainer.is_global_zero: 
            self.log("grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False, logger=True)

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
        # elif self.global_step >= 26582 and self.global_step <= 26617:
        #     VERBOSE = True
        #     freq = 1
        #     break_after = 1000000

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
                    rank_zero_info(f"{s} - SRC TOKS) ({len(srctoks)}) {srctoks}")
                    rank_zero_info(f"{s} - TGT) '{tgtseg}'")
                    rank_zero_info(f"{s} - TGT TOKS) ({len(tgttoks)}) {tgttoks}")
                    # with open(f"/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/fr-mfe_train_log_rank-{self.global_rank}.txt", "a") as outf:
                    #     outf.write(f"\n\n\n\n########## train batch step {self.global_step} ##########\n")
                    #     outf.write("----------------\n")
                    #     outf.write(f"{s} - SRC) '{srcseg}'\n")
                    #     outf.write(f"{s} - SRC TOKS) ({len(srctoks)}) {srctoks}\n")
                    #     outf.write(f"{s} - TGT) '{tgtseg}'\n")
                    #     outf.write(f"{s} - TGT TOKS) ({len(tgttoks)}) {tgttoks}\n")
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

        #TODO I'm doing (making occurences of pad_token_id (3) to be -100, the number ignored by pytorch in CrossEntropyLoss calculation) because ChatGPT says to. Check with Lawry to make sure it's the right thing to do.
        pad_tok = self.tgt_tokenizer.pad
        pad_tok_id = self.tgt_tokenizer.token2idx[pad_tok]
        assert pad_tok_id == 3
        tgt_segments_tensor[tgt_segments_tensor == pad_tok_id] = -100


        # if self.global_step >= 26570 and self.global_step <= 26617:
        #     with open(f"/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/fr-mfe_train_log_rank-{self.global_rank}.txt", "a") as outf:
        #         outf.write(f"======================== Step {self.global_step} Batch ========================\n")
        #         outf.write(f"SRC: {src_segments_tensor}\n")
        #         outf.write(f"TGT: {tgt_segments_tensor}\n")
        #         outf.write("SEGMENTS:")
        #         for s, src_seg in enumerate(src_segments):
        #             outf.write("***\n")
        #             tgt_seg = tgt_segments[s]
                    
        #             srctok_ids, srctoks = self.src_tokenizer.tokenize(src_seg)
        #             outf.write(f"   srctoks: `{srctoks}`\n")
        #             outf.write(f"srctok_ids: `{srctok_ids}`\n")
        #             tgttok_ids, tgttoks = self.tgt_tokenizer.tokenize(tgt_seg)
        #             outf.write(f"   tgttoks: `{tgttoks}`\n")
        #             outf.write(f"tgttok_ids: `{tgttok_ids}`\n")

        loss = self.model(
            input_ids=src_segments_tensor,
            labels=tgt_segments_tensor
        ).loss

        # self.log(
        #     "train_src_max_seq_len",
        #     src_segments_tensor.shape[1],
        #     on_step=True,
        #     on_epoch=False,
        #     prog_bar=False,
        #     logger=True
        # )

        # self.log(
        #     "train_tgt_max_seq_len",
        #     tgt_segments_tensor.shape[1],
        #     on_step=True,
        #     on_epoch=False,
        #     prog_bar=False,
        #     logger=True
        # )

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

        #TODO I'm doing (making occurences of pad_token_id (3) to be -100, the number ignored by pytorch in CrossEntropyLoss calculation) because ChatGPT says to. Check with Lawry to make sure it's the right thing to do.
        pad_tok = self.tgt_tokenizer.pad
        pad_tok_id = self.tgt_tokenizer.token2idx[pad_tok]
        assert pad_tok_id == 3
        tgt_segments_tensor[tgt_segments_tensor == pad_tok_id] = -100

        loss = self.model(
            input_ids=src_segments_tensor,
            labels=tgt_segments_tensor
        ).loss

        # self.log(
        #     "val_src_max_seq_len",
        #     src_segments_tensor.shape[1],
        #     on_step=True,
        #     on_epoch=False,
        #     prog_bar=False,
        #     logger=True
        # )

        # self.log(
        #     "val_tgt_max_seq_len",
        #     tgt_segments_tensor.shape[1],
        #     on_step=True,
        #     on_epoch=False,
        #     prog_bar=False,
        #     logger=True
        # )

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["val_batch_size"],
            sync_dist=True
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
