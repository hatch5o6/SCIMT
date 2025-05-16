from torch import optim
import lightning as L

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

    def training_step(self, batch, batch_idx):
        src_segments, tgt_segments = batch
        
        src_segments = self.src_tokenizer.tokenize(
            src_segments,
            return_tensor=True
        )
        src_segments = src_segments.to(self.config["device"])

        tgt_segments = self.tgt_tokenizer.tokenize(
            tgt_segments,
            return_tensor=True
        )
        tgt_segments = tgt_segments.to(self.config["device"])

        loss = self.model(
            input_ids=src_segments,
            labels=tgt_segments
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
        
        src_segments = self.src_tokenizer.tokenize(
            src_segments,
            return_tensor=True
        )
        src_segments = src_segments.to(self.config["device"])

        tgt_segments = self.tgt_tokenizer.char_batch_tokenize(
            tgt_segments,
            return_tensor=True
        )
        tgt_segments = tgt_segments.to(self.config["device"])

        loss = self.model(
            input_ids=src_segments,
            labels=tgt_segments
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
        
        src_segments = self.src_tokenizer.tokenize(
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

        prediction = self.tgt_tokenizer.detokenize(generated_ids, remove_special_toks=self.config["remove_special_toks"])
        results = list(zip(generated_ids, src_segments_text, prediction, tgt_segments_text))
        return results
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), 
            lr=self.config["learning_rate"]
        )
        return optimizer