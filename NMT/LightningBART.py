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
        if batch_idx % 500 == 0:
            print(f"\n########## train batch {batch_idx} ##########")
            for s, srcseg in enumerate(src_segments):
                tgtseg = tgt_segments[s]

                srctoks = self.src_tokenizer.tokenize(srcseg)
                tgttoks = self.tgt_tokenizer.tokenize(tgtseg)

                print("----------------")
                print(f"{s} - SRC) '{srcseg}'")
                print(f"{s} - SRC TOKS)", srctoks)
                print(f"{s} - TGT) '{tgtseg}'")
                print(f"{s} - TGT TOKS)", tgttoks)
                if s == 4:
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

        print(f"\n########## val batch {batch_idx} ##########")
        for s, srcseg in enumerate(src_segments):
            tgtseg = tgt_segments[s]

            srctoks = self.src_tokenizer.tokenize(srcseg)
            tgttoks = self.tgt_tokenizer.tokenize(tgtseg)

            print("----------------")
            print(f"{s} - SRC) '{srcseg}'")
            print(f"{s} - SRC TOKS)", srctoks)
            print(f"{s} - TGT) '{tgtseg}'")
            print(f"{s} - TGT TOKS)", tgttoks)
            if s == 4:
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
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), 
            lr=self.config["learning_rate"]
        )
        return optimizer