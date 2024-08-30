import datetime
import os
import os.path
import torch.utils.data

import lightning
import datasets

import arc.dataset
import arc.transformer
import arc.transform


class ARCEncoderLightning(lightning.LightningModule):
    def __init__(self, encoder: arc.transformer.ARCEncoder, alpha: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.alpha = torch.tensor(alpha)
        self.save_hyperparameters(ignore=["encoder"])

    def compute_losses(self, batch, batch_index) -> dict[str, torch.Tensor]:
        enable_problem_type_head = self.encoder.config.num_problem_types > 0
        outputs = self.encoder(
            input_ids=batch["input_ids"],
            padding_mask=batch["padding_mask"],
            return_lm_logits=True,
            return_problem_type_logits=enable_problem_type_head,
        )
        shifted_lm_logits = outputs["lm_logits"][:, 0:-1, :].contiguous()
        shifted_lm_labels = batch["input_ids"][:, 1:].contiguous()
        lm_loss = torch.nn.functional.cross_entropy(
            shifted_lm_logits.view(-1, shifted_lm_logits.shape[-1]),
            shifted_lm_labels.view(-1),
            ignore_index=self.encoder.config.padding_index,
            reduction="mean",
        )

        problem_type_loss = None
        if enable_problem_type_head:
            problem_type_loss = torch.nn.functional.cross_entropy(
                outputs["problem_type_logits"],
                batch["problem_type"],
                ignore_index=-100,
            )

        return lm_loss, problem_type_loss

    def training_step(self, batch, batch_index):
        lm_loss, problem_type_loss = self.compute_losses(batch, batch_index)
        loss = self.alpha * lm_loss
        if problem_type_loss:
            loss += (1.0 - self.alpha) * problem_type_loss
            self.log("train_pt_loss", problem_type_loss, prog_bar=True)
        self.log("train_lm_loss", lm_loss, prog_bar=True)
        self.log("loss", loss, prog_bar=True)
        self.log("global_step", self.global_step)

        return loss

    def validation_step(self, batch, batch_index):
        lm_loss, problem_type_loss = self.compute_losses(batch, batch_index)
        self.log("val_lm_loss", lm_loss, prog_bar=True)
        if problem_type_loss:
            self.log("val_pt_loss", problem_type_loss, prog_bar=True)

    def test_step(self, batch, batch_index):
        lm_loss, problem_type_loss = self.compute_losses(batch, batch_index)
        self.log("test_lm_loss", lm_loss)
        if problem_type_loss:
            self.log("test_pt_loss", problem_type_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


def main():
    torch.set_float32_matmul_precision("high")
    tokenizer_max_run_length = 30
    tokenizer = arc.tokenizer.ARCTokenizer(
        max_run_length=tokenizer_max_run_length
    )
    experiments_path = os.path.join(
        "./experiments", datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    )
    checkpoints_path = os.path.join(experiments_path, "checkpoints")
    logs_path = os.path.join(experiments_path, "logs")
    os.makedirs(checkpoints_path)
    os.makedirs(logs_path)
    config = arc.transformer.ARCEncoderConfig(
        num_token_ids=tokenizer.num_token_ids,
        embedding_dim=512,
        num_heads=8,
        dim_feedforward=2048,
        num_layers=4,
        tokenizer_max_run_length=tokenizer_max_run_length,
    )
    create_checkpoint = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=checkpoints_path,
        filename="arc_transformer_{epoch:02d}-{global_step}",
        monitor="global_step",
        save_last="link",
        save_top_k=5,
        mode="max",
        auto_insert_metric_name=False,
        every_n_train_steps=10_000,
        enable_version_counter=True,
    )

    dataset = datasets.load_dataset(
        "./data/augmented/", keep_in_memory=True
    ).with_format("torch")
    dataset = dataset.class_encode_column("problem")
    dataset = dataset["train"].train_test_split(
        test_size=0.25,
        stratify_by_column="problem",
        seed=432995820,
        keep_in_memory=True,
    )
    dataset = dataset.flatten_indices(keep_in_memory=True)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=dataset["train"],
        batch_size=1,
        shuffle=True,
        collate_fn=arc.dataset.ARCCollator(
            pad=tokenizer.special_tokens["<pad>"]
        ),
        pin_memory=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=dataset["test"],
        batch_size=1,
        shuffle=False,
        collate_fn=arc.dataset.ARCCollator(
            pad=tokenizer.special_tokens["<pad>"]
        ),
        pin_memory=True,
    )
    model = ARCEncoderLightning(arc.transformer.ARCEncoder(config))
    trainer = lightning.Trainer(
        max_epochs=10,
        callbacks=[create_checkpoint],
        default_root_dir=logs_path,
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
    )
    trainer.fit(model=model, train_dataloaders=train_dataloader)
    trainer.test(model=model, dataloaders=train_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
