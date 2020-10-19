from pathlib import Path
import torch
import tqdm


class Trainer:
    def __init__(
            self,
            logger,
            max_epoch,
            verbose=False,
    ):
        self._logger = logger
        self._max_epoch = max_epoch
        self._verbose = verbose

    def save_checkpoint(
            self,
            model,
            optimizer,
            epoch: int,
            checkpoints_dir: Path,
    ):
        checkpoint = {
            'model': model,
            'optimizer': optimizer,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }
        checkpoint_path = checkpoints_dir / f"{epoch}.hdf5"
        torch.save(checkpoint, checkpoint_path)

    @torch.enable_grad()
    def training_epoch(
            self,
            model,
            train_dataloader,
            optimizer,
    ):
        model.train()

        for batch_idx, batch in enumerate(tqdm.tqdm(train_dataloader)):
            loss = model.training_step(batch, batch_idx)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def fit(
            self,
            model,
            datamodule,
    ):
        train_dataloader = datamodule.train_dataloader()
        validation_dataloader = datamodule.validation_dataloader()
        optimizer = model.configure_optimizers()

        for epoch in range(self._max_epoch):
            self.training_epoch(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
            )

            self.validation_epoch(
                model=model,
                val_dataloader=validation_dataloader,
            )

            self.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                checkpoints_dir=Path.cwd() / "models",
            )

        return model
