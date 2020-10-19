from pathlib import Path
import torch

from utils import LJSpeechDataModule
from utils import JasperRecognizer
from utils import Trainer


def main(args):
    device = args['device']
    learning_rate = args['learning_rate']
    max_epoch = args['max_epoch']

    model = JasperRecognizer(
        b=10,
        r=5,
        device=device,
        lr=learning_rate,
    ).to(device)

    datamodule = LJSpeechDataModule(
        data_dir=Path("data/LJSpeech-1.1"),
        batch_size=16,
        num_workers=4,
    )

    trainer = Trainer(
        logger=None,
        max_epoch=max_epoch,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    args = dict(
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        learning_rate=3e-4,
        max_epoch=1,
    )

    main(args)
