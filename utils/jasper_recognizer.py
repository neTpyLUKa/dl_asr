import torch
from torch.nn import (
    BatchNorm1d,
    Conv1d,
    CTCLoss,
    Dropout,
    Module,
    ReLU,
    Sequential,
)
from torch.optim import AdamW
from torchaudio.transforms import MelSpectrogram

from .metrics import *


class BlockPiece(Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_width,
            dropout_p,
            stride=1,
            dilation=1,
    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_width = kernel_width
        self._dropout_p = dropout_p
        self._stride = stride
        self._dilation = dilation

        conv = Conv1d(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            kernel_size=self._kernel_width,
            stride=self._stride,
            dilation=self._dilation,
        )
        batch = BatchNorm1d(num_features=self._out_channels)
        relu = ReLU(inplace=True)
        dropout = Dropout(p=self._dropout_p, inplace=True)

        self.first_half = Sequential(
            conv,
            batch,
        )
        self.second_half = Sequential(
            relu,
            dropout,
        )

    def forward(self, x):
        x_1 = self._first_half(x)
        x_2 = self._second_half(x_1)

        return x_2


class Block(Module):
    def __init__(
            self,
            r,
            in_channels,
            out_channels,
            kernel_width,
            dropout_p=0,
    ):
        super().__init__()
        self._r = r
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_width = kernel_width
        self._dropout_p = dropout_p
        self._blocks_pieces_list = []

        for i in range(self._r):
            self._blocks_pieces_list.append(
                BlockPiece(
                    in_channels=self._in_channels,
                    out_channels=self._out_channels,
                    kernel_width=self._kernel_width,
                    dropout_p=self._dropout_p,
                )
            )

        self._blocks_pieces = Sequential(*self._blocks_pieces_list)
        self._last_piece = BlockPiece(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            kernel_width=self._kernel_width,
            dropout_p=self._dropout_p,
        )

    def forward(self, x):
        x_1 = self.subblocks(x)
        x_2 = self.residual_connection(x)
        x_3 = self._last_piece.first_half(x_1)
        x_4 = x_2 + x_3
        x_5 = self._last_piece.second_half(x_4)

        return x_5


class JasperRecognizer(Module):
    def __init__(
            self,
            b=16,
            r=4,
            in_channels=128,
            out_channels=256,
            device=torch.device('cpu'),
            lr=3e-4,
    ):
        super().__init__()
        self._device = device
        self._lr = lr
        self._criterion = CTCLoss().to(self._device)
        self._mel_spectrogrammer = MelSpectrogram(
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=0,
            f_max=8000,
            n_mels=80,
        ).to(self._device)

        self._b = b
        self._r = r
        self._in_channels = in_channels
        self._out_channels = out_channels

        in_channels_list = [256, 256, 384, 512, 640]
        out_channels_list = [256, 384, 514, 640, 768]
        kernel_widths_list = [11, 13, 17, 21, 25]
        dropouts_list = [0.2, 0.2, 0.2, 0.3, 0.3]
        self._blocks_list = []

        for i in range(5):
            self._blocks_list.append(
                Block(
                    r=self._r,
                    in_channels=in_channels_list[i],
                    out_channels=out_channels_list[i],
                    kernel_width=kernel_widths_list[i],
                    dropout_p=dropouts_list[i],
                )
            )
            self._blocks_list.append(
                Block(
                    r=self._r,
                    in_channels=out_channels_list[i],
                    out_channels=out_channels_list[i],
                    kernel_width=kernel_widths_list[i],
                    dropout_p=dropouts_list[i],
                )
            )

        self._begin = BlockPiece(
            in_channels=self._in_channels,
            out_channels=256,
            kernel_width=11,
            dropout_p=0.2,
            stride=2,
        )
        self._blocks = Sequential(*self._blocks_list)
        self._end = Sequential(
            BlockPiece(
                in_channels=768,
                out_channels=896,
                kernel_width=29,
                dropout_p=0.4,
                dilation=2,
            ),
            BlockPiece(
                in_channels=896,
                out_channels=1024,
                kernel_width=1,
                dropout_p=0.4,
            ),
            BlockPiece(
                in_channels=1024,
                out_channels=self._out_channels,
                kernel_width=1,
                dropout_p=0,
            ),
        )

    def forward(self, x):
        x_1 = self._begin(x)
        x_2 = self._blocks(x_1)
        x_3 = self._end(x_2)

        return x_3

    def training_step(self, batch):
        waveforms, targets, waveform_lengths, target_lengths = batch
        waveforms = waveforms.to(self._device)
        targets = targets.to(self._device)
        mel_spectrograms = self.mel_spectrogrammer(waveforms)

        predictions = self(mel_spectrograms)
        loss = self._criterion(
            log_probs=predictions,
            targets=targets,
            input_lengths=waveform_lengths,
            target_lengths=target_lengths,
        )
        log_probs = torch.nn.functional.log_softmax(predictions)
        answers = beam_search(log_probs)
        cer = calculate_cer(answers, targets)
        wer = calculate_wer(answers, targets)

        return loss, cer, wer

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.parameters(),
            lr=self._lr,
        )

        return optimizer


def beam_search(log_probs):
    pass
