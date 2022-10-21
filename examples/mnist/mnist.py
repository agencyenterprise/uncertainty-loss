r"""Example of using the uncertainty-loss package with PyTorch Lightning."""
import argparse
import os
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import yaml

from uncertainty_loss import old_code

# force reproduciblity
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

import uncertainty_loss as ul
from uncertainty_loss.old_code import maxnorm_uncertainty_loss

# force reproduciblity
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


DATA_PATH = Path("./data")


class LeNet(torch.nn.Module):
    r"""LeNet convolutional architecture."""

    def __init__(
        self,
        kernel_size: int = 5,
        d_hidden: int = 500,
        dropout: float = 0.3,
        activation: Optional[str] = None,
    ):
        """Initialize a LeNet instance.

        Args:
            kernel_size (int, optional): Size of the convolutional kernels. Defaults to 5.
            d_hidden (int, optional): Size of the hidden layer. Defaults to 500.
            activation (Callable, optional): Activation function. Defaults to F.relu.
            dropout (float, optional): Dropout rate. Defaults to .3.
            classifier_activation (Callable, optional): Activation function
                for the classifier. Defaults to None.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=kernel_size)
        self.max_pool = nn.MaxPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(20000, d_hidden)
        self.fc2 = nn.Linear(d_hidden, 10)
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)
        self.classifier_activation = self.get_activation(activation)

    def forward(self, x: torch.Tensor):
        """Run a forward pass of the model."""
        x = self.activation(self.max_pool(self.conv1(x)))
        x = self.activation(self.max_pool(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        if self.classifier_activation is not None:
            x = self.classifier_activation(x)
        return x

    @staticmethod
    def get_activation(activation: Optional[str] = None):
        if activation is None:
            return None
        if activation == "exp":
            return lambda x: torch.exp(torch.clamp(x, max=20))
        if hasattr(torch, activation):
            return getattr(torch, activation)
        elif hasattr(F, activation):
            return getattr(F, activation)
        elif hasattr(torch.special, activation):
            return getattr(torch.special, activation)
        else:
            raise ValueError(f"Activation function {activation} not found.")


class MNISTExperiment(pl.LightningModule):
    """A base class for running an experiment"""

    def __init__(
        self,
        kernel_size: int = 5,
        d_hidden: int = 500,
        dropout: float = 0.3,
        activation: Optional[str] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        """Initialize a MNistExperiment instance.

        Args:
            kernel_size (int, optional): Size of the convolutional kernels.
                Defaults to 5.
            d_hidden (int, optional): Size of the hidden layer. Defaults to 500.
            dropout (float, optional): Dropout rate. Defaults to .3.
            activation (str, optional): Activation function for the final layer
                of the networks. Defaults to None in which raw logits are returned.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay. Defaults to 1e-5.


        """
        super().__init__()
        self.save_hyperparameters(ignore=list(kwargs.keys()))
        self.model = LeNet(kernel_size, d_hidden, dropout, activation)
        self.lr = lr
        self.weight_decay = weight_decay

    @abstractmethod
    def criterion(self):
        pass

    @abstractmethod
    def uncertainty(self, y_hat):
        pass

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=5, verbose=True, factor=0.2, min_lr=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": lr_scheduler, "monitor": "loss/val"}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MNISTBaseModel")
        parser.add_argument("--kernel_size", type=int, default=5)
        parser.add_argument("--d_hidden", type=int, default=500)
        parser.add_argument("--dropout", type=float, default=0.3)
        parser.add_argument("--activation", type=str, default=None)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        return parent_parser

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        """Log the model architecture."""
        mlflow_logger = self.loggers[1]
        mlflow_logger.experiment.log_text(
            mlflow_logger.run_id, str(self.model), "model.txt"
        )

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, train=True)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, train=False)

    def _step(self, batch, batch_idx, train: bool):
        """A single train or validation step.

        Args:
            batch: A batch of data
            batch_idx: The current batch index in this epoch, unused
            training (bool): If true we are in a train step, else
                we are in a validation step.
        """
        mode = "train" if train else "val"

        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        hit_uncert, miss_uncert = self.hit_miss_uncertainty(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log(f"loss/{mode}", loss)
        self.log(f"hit_uncert/{mode}", hit_uncert)
        self.log(f"miss_uncert/{mode}", miss_uncert)
        self.log(f"accuracy/{mode}", acc)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def hit_miss_uncertainty(self, y_hat, y):
        """Compute the hit and miss uncertainty for a batch.

        Args:
            y_hat: The model predictions
            y: The true labels
        """
        u = self.uncertainty(y_hat)
        y_pred = torch.argmax(y_hat, dim=1)
        hit_uncert = u[y_pred == y].mean()
        miss_uncert = u[y_pred != y].mean()
        return hit_uncert, miss_uncert

    def accuracy(self, y_hat, y):
        """Compute the accuracy for a batch.

        Args:
            y_hat: The model predictions
            y: The true labels
        """
        y_hat = torch.argmax(y_hat, dim=1)
        return (y_hat == y).float().mean()


class MNISTCrossEntropy(MNISTExperiment):
    def on_train_start(self):
        super().on_train_start()
        mlflow_logger = self.loggers[1]
        mlflow_logger.log_hyperparams({"criterion": "cross-entropy"})

    def criterion(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def uncertainty(self, y_hat):
        y_proba = torch.softmax(y_hat, dim=1)
        return ul.entropy(y_proba, normalize=True)


class MNISTUncertainty(MNISTExperiment):
    def __init__(
        self, reg_steps: int, reg_factor: float = 0.0, max_reg: float = 1.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=list(kwargs.keys()))
        self.reg_steps = reg_steps
        self.reg_factor = reg_factor
        self.max_reg = max_reg
        self.reg_step_size = 1 / self.reg_steps

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, train=True)
        self.reg_factor = min(self.reg_factor + self.reg_step_size, self.max_reg)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = MNISTExperiment.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("MNISTUncertaintyModel")
        parser.add_argument("--reg_steps", type=int, default=500)
        parser.add_argument("--reg_factor", type=float, default=0.0)
        parser.add_argument("--max_reg", type=float, default=1.0)
        return parent_parser


class MNISTEDL(MNISTUncertainty):
    def on_train_start(self):
        super().on_train_start()
        mlflow_logger = self.loggers[1]
        mlflow_logger.log_hyperparams({"criterion": "evidential-deep-learning"})

    def criterion(self, y_hat, y):
        return ul.evidential_loss(y_hat, y, self.reg_factor)

    def uncertainty(self, y_hat):
        return ul.uncertainty(y_hat)


class MNISTMaxNorm(MNISTUncertainty):
    def on_train_start(self):
        """Log the model architecture."""
        super().on_train_start()
        mlflow_logger = self.loggers[1]
        mlflow_logger.log_hyperparams({"criterion": "max-norm"})

    def criterion(self, y_hat, y):
        return ul.maxnorm_loss(y_hat, y, self.reg_factor)

    def uncertainty(self, y_hat):
        return ul.uncertainty(y_hat)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = str(DATA_PATH),
        seed=0,
        batch_size=1000,
        num_workers=0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=list(kwargs.keys()))
        self.data_dir = data_dir
        self.transform = transforms.ToTensor()
        self.seed = seed
        self.generator = torch.Generator().manual_seed(self.seed)
        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        # seed is added to the parent parser in main
        parser = parent_parser.add_argument_group("MNistDataModule")
        parser.add_argument("--data-dir", type=str, default=str(DATA_PATH))
        parser.add_argument("--batch-size", type=int, default=1000)
        parser.add_argument("--num-workers", type=int, default=0)
        return parent_parser

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [50000, 10000], generator=self.generator
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)


def get_artifact_uri(mlflow_logger):
    """Get the artifact uri from the mlflow logger."""
    run_id = mlflow_logger.run_id
    artifact_uri = mlflow_logger.experiment.get_run(run_id).info.artifact_uri
    return artifact_uri


def get_loggers(args):
    """Get the loggers for the experiment."""
    if args.experiment_name is None:
        exp_name = f"mnist-uncertainty-experiment"
    else:
        exp_name = args.experiment_name
    log_dir = Path(args.log_dir)
    mlflow_logger = pl.loggers.mlflow.MLFlowLogger(
        experiment_name=exp_name,
        save_dir=log_dir / "mlflow",
    )

    artifact_uri = get_artifact_uri(mlflow_logger)

    if artifact_uri.startswith("file:"):
        artifact_uri = artifact_uri[5:]

    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(
        artifact_uri,
        name="",
        version="",
    )
    return tb_logger, mlflow_logger


def get_callbacks(log_dir, early_stop=False, patience=10):
    callbacks = [
        pl.callbacks.ModelCheckpoint(log_dir, monitor="loss/val", mode="min"),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]
    if early_stop:
        callbacks.append(
            pl.callbacks.EarlyStopping(monitor="loss/val", patience=patience)
        )
    return callbacks


def save_args(args, path):
    with open(path, "w") as f:
        yaml.dump(vars(args), f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-name", type=str, default="mnist-uncertainty")
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=10)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = MNISTDataModule.add_model_specific_args(parser)  # adds seed

    # add model specific args
    names = ["cross-entropy", "edl", "maxnorm"]
    models = [MNISTCrossEntropy, MNISTEDL, MNISTMaxNorm]
    parser.add_argument("--model", type=str, default="cross-entropy")
    temp_args, _ = parser.parse_known_args()

    for name, _model_cls in zip(names, models):
        if temp_args.model == name:
            parser = _model_cls.add_model_specific_args(parser)
            model_cls = _model_cls

    args = parser.parse_args()
    dict_args = vars(args)

    loggers = get_loggers(args)
    log_dir = Path(loggers[0].log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    callbacks = get_callbacks(
        log_dir / "checkpoints", early_stop=args.early_stop, patience=args.patience
    )
    save_args(args, log_dir / "args.yaml")

    torch.manual_seed(args.seed)
    model = model_cls(**dict_args)
    dm = MNISTDataModule(**dict_args)
    trainer = pl.Trainer.from_argparse_args(args, logger=loggers, callbacks=callbacks)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
