from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass, field
from random import shuffle
from typing import TYPE_CHECKING, Any, Mapping, Optional, cast

import torch
from torch.utils.data import DataLoader, Subset, random_split
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

from rich.progress import Progress
from rich.console import Group
from rich.live import Live
from rich.layout import Layout
from rich.status import Status

from shared.abc import Inferable
from shared.traindata import DataMmres
import shared.graph as graph
import shared.env
from utils.baroque import BaroqueProgress

from . import data, model, metrics

from shared.console import console
from shared import env

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


@dataclass
class PointerOption:
    randomize_dataset: bool = field(kw_only=True)
    split_dataset_ratio: tuple[float, float, float] = field(
        kw_only=True, default=(0.8, 0.1, 0.1)
    )


@dataclass
class PointerTrainOption:
    batch_size: int = field(kw_only=True)
    learning_rate: float = field(kw_only=True)
    optim_eps: float = field(kw_only=True, default=1e-8)
    weight_decay: float = field(kw_only=True)
    epoch: int = field(kw_only=True)
    warmup_step: int = field(kw_only=True)
    shuffle: bool = field(kw_only=True, default=True)


class Pointer:
    option: PointerOption

    model: model.ModelPointer

    tokenizer: PreTrainedTokenizerFast

    dataset: data.DatasetPointer
    dataset_train: Subset[train.pointer.DataMmres.DataMmres]
    dataset_valid: Subset[train.pointer.DataMmres.DataMmres]
    dataset_test: Subset[train.pointer.DataMmres.DataMmres]

    def __init__(
        self, option: PointerOption, state_dict: Optional[Mapping[str, Any]] = None
    ):
        self.option = option

        with Status("Initializing pretrained weights...", console=console) as status:
            self.embedder = model.ModelPointerEmbedderPooling()

        with Status("Initializing pretrained tokenizer...") as status:
            self.tokenizer = self.__init_tokenizer()

        with Status("Initializing model...", console=console) as status:
            self.model = self.__init_model(option, self.embedder)
            if state_dict is not None:
                self.model.load_state_dict(state_dict)

        self.device = torch.device("cpu")
        self.device_subsidary = torch.device("cpu")

    def __init_tokenizer(self) -> PreTrainedTokenizerFast:
        return AutoTokenizer.from_pretrained(shared.env.PRETRAINED_MODEL_POINTER)  # type: ignore

    def __init_dataset(self, option: PointerOption):
        self.dataset = data.DatasetPointer(
            {
                "max_graph_size": 16,
                "min_graph_size": 3,
                "seed": None,
            },
            self.tokenizer,
        )

        dataset_split_generator = (
            torch.Generator()
            if option.randomize_dataset
            else torch.Generator().manual_seed(42)
        )

        len_dataset = len(self.dataset)
        split_count = [int(len_dataset * r) for r in option.split_dataset_ratio]
        split_count[-1] += len_dataset - sum(split_count)

        self.dataset_train, self.dataset_valid, self.dataset_test = random_split(
            self.dataset, split_count, dataset_split_generator
        )

    def __init_model(
        self,
        option: PointerOption,
        embedder: model.ModelPointerEmbedder,
    ):
        layer_option = model.ModelOptionPointerLayer(
            feature=embedder.get_feature_size(),
            attention_head=3,
            ffn_dim=2048,
        )
        layer_count = 3

        model_option = model.ModelPointerOption(
            embedder=embedder,
            encoder_layer=lambda: model.ModelPointerEncoderLayer(layer_option),
            encoder_layer_count=layer_count,
            decoder_layer=lambda: model.ModelPointerDecoderLayer(layer_option),
            decoder_layer_count=layer_count,
            # transform_adj_mat=graph.TransformOriginal(threshold=option.threshold),
        )

        return model.ModelPointer(model_option)

    def get_optimizer(self, train_option: PointerTrainOption):
        NO_WEIGHT_DECAY = {"bias", "LayerNorm.weight"}
        optimizer = torch.optim.AdamW(
            # [
            #     {
            #         "params": [
            #             p
            #             for name, p in self.model.named_parameters()
            #             if name not in NO_WEIGHT_DECAY
            #         ],
            #         "weight_decay": train_option.weight_decay,
            #     },
            #     {
            #         "params": [
            #             p
            #             for name, p in self.model.named_parameters()
            #             if name in NO_WEIGHT_DECAY
            #         ],
            #         "weight_decay": 0.0,
            #     },
            # ],
            self.model.parameters(),
            lr=train_option.learning_rate,
            eps=train_option.optim_eps,
        )

        return optimizer

    def train(self, train_option: PointerTrainOption):
        with Status("Initializing dataset...", console=console) as status:
            self.__init_dataset(self.option)

        dataloader_train = cast(
            DataLoader[data.DataBatchPointer],
            DataLoader(
                self.dataset_train,
                batch_size=train_option.batch_size,
                shuffle=train_option.shuffle,
                collate_fn=data.DataBatchPointer.collate,
            ),
        )
        optimizer = self.get_optimizer(train_option)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_option.warmup_step,
            num_training_steps=len(dataloader_train) * train_option.epoch,
        )

        self.model: model.ModelPointer
        dataloader_train: DataLoader[data.DataBatchPointer]

        self.set_device()

        with BaroqueProgress() as (live, progress, status):
            task_train = progress.add_task("Training...", total=train_option.epoch)

            for epoch in range(train_option.epoch):
                progress.update(task_train, description=f"Training, Epoch {epoch}")
                task_epoch = progress.add_task("", total=len(dataloader_train))

                metrics_train = metrics.MetricsPointer()
                self.model.train()

                for idx, batch in enumerate(dataloader_train):
                    batch: data.DataBatchPointer

                    progress.update(task_epoch, description=f"Batch {idx}")
                    status.update(
                        f"loss {metrics_train.loss:.3f}, accuracy {metrics_train.accuracy():.3%}, precision {metrics_train.precision():.3%}, recall {metrics_train.recall():.3%}"
                    )

                    optimizer.zero_grad(set_to_none=True)
                    batch.move(self.device, self.device_subsidary)
                    prediction = self.model(batch)

                    loss = metrics_train.accept(
                        prediction=prediction, target=batch.target, batch=batch
                    )
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    progress.advance(task_epoch)

                progress.advance(task_train)
                console.log(
                    f"epoch {epoch}, train: loss {metrics_train.loss:.3f}, accuracy {metrics_train.accuracy():.3%}, precision {metrics_train.precision():.3%}, recall {metrics_train.recall():.3%}"
                )

                metrics_valid = self.eval(
                    self.dataset_valid, progress=progress, status=status
                )
                console.log(
                    f"epoch {epoch}, valid: loss {metrics_valid.loss:.3f}, accuracy {metrics_valid.accuracy():.3%}, precision {metrics_valid.precision():.3%}, recall {metrics_valid.recall():.3%}"
                )

        self.model = self.model.to("cpu")

        from datetime import datetime

        timestamp = datetime.now().strftime("%m%dT%H%M%S")
        torch.save(self.model.state_dict(), f"models/weights-{timestamp}.pth")

    def eval(
        self,
        dataset_eval,
        *,
        progress: Optional[Progress] = None,
        status: Optional[Status] = None,
    ):
        self.model.eval()
        dataloader_eval = DataLoader(
            dataset_eval,
            batch_size=16,
            shuffle=False,
            collate_fn=data.DataBatchPointer.collate,
        )
        dataloader_eval: DataLoader[DataMmres]

        task_eval = (
            progress.add_task("Evaling...", total=len(dataloader_eval))
            if progress
            else None
        )

        metrics_eval = metrics.MetricsPointer()

        with torch.no_grad():
            for idx, batch in enumerate(dataloader_eval):
                if (
                    progress is not None
                    and task_eval is not None
                    and status is not None
                ):
                    progress.update(task_eval, description=f"Evaling, Batch {idx}")
                    status.update(
                        f"loss {metrics_eval.loss:.3f}, accuracy {metrics_eval.accuracy():.3%}, precision {metrics_eval.precision():.3%}, recall {metrics_eval.recall():.3%}"
                    )

                batch.move(self.device, self.device_subsidary)
                prediction = self.model(batch)

                _ = metrics_eval.accept(
                    prediction=prediction, target=batch.target, batch=batch
                )
                if progress is not None and task_eval is not None:
                    progress.advance(task_eval)

        if progress is not None and task_eval is not None:
            progress.remove_task(task_eval)

        return metrics_eval

    def test(self):
        with Status("Initializing dataset...", console=console) as status:
            self.__init_dataset(self.option)

        self.set_device()

        with BaroqueProgress() as (live, progress, status):
            metrics_eval = self.eval(
                self.dataset_test, progress=progress, status=status
            )

        console.log(
            f"test: loss {metrics_eval.loss:.3f}, accuracy {metrics_eval.accuracy():.3%}, precision {metrics_eval.precision():.3%}, recall {metrics_eval.recall():.3%}"
        )

    def set_device(self):
        with Status("Preparing device...", console=console) as status:
            self.device: torch.device = (
                torch.device(env.CUDA_DEVICE)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.model.to(self.device)
            self.device_subsidary = (
                torch.device("cpu")
                if not torch.cuda.is_available()
                else torch.device(env.CUDA_DEVICE_SUBSIDARY)
                if torch.cuda.device_count() > 1
                else torch.device(env.CUDA_DEVICE_SUBSIDARY)
            )
            self.embedder.to(self.device_subsidary)
            self.embedder.set_main_device(self.device)
