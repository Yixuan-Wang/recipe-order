from __future__ import annotations
from collections.abc import Sequence

from dataclasses import field, dataclass
from functools import partial
from itertools import starmap
from sched import scheduler
from typing import Optional, cast

from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch
import transformers
from transformers import (
    get_linear_schedule_with_warmup,
    PreTrainedTokenizerFast,
    AutoTokenizer,
)

from rich.progress import Progress
from rich.console import Group
from rich.live import Live
from rich.layout import Layout
from rich.status import Status

from shared.abc import Inferable
from shared.traindata import DataRawList
from shared.console import console
import shared.env as env
from utils.baroque import BaroqueProgress
from utils.stub import nonnull, tokenize
from utils.functional import regroup

from . import data, model, metrics


@dataclass
class PermOption:
    randomize_dataset: bool = field(kw_only=True)
    split_dataset_ratio: tuple[float, float, float] = field(
        kw_only=True, default=(0.8, 0.1, 0.1)
    )


@dataclass
class PermTrainOption:
    batch_size: int = field(kw_only=True)
    learning_rate: float = field(kw_only=True)
    optim_eps: float = field(kw_only=True, default=1e-8)
    weight_decay: float = field(kw_only=True)
    epoch: int = field(kw_only=True)
    warmup_step: int = field(kw_only=True)
    shuffle: bool = field(kw_only=True)


class Perm(Inferable):
    option: PermOption

    model: model.ModelPerm

    tokenizer: PreTrainedTokenizerFast

    dataset: data.DatasetPerm
    dataset_train: Subset[DataRawList]
    dataset_valid: Subset[DataRawList]
    dataset_test: Subset[DataRawList]

    def __init__(self, option: PermOption, *, state_dict=None) -> None:
        self.option = option

        with Status("Initializing model...", console=console) as status:
            self.model = self.__init_model()
            if state_dict is not None:
                self.model.load_state_dict(state_dict)

        with Status("Initializing pretrained tokenizer...", console=console):
            self.tokenizer = self.__init_tokenizer()

        self.device = torch.device("cpu")

    def __init_model(self):
        return model.ModelPerm()

    def __init_tokenizer(self) -> PreTrainedTokenizerFast:
        return AutoTokenizer.from_pretrained(env.PRETRAINED_MODEL_PERM)  # type: ignore

    def __init_dataset(self):
        self.dataset = data.DatasetPerm(
            {
                "max_graph_size": 16,
                "min_graph_size": 3,
                "seed": None,
            },
            self.tokenizer,
        )

        dataset_split_generator = (
            torch.Generator()
            if self.option.randomize_dataset
            else torch.Generator().manual_seed(42)
        )

        len_dataset = len(self.dataset)
        split_count = [int(len_dataset * r) for r in self.option.split_dataset_ratio]
        split_count[-1] += len_dataset - sum(split_count)

        self.dataset_train, self.dataset_valid, self.dataset_test = random_split(
            self.dataset, split_count, dataset_split_generator
        )

    def get_optimizer(self, train_option: PermTrainOption):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_option.learning_rate,
            eps=train_option.optim_eps,
        )

        return optimizer

    def set_device(self):
        with Status("Preparing device...", console=console) as status:
            self.device: torch.device = (
                torch.device(env.CUDA_DEVICE)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.model.to(self.device)

    def train(self, train_option: PermTrainOption):
        with Status("Initializing dataset...", console=console) as status:
            self.__init_dataset()

        with Status("Reshaping data....", console=console):
            dataloader_train = cast(
                DataLoader[data.DataBatchPerm],
                DataLoader(
                    data.DatasetPermBalanced(
                        self.dataset_train, train_option.batch_size
                    ),
                    batch_size=1,
                    shuffle=train_option.shuffle,
                    collate_fn=data.DatasetPermBalanced.collate,
                ),
            )

        optimizer = self.get_optimizer(train_option)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_option.warmup_step,
            num_training_steps=len(dataloader_train) * train_option.epoch,
        )

        from datetime import datetime

        timestamp = datetime.now().strftime("%m%dT%H%M%S")

        self.model: model.ModelPerm
        self.set_device()

        with BaroqueProgress() as (live, progress, status):
            task_train = progress.add_task("Training...", total=train_option.epoch)

            for epoch in range(train_option.epoch):
                progress.update(
                    task_train,
                    description=f"Training, epoch {epoch + 1} of {train_option.epoch}",
                )
                task_epoch = progress.add_task("", total=len(dataloader_train))

                metrics_train = metrics.MetricsPerm()
                self.model.train()

                for idx, batch in enumerate(dataloader_train):
                    batch: data.DataBatchPerm
                    progress.update(
                        task_epoch,
                        description=f"Batch {1 + idx} of {len(dataloader_train)}",
                    )
                    status.update(
                        f"loss {metrics_train.loss:.3f}, accuracy {metrics_train.accuracy():.3%}, precision {metrics_train.precision():.3%}, recall {metrics_train.recall():.3%}"
                    )

                    optimizer.zero_grad(set_to_none=True)
                    batch.move(self.device)
                    prediction = self.model(batch)

                    loss = metrics_train.accept(
                        prediction=prediction, target=nonnull(batch.target), batch=batch
                    )
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    progress.advance(task_epoch)

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

        if status:
            status.update("Reshapping data...")

        dataloader_eval = DataLoader(
            data.DatasetPermBalanced(dataset_eval, 16),
            batch_size=1,
            shuffle=False,
            collate_fn=data.DatasetPermBalanced.collate,
        )  # type: ignore
        dataloader_eval: DataLoader[data.DataBatchPerm]

        if status:
            status.update("")

        task_eval = (
            progress.add_task("Evaling...", total=len(dataloader_eval))
            if progress
            else None
        )

        metrics_eval = metrics.MetricsPerm()

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

                batch.move(self.device)
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
            self.__init_dataset()
        self.set_device()

        with BaroqueProgress() as (live, progress, status):
            metrics_eval = self.eval(
                self.dataset_test, progress=progress, status=status
            )

        console.log(
            f"test: loss {metrics_eval.loss:.3f}, accuracy {metrics_eval.accuracy():.3%}, precision {metrics_eval.precision():.3%}, recall {metrics_eval.recall():.3%}"
        )

    def infer(self, input: Sequence[Sequence[str]]) -> torch.Tensor:     
        with Status("Preparing data..."):
            def tokenize_sentence(idx: int, steps: Sequence[str]):
                tokenize_result = tokenize(self.tokenizer, steps) # type: ignore
                return DataRawList(
                    id=idx,
                    id_label=str(idx),
                    input_ids=tokenize_result["input_ids"],
                    attention_mask=tokenize_result["attention_mask"],
                    target=None,
                )
            
            all_data_raw_lists = list(starmap(tokenize_sentence, enumerate(input)))
            subset = Subset(all_data_raw_lists, list(range(len(all_data_raw_lists)))) # type: ignore
            dataset = data.DatasetPermBalanced(subset, 32)
            dataloader = cast(
                DataLoader[data.DataBatchPerm],
                DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=data.DatasetPermBalanced.collate,
                )
            )

        self.model.eval()
        self.set_device()

        max_token_len = max([len(steps) for steps in input])

        result_matrix = torch.zeros((len(input), max_token_len, max_token_len))

        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                batch: data.DataBatchPerm

                batch.move(self.device)
                prediction: torch.Tensor = self.model(batch)

                for (idx, i, j), t in zip(batch.pair_idx, prediction):
                    v = t.argmax(-1)
                    result_matrix[idx, i, j] = 0. if v == 0 else t[1]
        
        return result_matrix
