from __future__ import annotations
from collections import deque

from dataclasses import dataclass, field, fields
import itertools
from typing import Optional
import torch

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from torch.utils.data import Dataset, IterableDataset, Subset
from shared.graph import reachability_matrix

from shared.traindata import DataMmresList, DatasetMmres, DatasetMmresOption
import shared.env
from utils.functional import regroup, take
from utils.stub import pad_to

# @dataclass
# class DataMmresList(DataMmresList):


#     @classmethod
#     def new(cls, datum: DataMmresList | DataMmresList, view: Optional[slice] = None):
#         view = view or slice(None)
#         if isinstance(datum, DataMmresList):
#             datum.view = view
#             return datum
#         else:
#             datum_view =  DataMmresList(**asdict_shallow(datum))
#             datum_view.view = view
#             return datum_view


@dataclass
class DataBatchPerm:
    # original_size: list[torch.Size] = field(kw_only=True)
    batch_size: int = field(kw_only=True)
    max_size_token: int = field(kw_only=True)
    # max_size: tuple[int, int] = field(kw_only=True)

    id: list[int] = field(kw_only=True)
    id_label: list[str] = field(kw_only=True)

    pair_idx: list[tuple[int, int, int]] = field(kw_only=True)

    input_ids: torch.Tensor = field(kw_only=True)
    attention_mask: torch.Tensor = field(kw_only=True)
    token_type_ids: torch.Tensor = field(kw_only=True)

    target: torch.Tensor = field(kw_only=True)

    def get_input(self):
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "token_type_ids": self.token_type_ids,
        }

    def move(self, device: torch.device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.token_type_ids = self.token_type_ids.to(device)
        self.target = self.target.to(device)


class DatasetPerm(DatasetMmres, Dataset[DataMmresList]):
    def __init__(self, option: DatasetMmresOption, tokenizer: PreTrainedTokenizerFast):
        super().__init__(option, tokenizer)

    def init_tokenizer(self):
        return AutoTokenizer.from_pretrained(shared.env.PRETRAINED_MODEL_PERM)

    def __getitem__(self, index: int):
        line = self.df.loc[index]

        datum = DataMmresList(
            id=index,
            id_label=line["id"],
            input_ids=line["input_ids"],
            attention_mask=line["attention_mask"],
            target=line["adj"].to(dtype=torch.int),
        )

        return datum

    def __len__(self):
        return len(self.df)


class DatasetPermBalanced(Dataset[list[DataMmresList]]):
    def __init__(self, subset: Subset[DataMmresList], batch_max_len: int):
        self.subset = subset
        self.batch_max_len = batch_max_len
        self.indices = self.balance()

    def balance(self):
        all_views: list[list[int | tuple[int, slice]]] = [[]]
        residuals: deque[tuple[int, slice]] = deque()
        cur_length = 0

        for idx_datum in self.subset.indices:
            datum = self.subset.dataset[idx_datum]
            step = len(datum)
            length = step * (step - 1)

            if cur_length + length > self.batch_max_len:
                cut = self.batch_max_len - cur_length
                all_views[-1].append((idx_datum, slice(0, cut)))
                residuals.append((idx_datum, slice(cut, None)))
                all_views.append([])
                cur_length = 0
            else:
                all_views[-1].append(idx_datum)
                cur_length += length

        while len(residuals):
            residual = residuals.popleft()
            step = len(self.subset.dataset[residual[0]])
            length = step * (step - 1) - (residual[1].start or 0)

            if cur_length + length > self.batch_max_len:
                cut = residual[1].start + self.batch_max_len - cur_length
                all_views[-1].append((residual[0], slice(residual[1].start, cut)))
                residuals.appendleft((residual[0], slice(cut, None)))

                all_views.append([])
                cur_length = 0
            else:
                all_views[-1].append(residual)
                cur_length += length

        if len(all_views[-1]) == 0:
            all_views.pop()

        return all_views

    def __getitem__(self, idx: int) -> list[DataMmresList]:
        l = [
            self.subset.dataset[idx_datum]
            if isinstance(idx_datum, int)
            else self.subset.dataset[idx_datum[0]].with_view(idx_datum[1])
            for idx_datum in self.indices[idx]
        ]
        assert not len(l) > self.batch_max_len

        return l

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def collate(one_data: list[list[DataMmresList]]) -> DataBatchPerm:
        assert len(one_data) == 1
        data = one_data[0]

        def permute_one(idx: int, datum: DataMmresList):
            step = len(datum.input_ids)

            all_input_ids: list[torch.Tensor] = []
            all_attention_mask: list[torch.Tensor] = []
            all_token_type_ids: list[torch.Tensor] = []
            all_target: list[int] = []
            all_pair_idx: list[tuple[int, int, int]] = []

            _ = itertools.permutations(range(step), 2)
            _ = itertools.islice(_, datum.view.start, datum.view.stop)

            reachability = reachability_matrix(datum.target)

            for i, j in _:
                input_ids = torch.concat((datum.input_ids[i], datum.input_ids[j][1:]))
                attention_mask = torch.concat(
                    (datum.attention_mask[i], datum.attention_mask[j][1:])
                )
                token_type_ids = torch.ones(
                    (ilen := datum.input_ids[i].shape[-1])
                    + datum.input_ids[j].shape[-1]
                    - 1,
                    dtype=torch.int,
                )
                token_type_ids[:ilen] = 0
                target: int = reachability[i, j].item()  # type: ignore

                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_token_type_ids.append(token_type_ids)
                all_target.append(target)
                all_pair_idx.append((idx, i, j))

            return {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_mask,
                "token_type_ids": all_token_type_ids,
                "target": all_target,
                "pair_idx": all_pair_idx,
            }

        regrouped = regroup([permute_one(idx, datum) for idx, datum in enumerate(data)])

        max_size_token = max(tokens.shape[0] for tokens in regrouped["input_ids"])

        def pad_sentence(t: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(
                t, (0, max_size_token - t.shape[-1]), value=0.0
            )

        input_ids = torch.stack([pad_sentence(t) for t in regrouped["input_ids"]])
        attention_mask = torch.stack(
            [pad_sentence(t) for t in regrouped["attention_mask"]]
        )
        token_type_ids = torch.stack(
            [pad_sentence(t) for t in regrouped["token_type_ids"]]
        )
        target = torch.tensor(regrouped["target"], dtype=torch.int64)

        id = [datum.id for datum in data]
        id_label = [datum.id_label for datum in data]

        return DataBatchPerm(
            batch_size=len(data),
            max_size_token=max_size_token,
            id=id,
            id_label=id_label,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pair_idx=regrouped["pair_idx"],
            target=target,
        )
