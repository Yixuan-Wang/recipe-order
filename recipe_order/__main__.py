from pathlib import Path
import typer
from typing import Annotated, Optional
from enum import Enum

from rich.status import Status


class NetworkFlavor(str, Enum):
    Perm = "Perm"
    Pointer = "Pointer"


app = typer.Typer()


@app.command()
def train(
    flavor: Annotated[NetworkFlavor, typer.Argument(case_sensitive=False)],
    dataset_seed: Annotated[Optional[int], typer.Option("-d", "--dataset-seed")] = None,
    epoch: Annotated[int, typer.Option("-e", "--epoch")] = 3,
    batch_size: Annotated[Optional[int], typer.Option("-b", "--batch")] = None,
    learning_rate: Annotated[Optional[float], typer.Option("--lr")] = None,
    weight_decay: Annotated[Optional[float], typer.Option()] = None,
    warmup_step: Annotated[int, typer.Option()] = 20,
    rand: Annotated[bool, typer.Option()] = True,
):
    match flavor:
        case NetworkFlavor.Perm:
            from network.perm import Perm, PermOption, PermTrainOption

            perm = Perm(
                PermOption(
                    randomize_dataset=rand,
                )
            )

            perm.train(
                PermTrainOption(
                    batch_size=batch_size or 16,
                    learning_rate=learning_rate or 5e-5,
                    weight_decay=weight_decay or 0,
                    epoch=epoch,
                    warmup_step=warmup_step,
                    shuffle=rand,
                )
            )
        case NetworkFlavor.Pointer:
            from network.pointer import Pointer, PointerOption, PointerTrainOption

            pointer = Pointer(
                PointerOption(
                    randomize_dataset=rand,
                )
            )

            pointer.train(
                PointerTrainOption(
                    batch_size=8,
                    learning_rate=5e-5,
                    weight_decay=weight_decay or 1e-3,
                    epoch=epoch,
                    warmup_step=warmup_step,
                    shuffle=rand,
                )
            )


@app.command()
def test(
    flavor: Annotated[NetworkFlavor, typer.Argument(case_sensitive=False)],
    weights: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
):
    import torch

    with Status("Loading pretrained weights [u]%s[/u]" % str(weights)):
        pretrained_weights = torch.load(weights)

    match flavor:
        case NetworkFlavor.Pointer:
            from network.pointer import Pointer, PointerOption

            pointer = Pointer(
                PointerOption(
                    randomize_dataset=False,
                ),
                pretrained_weights,
            )
            pointer.test()
        case NetworkFlavor.Perm:
            from network.perm import Perm, PermOption

            perm = Perm(
                PermOption(
                    randomize_dataset=False,
                ),
                state_dict=pretrained_weights,
            )
            perm.test()


if __name__ == "__main__":
    app()
