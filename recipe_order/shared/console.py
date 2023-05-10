from __future__ import annotations

from rich.console import Console
from rich.logging import RichHandler
import logging

console = Console()

_FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO",
    format=_FORMAT,
    handlers=[RichHandler(console=console)],
)
log = logging.getLogger("rich")
