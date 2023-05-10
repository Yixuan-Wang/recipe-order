from __future__ import annotations

from rich.progress import Progress
from rich.console import Group
from rich.live import Live
from rich.status import Status

from shared.console import console

import contextlib

@contextlib.contextmanager
def BaroqueProgress(transient: bool = True):
    try:
        live = Live(
            Group(    
                progress := Progress(console=console),
                status := Status("dots"),
            ),
            console=console,
            transient=transient,
        )
        clock_start = console.get_datetime()
        with live:
            yield (live, progress, status)
    except:
        if transient and "status" in locals():
            console.log(locals()["status"].renderable)
        raise
    finally:
        if "clock_start" in locals():
            clock_start = locals()["clock_start"]
            clock_end = console.get_datetime()  
            console.log("Task started: ", clock_start, ", ended:", clock_end, ", elapsed: ", clock_end - clock_start, sep="")
