"""File output helpers that preserve existing data on write failures."""

import os
import stat
from pathlib import Path
from tempfile import NamedTemporaryFile


def validate_output_path(output: Path, *inputs: Path) -> None:
    """Reject output paths aliasing an input, including symbolic and hard links."""
    destination = output.expanduser().resolve()
    for source in inputs:
        source = source.expanduser().resolve()
        if destination == source or (
            destination.exists() and source.exists() and destination.samefile(source)
        ):
            raise ValueError(f"output must differ from input file: {source}")


def write_text_atomic(path: Path, text: str) -> None:
    """Publish UTF-8 text only after a complete write, preserving existing modes.

    A temporary file in the destination directory keeps replacement on the
    same filesystem. Symbolic links retain their normal write-through behavior.
    """
    destination = path.expanduser().resolve()
    try:
        mode = stat.S_IMODE(destination.stat().st_mode)
    except FileNotFoundError:
        mode = None
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            temporary.chmod(mode)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
