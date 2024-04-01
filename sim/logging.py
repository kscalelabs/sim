"""Defines utility functions for enabling logging."""

import datetime
import logging
import re
import sys
from typing import Literal

# Show as a transient message.
LOG_PING: int = logging.INFO + 2

# Show as a persistent status message.
LOG_STATUS: int = logging.INFO + 3

RESET_SEQ = "\033[0m"
REG_COLOR_SEQ = "\033[%dm"
BOLD_COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

Color = Literal[
    "black",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
    "grey",
    "light-red",
    "light-green",
    "light-yellow",
    "light-blue",
    "light-magenta",
    "light-cyan",
]

COLOR_INDEX: dict[Color, int] = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
    "grey": 90,
    "light-red": 91,
    "light-green": 92,
    "light-yellow": 93,
    "light-blue": 94,
    "light-magenta": 95,
    "light-cyan": 96,
}


def color_parts(color: Color, bold: bool = False) -> tuple[str, str]:
    if bold:
        return BOLD_COLOR_SEQ % COLOR_INDEX[color], RESET_SEQ
    return REG_COLOR_SEQ % COLOR_INDEX[color], RESET_SEQ


def uncolored(s: str) -> str:
    return re.sub(r"\033\[[\d;]+m", "", s)


def colored(s: str, color: Color | None = None, bold: bool = False) -> str:
    if color is None:
        return s
    start, end = color_parts(color, bold=bold)
    return start + s + end


def wrapped(
    s: str,
    length: int | None = None,
    space: str = " ",
    spaces: str | re.Pattern = r" ",
    newlines: str | re.Pattern = r"[\n\r]",
    too_long_suffix: str = "...",
) -> list[str]:
    strings = []
    lines = re.split(newlines, s.strip(), flags=re.MULTILINE | re.UNICODE)
    for line in lines:
        cur_string = []
        cur_length = 0
        for part in re.split(spaces, line.strip(), flags=re.MULTILINE | re.UNICODE):
            if length is None:
                cur_string.append(part)
                cur_length += len(space) + len(part)
            else:
                if len(part) > length:
                    part = part[: length - len(too_long_suffix)] + too_long_suffix
                if cur_length + len(part) > length:
                    strings.append(space.join(cur_string))
                    cur_string = [part]
                    cur_length = len(part)
                else:
                    cur_string.append(part)
                    cur_length += len(space) + len(part)
        if cur_length > 0:
            strings.append(space.join(cur_string))
    return strings


def outlined(
    s: str,
    inner: Color | None = None,
    side: Color | None = None,
    bold: bool = False,
    max_length: int | None = None,
    space: str = " ",
    spaces: str | re.Pattern = r" ",
    newlines: str | re.Pattern = r"[\n\r]",
) -> str:
    strs = wrapped(uncolored(s), max_length, space, spaces, newlines)
    max_len = max(len(s) for s in strs)
    strs = [f"{s}{' ' * (max_len - len(s))}" for s in strs]
    strs = [colored(s, inner, bold=bold) for s in strs]
    strs_with_sides = [f"{colored('│', side)} {s} {colored('│', side)}" for s in strs]
    top = colored("┌─" + "─" * max_len + "─┐", side)
    bottom = colored("└─" + "─" * max_len + "─┘", side)
    return "\n".join([top] + strs_with_sides + [bottom])


def show_info(s: str, important: bool = False) -> None:
    if important:
        s = outlined(s, inner="light-cyan", side="cyan", bold=True)
    else:
        s = colored(s, "light-cyan", bold=False)
    sys.stdout.write(s)
    sys.stdout.write("\n")
    sys.stdout.flush()


def show_error(s: str, important: bool = False) -> None:
    if important:
        s = outlined(s, inner="light-red", side="red", bold=True)
    else:
        s = colored(s, "light-red", bold=False)
    sys.stdout.write(s)
    sys.stdout.write("\n")
    sys.stdout.flush()


def show_warning(s: str, important: bool = False) -> None:
    if important:
        s = outlined(s, inner="light-yellow", side="yellow", bold=True)
    else:
        s = colored(s, "light-yellow", bold=False)
    sys.stdout.write(s)
    sys.stdout.write("\n")
    sys.stdout.flush()


def format_timedelta(timedelta: datetime.timedelta, short: bool = False) -> str:
    """Formats a delta time to human-readable format.

    Args:
        timedelta: The delta to format
        short: If set, uses a shorter format

    Returns:
        The human-readable time delta
    """
    parts = []
    if timedelta.days > 0:
        if short:
            parts += [f"{timedelta.days}d"]
        else:
            parts += [f"{timedelta.days} day" if timedelta.days == 1 else f"{timedelta.days} days"]

    seconds = timedelta.seconds

    if seconds > 60 * 60:
        hours, seconds = seconds // (60 * 60), seconds % (60 * 60)
        if short:
            parts += [f"{hours}h"]
        else:
            parts += [f"{hours} hour" if hours == 1 else f"{hours} hours"]

    if seconds > 60:
        minutes, seconds = seconds // 60, seconds % 60
        if short:
            parts += [f"{minutes}m"]
        else:
            parts += [f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"]

    if short:
        parts += [f"{seconds}s"]
    else:
        parts += [f"{seconds} second" if seconds == 1 else f"{seconds} seconds"]

    return ", ".join(parts)


def format_datetime(dt: datetime.datetime) -> str:
    """Formats a datetime to human-readable format.

    Args:
        dt: The datetime to format

    Returns:
        The human-readable datetime
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def camelcase_to_snakecase(s: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def snakecase_to_camelcase(s: str) -> str:
    return "".join(word.title() for word in s.split("_"))


def highlight_exception_message(s: str) -> str:
    s = re.sub(r"(\w+Error)", r"\033[1;31m\1\033[0m", s)
    s = re.sub(r"(\w+Exception)", r"\033[1;31m\1\033[0m", s)
    s = re.sub(r"(\w+Warning)", r"\033[1;33m\1\033[0m", s)
    s = re.sub(r"\^+", r"\033[1;35m\g<0>\033[0m", s)
    s = re.sub(r"File \"(.+?)\"", r'File "\033[36m\1\033[0m"', s)
    return s


def is_interactive_session() -> bool:
    return hasattr(sys, "ps1") or hasattr(sys, "ps2") or sys.stdout.isatty() or sys.stderr.isatty()


class ColoredFormatter(logging.Formatter):
    """Defines a custom formatter for displaying logs."""

    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    COLORS: dict[str, Color] = {
        "WARNING": "yellow",
        "INFO": "cyan",
        "DEBUG": "grey",
        "CRITICAL": "yellow",
        "FATAL": "red",
        "ERROR": "red",
        "STATUS": "green",
        "PING": "magenta",
    }

    def __init__(
        self,
        *,
        prefix: str | None = None,
        use_color: bool = True,
    ) -> None:
        asc_start, asc_end = color_parts("grey")
        name_start, name_end = color_parts("blue", bold=True)

        message_pre = [
            "{levelname:^19s}",
            asc_start,
            "{asctime}",
            asc_end,
            " [",
            name_start,
            "{name}",
            name_end,
            "]",
        ]
        message_post = [" {message}"]

        if prefix is not None:
            message_pre += [" ", colored(prefix, "magenta", bold=True)]

        message = "".join(message_pre + message_post)

        super().__init__(message, style="{", datefmt="%Y-%m-%d %H:%M:%S")

        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if record.levelname and self.use_color and levelname in self.COLORS:
            record.levelname = colored(record.levelname, self.COLORS[levelname], bold=True)
        return logging.Formatter.format(self, record)


def configure_logging(prefix: str | None = None, level: int = logging.INFO) -> None:
    """Instantiates logging.

    This captures logs and reroutes them to the Toasts module, which is
    pretty similar to Python logging except that the API is a lot easier to
    interact with.

    Args:
        prefix: An optional prefix to add to the logger
        level: The logging level
    """
    root_logger = logging.getLogger()

    # Adds new level names.
    logging.addLevelName(LOG_PING, "PING")
    logging.addLevelName(LOG_STATUS, "STATUS")

    # Captures warnings from the warnings module.
    logging.captureWarnings(True)

    # Clears all existing handlers.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Adds new handler.
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(ColoredFormatter(prefix=prefix))
    root_logger.addHandler(stream_handler)

    root_logger.setLevel(level)
