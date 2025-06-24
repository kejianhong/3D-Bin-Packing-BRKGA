import logging
import sys

from colorama import Fore, Style, init

init(autoreset=True)

log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

COLORS = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.MAGENTA,
}


class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelno, Fore.WHITE)
        logMessage = super().format(record)
        coloredLogMessage = f"{color}{logMessage}{Style.RESET_ALL}"
        return coloredLogMessage


formatter = ColoredFormatter("%(asctime)s  [%(levelname)s] [%(module)s] [%(filename)s: %(lineno)d %(funcName)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)

log.addHandler(handler)
