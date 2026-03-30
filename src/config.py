"""
Global Configuration and Initialization Core.

Establishes OS-agnostic repository directory paths, configures the runtime
logger formatting, and defines foundational hyperparameter constants.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_RANDOM_SEED: int = 42
CROSS_ENTROPY_L2_REGULARIZATION: float = 1e-4


def setup_logger(logger_name: str) -> logging.Logger:
    """
    Constructs a timestamped console logger replacing standard print output.

    Args:
        logger_name (str): Identifier for the logger instance, typically __name__.

    Returns:
        logging.Logger: Configured logger with INFO-level console output.
    """
    logger = logging.getLogger(logger_name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger
