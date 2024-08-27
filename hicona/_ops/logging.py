"""Custom logging module for hicona."""

import logging


def get_console_logger(name: str) -> logging.Logger:
    """Get a logger for console output."""

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%H:%M:%S",
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    return logging.getLogger(name)
