#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import stdout
from pathlib import Path
from pprint import PrettyPrinter

from loguru import logger
from loguru._handler import StrRecord
from _io import TextIOWrapper

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_pretty_printer',
           'init_loggers']


def _rotation_logger(x: StrRecord,
                     _: TextIOWrapper) \
        -> bool:
    """Callable to determine the rotation of files in logger.

    :param x: Str to be logged.
    :type x: loguru._handler.StrRecord
    :param _: File used for logging.
    :type _: _io.TextIOWrapper
    :return: Shall we switch to a new file?
    :rtype: bool
    """
    return 'Captions start' in x


def get_pretty_printer():
    """Gets the pprint.

    :return: Pretty printer.
    :rtype: pprint.PrettyPrinter
    """
    return PrettyPrinter(indent=4, width=100)


def init_loggers(verbose, settings):
    """Initializes the logging process.

    :param verbose: Be verbose?
    :type verbose: bool
    :param settings: Settings to use.
    :type settings: dict
    """
    logger.remove()

    for indent in range(3):
        log_string = '{level} | [{time}] {name} -- {space}{message}'.format(
            level='{level}',
            time='{time:HH:mm:ss}',
            name='{name}',
            message='{message}',
            space=' ' * (indent*2))
        logger.add(
            stdout,
            format=log_string,
            level='INFO',
            filter=lambda record, i=indent:
            record['extra']['indent'] == i and not record['extra']['is_caption'])

    logging_path = Path(settings['root_dirs']['outputs'],
                        settings['logging']['logger_dir'])

    log_file_main = f'{settings["logging"]["caption_logger_file"]}'

    logging_file = logging_path.joinpath(log_file_main)

    logger.add(str(logging_file), format='{message}', level='INFO',
               filter=lambda record: record['extra']['is_caption'],
               rotation=_rotation_logger)

    logging_path.mkdir(parents=True, exist_ok=True)

    if not verbose:
        logger.disable()


# EOF
