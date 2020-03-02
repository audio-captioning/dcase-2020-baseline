#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from loguru import logger

import evaluation
from processes import method, dataset
from tools.file_io import load_yaml_file
from tools.printing import init_loggers
from tools.argument_parsing import get_argument_parser

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['main']


def main():
    args = get_argument_parser().parse_args()

    file_dir = args.file_dir
    config_file = args.config_file
    file_ext = args.file_ext
    verbose = args.verbose

    settings = load_yaml_file(Path(
        file_dir, f'{config_file}.{file_ext}'))

    init_loggers(verbose=verbose,
                 settings=settings['dirs_and_files'])

    logger_main = logger.bind(is_caption=False, indent=1)

    if settings['workflow']['dataset_creation']:
        logger_main.info('Starting creation of dataset')
        dataset.create_dataset(
            settings_dataset=settings['dataset_creation_settings'],
            settings_dirs_and_files=settings['dirs_and_files'])
        logger_main.info('Creation of dataset ended')

    if settings['workflow']['dnn_training']:
        logger_main.info('Starting optimization of method')
        method.method(settings)
        logger_main.info('Optimization of method ended')

    if settings['workflow']['captions_evaluation']:
        logger_main.info('Starting evaluation of captions')
        evaluation.evaluate_captions(settings)
        logger_main.info('Evaluation of captions ended')


if __name__ == '__main__':
    main()

# EOF
