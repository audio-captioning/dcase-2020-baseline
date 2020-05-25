#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from loguru import logger

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

    logger_main = logger.bind(is_caption=False, indent=0)
    logger_inner = logger.bind(is_caption=False, indent=1)

    if settings['workflow']['dataset_creation']:
        logger_main.info('Starting creation of dataset')

        logger_inner.info('Creating examples')
        dataset.create_dataset(
            settings_dataset=settings['dataset_creation_settings'],
            settings_dirs_and_files=settings['dirs_and_files'])
        logger_inner.info('Examples created')

        logger_inner.info('Extracting features')
        dataset.extract_features(
            root_dir=settings['dirs_and_files']['root_dirs']['data'],
            settings_data=settings['dirs_and_files']['dataset'],
            settings_features=settings['feature_extraction_settings'])
        logger_inner.info('Features extracted')
        logger_main.info('Creation of dataset ended')

    if settings['workflow']['dnn_testing']:
        # Create test dataset if not yet created
        test_split_feat_dir = Path(settings['dirs_and_files']['root_dirs']['data']) \
            .joinpath(settings['dirs_and_files']['dataset']['features_dirs']['output'],
                      settings['dirs_and_files']['dataset']['features_dirs']['test'])
        if not test_split_feat_dir.exists():
            logger_main.info('Starting creation of test dataset')
            logger_inner.info('Extracting features')
            dataset.extract_features_test(
                root_dir=settings['dirs_and_files']['root_dirs']['data'],
                settings_data=settings['dirs_and_files']['dataset'],
                settings_features=settings['feature_extraction_settings'],
                settings_audio=settings['dataset_creation_settings']['audio'])
            logger_inner.info('Features extracted')
            logger_main.info('Creation of test dataset ended')
        else:
            logger_inner.info('Found existing test data')

    if settings['workflow']['dnn_training'] or \
            settings['workflow']['dnn_evaluation'] or \
            settings['workflow']['dnn_testing']:
        method.method(settings)


if __name__ == '__main__':
    main()

# EOF
