#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import MutableMapping, Any
from datetime import datetime
from pathlib import Path
from functools import partial

from loguru import logger

from tools.printing import init_loggers
from tools.argument_parsing import get_argument_parser
from tools.dataset_creation import get_annotations_files, \
    get_amount_of_file_in_dir, check_data_for_split, \
    create_split_data, create_lists_and_frequencies
from tools.file_io import load_settings_file, load_yaml_file

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['create_dataset']


def create_dataset(settings_dataset: MutableMapping[str, Any],
                   settings_dirs_and_files: MutableMapping[str, Any]) \
        -> None:
    """Creates the dataset.

    Gets the dictionary with the settings and creates
    the files of the dataset.

    :param settings_dataset: Settings to be used for dataset\
                             creation.
    :type settings_dataset: dict
    :param settings_dirs_and_files: Settings to be used for\
                                    handling directories and\
                                    files.
    :type settings_dirs_and_files: dict
    """
    # Get logger
    inner_logger = logger.bind(
        indent=2, is_caption=False)

    # Get root dir
    dir_root = Path(settings_dirs_and_files[
                        'root_dirs']['data'])

    # Read the annotation files
    inner_logger.info('Reading annotations files')
    csv_dev, csv_eva = get_annotations_files(
        settings_ann=settings_dataset['annotations'],
        dir_ann=dir_root.joinpath(
            settings_dirs_and_files['dataset'][
                'annotations_dir']))
    inner_logger.info('Done')

    # Get all captions
    inner_logger.info('Getting the captions')
    captions_development = [
        csv_field.get(
            settings_dataset['annotations'][
                'captions_fields_prefix'].format(c_ind))
        for csv_field in csv_dev
        for c_ind in range(1, 6)]
    inner_logger.info('Done')

    # Create lists of indices and frequencies for words and\
    # characters.
    inner_logger.info('Creating and saving words and chars '
                      'lists and frequencies')
    words_list, chars_list = create_lists_and_frequencies(
        captions=captions_development, dir_root=dir_root,
        settings_ann=settings_dataset['annotations'],
        settings_cntr=settings_dirs_and_files['dataset'])
    inner_logger.info('Done')

    # Aux partial function for convenience.
    split_func = partial(
        create_split_data,
        words_list=words_list, chars_list=chars_list,
        settings_ann=settings_dataset['annotations'],
        settings_audio=settings_dataset['audio'],
        settings_output=settings_dirs_and_files['dataset'])

    settings_audio_dirs = settings_dirs_and_files[
        'dataset']['audio_dirs']

    # For each data split (i.e. development and evaluation)
    for split_data in [(csv_dev, 'development'),
                       (csv_eva, 'evaluation')]:

        # Get helper variables.
        split_name = split_data[1]
        split_csv = split_data[0]

        dir_split = dir_root.joinpath(
            settings_audio_dirs['output'],
            settings_audio_dirs[f'{split_name}'])

        dir_downloaded_audio = dir_root.joinpath(
            settings_audio_dirs['downloaded'],
            settings_audio_dirs[f'{split_name}'])

        # Create the data for the split.
        inner_logger.info(f'Creating the {split_name} '
                          f'split data')
        split_func(split_csv, dir_split,
                   dir_downloaded_audio)
        inner_logger.info('Done')

        # Count and print the amount of initial and resulting\
        # files.
        nb_files_audio = get_amount_of_file_in_dir(
            dir_downloaded_audio)
        nb_files_data = get_amount_of_file_in_dir(dir_split)

        inner_logger.info(f'Amount of {split_name} '
                          f'audio files: {nb_files_audio}')
        inner_logger.info(f'Amount of {split_name} '
                          f'data files: {nb_files_data}')
        inner_logger.info(f'Amount of {split_name} data '
                          f'files per audio: '
                          f'{nb_files_data / nb_files_audio}')

        if settings_dataset['workflow']['validate_dataset']:
            # Check the created lists of indices for words and characters.
            inner_logger.info(f'Checking the {split_name} split')
            check_data_for_split(
                dir_audio=dir_downloaded_audio,
                dir_data=Path(settings_audio_dirs['output'],
                              settings_audio_dirs[f'{split_name}']),
                dir_root=dir_root, csv_split=split_csv,
                settings_ann=settings_dataset['annotations'],
                settings_audio=settings_dataset['audio'],
                settings_cntr=settings_dirs_and_files['dataset'])
            inner_logger.info('Done')
        else:
            inner_logger.info(f'Skipping validation of {split_name} split')


def main():

    args = get_argument_parser().parse_args()

    file_dir = args.file_dir
    config_file = args.config_file
    file_ext = args.file_ext
    verbose = args.verbose

    settings = load_yaml_file(Path(
        file_dir, f'{config_file}.{file_ext}'))

    init_loggers(verbose=verbose,
                 settings=settings['dirs_and_files']['logging'])

    logger_main = logger.bind(is_caption=False, indent=1)

    logger_main.info(datetime.now().strftime('%Y-%m-%d %H:%M'))

    logger_main.info('Doing only dataset creation')

    # Load settings file.
    logger_main.info('Loading settings')
    settings = load_settings_file(args.config_file)
    logger_main.info('Settings loaded')

    # Create the dataset.
    logger_main.info('Starting Clotho dataset creation')
    create_dataset(
        settings_dataset=settings['dataset_creation_settings'],
        settings_dirs_and_files=settings['dirs_and_files'])
    logger_main.info('Dataset created')


if __name__ == '__main__':
    main()

# EOF
