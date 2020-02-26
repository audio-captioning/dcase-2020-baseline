#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Dict, Union
from itertools import chain
from argparse import ArgumentParser

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_argument_parser']


def _arguments_dataset() \
        -> List[List[Dict[str, Union[str, bool, int]]]]:
    """Returns the necessary arguments for the creation\
       of the dataset.

    :return: Command line arguments for the creation of\
             the dataset.
    :rtype: list[list[dict[str, str|bool|int]]]
    """
    return [
        # ---------------------------------
        [['--config-file-dataset'],
         {'type': str, 'required': True,
          'default': 'dataset_creation',
          'help': 'The settings file for the '
                  'creation of dataset (without extension).'}],
        # ---------------------------------
        [['--config-file-files'],
         {'type': str, 'required': True,
          'default': 'dirs_and_files',
          'help': 'The settings file for the '
                  'file i/o (without extension).'}]]


def _arguments_default() \
        -> List[List[Dict[str, Union[str, bool, int]]]]:
    """Returns the default command line arguments for the project.

    :return: Default command line arguments for the\
             project.
    :rtype: list[list[dict[str, str|bool|int]]]
    """
    return [
        # ---------------------------------
        [['--file-dir', '-d'],
         {'type': str, 'default': 'settings',
          'help': 'Directory that holds the settings file (default: `settings`).'}],
        # ---------------------------------
        [['--file-ext', '-e'],
         {'type': str, 'default': 'yaml',
          'help': 'Extension of the settings file (default: `yaml`).'}],
        # ---------------------------------
        [['--verbose', '-v'],
         {'default': False, 'action': 'store_true',
          'help': 'Be verbose flag (default False).'}]]


def _arguments_dnn() \
        -> List[List[Dict[str, Union[str, bool, int]]]]:
    """Returns the necessary arguments for the optimization\
       of the DNN.

    :return: Command line arguments for the process of\
             optimizing the DNN.
    :rtype: list[list[dict[str, str|bool|int]]]
    """
    return [
        # ---------------------------------
        [['--config-file-dnn'],
         {'type': str, 'required': True,
          'default': 'method_baseline',
          'help': 'The settings file for the '
                  'optimization of the DNN (without extension).'}]]


def _arguments_evaluation() \
        -> List[List[Dict[str, Union[str, bool, int]]]]:
    """Returns the necessary arguments for the evaluation\
       of captions.

    :return: Command line arguments for the process of\
             optimizing the DNN.
    :rtype: list[list[dict[str, str|bool|int]]]
    """
    return [
        # ---------------------------------
        [['--config-file-captions'],
         {'type': str, 'required': True,
          'default': 'captions_evaluation',
          'help': 'The settings file for the '
                  'evaluation of captions (without extension).'}]]


def get_argument_parser(create_dataset: bool,
                        optimize_dnn: bool,
                        evaluate_captions: bool) \
        -> ArgumentParser:
    """Creates and return the command line argument parser.

    :param create_dataset: Include arguments for creating the\
                           dataset?
    :type create_dataset: bool
    :param optimize_dnn: Include arguments for optimizing the\
                         DNN?
    :type optimize_dnn: bool
    :param evaluate_captions: Include arguments for evaluating\
                              the captions?
    :type evaluate_captions: bool
    :return: Argument parser having the indicated command line\
             arguments.
    :rtype: argparse.ArgumentParser
    """
    fncs_list = [_arguments_dataset, _arguments_dnn, _arguments_evaluation]
    workflow_list = [create_dataset, optimize_dnn, evaluate_captions]

    the_args = list(chain.from_iterable([
        i[0]() for i in zip(fncs_list, workflow_list) if i[1]]))

    the_args.extend(_arguments_default())

    arg_parser = ArgumentParser()
    [arg_parser.add_argument(*i[0], **i[1]) for i in the_args]

    return arg_parser

# EOF
