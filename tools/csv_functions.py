#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, List, Union, Dict
from pathlib import Path
from collections import OrderedDict
from datetime import datetime

import csv

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['read_csv_file']


def read_csv_file(file_name: str,
                  base_dir: Optional[Union[str, Path]] = 'csv_files') \
        -> List[OrderedDict]:
    """Reads a CSV file.

    :param file_name: The full file name of the CSV.
    :type file_name: str
    :param base_dir: The root dir of the CSV files.
    :type base_dir: str|pathlib.Path
    :return: The contents of the CSV of the task.
    :rtype: list[collections.OrderedDict]
    """
    file_path = Path().joinpath(base_dir, file_name)
    with file_path.open(mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        return [csv_line for csv_line in csv_reader]


def write_csv_file(data: List[Dict[str, str]],
                   file_name: Union[str, Path],
                   base_dir: Union[str, Path],
                   add_timestamp: Optional[bool] = False) \
        -> None:
    """Writes a CSV file with an optional timestamp.

    :param data: Data to write. Format as taken by DictWriter (i.e. as given by DictReader).
    :type data: list[dict[str, str]]
    :param file_name: Name of the output file.
    :type file_name: str|pathlib.Path
    :param base_dir: Directory of the output file.
    :type base_dir: str|pathlib.Path
    :param add_timestamp: Wether to add timestamp to the file name or not.
    :type add_timestamp: bool
    """
    file_name = Path(str(file_name))
    if add_timestamp:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
        file_name = Path(file_name.stem + '.' + timestamp + file_name.suffix)
    file_path = Path().joinpath(base_dir, file_name)
    with file_path.open(mode='w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, data[0].keys())
        csv_writer.writeheader()
        for row in data:
            csv_writer.writerow(row)

# EOF
