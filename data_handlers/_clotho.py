#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple
from pathlib import Path

from torch.utils.data import Dataset
from numpy import load as np_load, ndarray

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['ClothoDataset']


class ClothoDataset(Dataset):

    def __init__(self,
                 data_dir: Path,
                 split: str,
                 input_field_name: str,
                 output_field_name: str,
                 load_into_memory: bool) \
            -> None:
        """Initialization of a Clotho dataset object.

        :param data_dir: Data directory with Clotho dataset files.
        :type data_dir: pathlib.Path
        :param split: The split to use (`development`, `validation`)
        :type split: str
        :param input_field_name: Field name for the input values
        :type input_field_name: str
        :param output_field_name: Field name for the output (target) values.
        :type output_field_name: str
        :param load_into_memory: Load the dataset into memory?
        :type load_into_memory: bool
        """
        super(ClothoDataset, self).__init__()
        the_dir: Path = data_dir.joinpath(split)

        self.examples: List[Path] = sorted(the_dir.iterdir())
        self.input_name: str = input_field_name
        self.output_name: str = output_field_name
        self.load_into_memory: bool = load_into_memory

        if load_into_memory:
            self.examples: List[ndarray] = [
                np_load(str(f), allow_pickle=True)
                for f in self.examples]

    def __len__(self) \
            -> int:
        """Gets the amount of examples in the dataset.

        :return: Amount of examples in the dataset.
        :rtype: int
        """
        return len(self.examples)

    def __getitem__(self,
                    item: int) \
            -> Tuple[ndarray, ndarray, Path]:
        """Gets an example from the dataset.

        :param item: Index of the item.
        :type item: int
        :return: Input and output values, and the Path of the file.
        :rtype: numpy.ndarray. numpy.ndarray, Path
        """
        ex = self.examples[item]
        if not self.load_into_memory:
            ex = np_load(str(ex), allow_pickle=True)

        in_e, ou_e = [ex[i].item()
                      for i in [self.input_name, self.output_name]]

        return in_e, ou_e, ex.file_name[0]

# EOF
