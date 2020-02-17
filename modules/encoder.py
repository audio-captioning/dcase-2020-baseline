#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import cat, Tensor
from torch.nn import Module, GRU, Dropout

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['Encoder']


class Encoder(Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 dropout_p: float) \
            -> None:
        """Encoder module.

        :param input_dim: Input dimensionality.
        :type input_dim: int
        :param hidden_dim: Hidden dimensionality.
        :type hidden_dim: int
        :param output_dim: Output dimensionality.
        :type output_dim: int
        :param dropout_p: Dropout.
        :type dropout_p: float
        """
        super(Encoder, self).__init__()

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.output_dim: int = output_dim

        self.dropout: Module = Dropout(p=dropout_p)

        rnn_common_args = {
            'num_layers': 1,
            'bias': True,
            'batch_first': True,
            'bidirectional': True}

        self.gru_1: Module = GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            **rnn_common_args)

        self.gru_2: Module = GRU(
            input_size=self.hidden_dim*2,
            hidden_size=self.hidden_dim,
            **rnn_common_args)

        self.gru_3: Module = GRU(
            input_size=self.hidden_dim*2,
            hidden_size=self.output_dim,
            **rnn_common_args)

    def _l_pass(self,
                layer: Module,
                layer_input: Tensor) \
            -> Tensor:
        """Does the forward passing for a GRU layer.

        :param layer: GRU layer for forward passing.
        :type layer: torch.nn.Module
        :param layer_input: Input to the GRU layer.
        :type layer_input: torch.Tensor
        :return: Output of the GRU layer.
        :rtype: torch.Tensor
        """
        b_size, t_steps, _ = layer_input.size()
        h = layer(layer_input)[0].view(b_size, t_steps, 2, -1)
        return self.dropout(cat([h[:, :, 0, :], h[:, :, 1, :]], dim=-1))

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the encoder.

        :param x: Input to the encoder.
        :type x: torch.Tensor
        :return: Output of the encoder.
        :rtype: torch.Tensor
        """
        h = self._l_pass(self.gru_1, x)

        for a_layer in [self.gru_2, self.gru_3]:
            h_ = self._l_pass(a_layer, h)
            h = h + h_ if h.size()[-1] == h_.size()[-1] else h_

        return h

# EOF
