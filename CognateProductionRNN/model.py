from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from params import *

# Used code from https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

class Seq2Seq(nn.Module):
    #TODO Pass in parameters for encoder and decoder instead of the encoder and decoder directly
    # see https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(
        self, 
        encoder_layers,
        decoder_layers,
        encoder_input_size,
        encoder_hidden_size,
        encoder_dropout,
        decoder_output_size,
        decoder_hidden_size,
        decoder_dropout,
        device
    ):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderRNN(
            num_layers=encoder_layers,
            input_size=encoder_input_size,
            hidden_size=encoder_hidden_size,
            dropout_p=encoder_dropout,
            device=device
        )
        self.decoder = DecoderRNN(
            num_layers=decoder_layers,
            output_size=decoder_output_size,
            hidden_size=decoder_hidden_size,
            dropout_p=decoder_dropout,
            device=device
        )

    def forward(self, input, target_tensor=None, teacher_forcing_ratio=0.5):
        #TODO teacher_forcing_ratio
        encoder_outputs, encoder_hidden = self.encoder(input)
        decoder_outputs, decoder_hidden, attentions = self.decoder(encoder_outputs, encoder_hidden, target_tensor=target_tensor)
        return decoder_outputs, decoder_hidden, attentions
        

class EncoderRNN(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout_p=0.1, device='cuda'):
        super(EncoderRNN, self).__init__()
        print("ENCODER INPUT_SIZE:", input_size)
        print("ENCODER HIDDEN_SIZE:", hidden_size)
        print("ENCODER LAYERS:", num_layers)
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout_p)
        self.device = device

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, num_layers, hidden_size, output_size, dropout_p=0.1, device='cuda'):
        super(DecoderRNN, self).__init__()
        print("DECODER OUTPUT_SIZE:", output_size)
        print("DECODER HIDDEN_SIZE:", hidden_size)
        print("DECODER LAYERS:", num_layers)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.device = device

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # print("----------------------")
        # print("query", query.shape)
        # # print(query)
        # print("keys", keys.shape)
        # # print(keys)
        # print("wa query", self.Wa(query).shape)
        # # print(self.Wa(query))
        # print("ua keys", self.Ua(keys).shape)
        # # print(self.Ua(keys))
        # sum_result = self.Wa(query) + self.Ua(keys)
        # tanh_result = torch.tanh(sum_result)
        # scores = self.Va(tanh_result)

        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights