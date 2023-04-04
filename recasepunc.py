import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertTokenizer

default_config = argparse.Namespace(
    seed=871253,
    lang='ru',
    # flavor='flaubert/flaubert_base_uncased',
    flavor=None,
    max_length=256,
    batch_size=16,
    updates=50000,
    period=1000,
    lr=1e-5,
    dab_rate=0.1,
    device='cuda',
    debug=False
)

default_flavors = {
    'fr': 'flaubert/flaubert_base_uncased',
    'en': 'bert-base-uncased',
    'zh': 'ckiplab/bert-base-chinese',
    'tr': 'dbmdz/bert-base-turkish-uncased',
    'de': 'dbmdz/bert-base-german-uncased',
    'pt': 'neuralmind/bert-base-portuguese-cased',
    'ru': 'DeepPavlov/rubert-base-cased'
}


class Config(argparse.Namespace):
    def __init__(self, **kwargs):
        for key, value in default_config.__dict__.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

        assert self.lang in ['fr', 'en', 'zh', 'tr', 'pt', 'de', 'ru']

        if 'lang' in kwargs and ('flavor' not in kwargs or kwargs['flavor'] is None):
            self.flavor = default_flavors[self.lang]

        # print(self.lang, self.flavor)


def init_random(seed):
    # make sure everything is deterministic
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# NOTE: it is assumed in the implementation that y[:,0] is the punctuation label, and y[:,1] is the case label!

punctuation = {
    'O': 0,
    'COMMA': 1,
    'PERIOD': 2,
    'QUESTION': 3,
    'EXCLAMATION': 4,
}

punctuation_syms = ['', ',', '.', ' ?', ' !']

case = {
    'LOWER': 0,
    'UPPER': 1,
    'CAPITALIZE': 2,
    'OTHER': 3,
}


class Model(nn.Module):
    def __init__(self, flavor, device):
        super().__init__()
        self.bert = AutoModel.from_pretrained(flavor)
        # need a proper way of determining representation size
        size = self.bert.dim if hasattr(self.bert, 'dim') else self.bert.config.pooler_fc_size if hasattr(
            self.bert.config, 'pooler_fc_size') else self.bert.config.emb_dim if hasattr(self.bert.config,
                                                                                         'emb_dim') else self.bert.config.hidden_size
        self.punc = nn.Linear(size, 5)
        self.case = nn.Linear(size, 4)
        self.dropout = nn.Dropout(0.3)
        self.to(device)

    def forward(self, x):
        output = self.bert(x)
        representations = self.dropout(F.gelu(output['last_hidden_state']))
        punc = self.punc(representations)
        case = self.case(representations)
        return punc, case


def recase(token, label):
    if label == case['LOWER']:
        return token.lower()
    elif label == case['CAPITALIZE']:
        return token.lower().capitalize()
    elif label == case['UPPER']:
        return token.upper()
    else:
        return token


class CasePuncPredictor:
    def __init__(self, checkpoint_path, lang=default_config.lang, flavor=default_config.flavor,
                 device=default_config.device):
        loaded = torch.load(checkpoint_path, map_location='cpu')
        if 'config' in loaded:
            self.config = Config(**loaded['config'])
        else:
            self.config = Config(lang=lang, flavor=flavor, device=device)
        init(self.config)

        self.model = Model(self.config.flavor, self.config.device)
        self.model.load_state_dict(loaded['model_state_dict'])

        self.model.eval()
        self.model.to(self.config.device)

        self.rev_case = {b: a for a, b in case.items()}
        self.rev_punc = {b: a for a, b in punctuation.items()}

    def tokenize(self, text):
        return [self.config.cls_token] + self.config.tokenizer.tokenize(text) + [self.config.sep_token]

    def predict(self, tokens, getter=lambda x: x):
        max_length = self.config.max_length
        device = self.config.device
        if type(tokens) == str:
            tokens = self.tokenize(tokens)
        previous_label = punctuation['PERIOD']
        for start in range(0, len(tokens), max_length):
            instance = tokens[start: start + max_length]
            if type(getter(instance[0])) == str:
                ids = self.config.tokenizer.convert_tokens_to_ids(getter(token) for token in instance)
            else:
                ids = [getter(token) for token in instance]
            if len(ids) < max_length:
                ids += [0] * (max_length - len(ids))
            x = torch.tensor([ids]).long().to(device)
            y_scores1, y_scores2 = self.model(x)
            y_pred1 = torch.max(y_scores1, 2)[1]
            y_pred2 = torch.max(y_scores2, 2)[1]
            for i, id, token, punc_label, case_label in zip(range(len(instance)), ids, instance,
                                                            y_pred1[0].tolist()[:len(instance)],
                                                            y_pred2[0].tolist()[:len(instance)]):
                if id == self.config.cls_token_id or id == self.config.sep_token_id:
                    continue
                if previous_label != None and previous_label > 1:
                    if case_label in [case['LOWER'], case['OTHER']]:  # LOWER, OTHER
                        case_label = case['CAPITALIZE']
                if i + start == len(tokens) - 2 and punc_label == punctuation['O']:
                    punc_label = punctuation['PERIOD']
                yield (token, self.rev_case[case_label], self.rev_punc[punc_label])
                previous_label = punc_label

    def map_case_label(self, token, case_label):
        if token.endswith('</w>'):
            token = token[:-4]
        if token.startswith('##'):
            token = token[2:]
        return recase(token, case[case_label])

    def map_punc_label(self, token, punc_label):
        if token.endswith('</w>'):
            token = token[:-4]
        if token.startswith('##'):
            token = token[2:]
        return token + punctuation_syms[punctuation[punc_label]]


mapped_punctuation = {
    '.': 'PERIOD',
    '...': 'PERIOD',
    ',': 'COMMA',
    ';': 'COMMA',
    ':': 'COMMA',
    '(': 'COMMA',
    ')': 'COMMA',
    '?': 'QUESTION',
    '!': 'EXCLAMATION',
    '，': 'COMMA',
    '！': 'EXCLAMATION',
    '？': 'QUESTION',
    '；': 'COMMA',
    '：': 'COMMA',
    '（': 'COMMA',
    '(': 'COMMA',
    '）': 'COMMA',
    '［': 'COMMA',
    '］': 'COMMA',
    '【': 'COMMA',
    '】': 'COMMA',
    '└': 'COMMA',
    '└ ': 'COMMA',
    '_': 'O',
    '。': 'PERIOD',
    '、': 'COMMA',  # enumeration comma
    '、': 'COMMA',
    '…': 'PERIOD',
    '—': 'COMMA',
    '「': 'COMMA',
    '」': 'COMMA',
    '．': 'PERIOD',
    '《': 'O',
    '》': 'O',
    '，': 'COMMA',
    '“': 'O',
    '”': 'O',
    '"': 'O',
    '-': 'O',
    '-': 'O',
    '〉': 'COMMA',
    '〈': 'COMMA',
    '↑': 'O',
    '〔': 'COMMA',
    '〕': 'COMMA',
}


# modification of the wordpiece tokenizer to keep case information even if vocab is lower cased
# forked from https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/tokenization_bert.py

class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100, keep_case=True):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.keep_case = keep_case

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.
        For example, :obj:`input = "unaffable"` wil return as output :obj:`["un", "##aff", "##able"]`.
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.
        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in text.strip().split():
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    # optionaly lowercase substring before checking for inclusion in vocab
                    if (self.keep_case and substr.lower() in self.vocab) or (substr in self.vocab):
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def init(config):
    init_random(config.seed)
    config.tokenizer = tokenizer = BertTokenizer.from_pretrained(config.flavor, do_lower_case=False)
    config.tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token)
    config.pad_token_id = tokenizer.pad_token_id
    config.cls_token_id = tokenizer.cls_token_id
    config.cls_token = tokenizer.cls_token
    config.sep_token_id = tokenizer.sep_token_id
    config.sep_token = tokenizer.sep_token

    if not torch.cuda.is_available() and config.device == 'cuda':
        print('WARNING: reverting to cpu as cuda is not available', file=sys.stderr)
    config.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
