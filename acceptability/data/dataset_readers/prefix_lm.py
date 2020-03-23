import json
import logging
from typing import Dict

import numpy as np
from overrides import overrides

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, ArrayField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("prefix_lm")
class PrefixLmReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(
        self,
        prefix: str,
        suffix: str,
    ) -> Instance:
        # HuggingFace's tokenizers require leading whitespace.
        prefix_tokens = self._tokenizer.tokenize(' ' + prefix)
        suffix_tokens = self._tokenizer.tokenize(' ' + suffix)
        tokens = prefix_tokens + suffix_tokens
        eval_mask = np.array([0] * len(prefix_tokens) + [1] * len(suffix_tokens))
        metadata = {
            'prefix': [t.text for t in prefix_tokens],
            'suffix': [t.text for t in suffix_tokens],
        }
        fields = {
            'tokens': TextField(tokens, token_indexers=self._token_indexers),
            'eval_mask': ArrayField(eval_mask, dtype=bool),
            'metadata': MetadataField(metadata),
        }
        return Instance(fields)
