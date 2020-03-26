import json
import logging
from typing import Dict, Iterable

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, ArrayField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
import numpy as np
from overrides import overrides

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
    def _read(
        self,
        file_path: str,
    ) -> Iterable[Instance]:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                yield self.text_to_instance(
                    prefix_a=data['prefix_a'],
                    prefix_b=data['prefix_b'],
                    suffix=data['suffix']
                )

    @overrides
    def text_to_instance(
        self,
        prefix_a: str,
        prefix_b: str,
        suffix: str,
    ) -> Instance:

        # HuggingFace's tokenizers require leading whitespace.
        prefix_a_tokens = self._tokenizer.tokenize(' ' + prefix_a)
        prefix_b_tokens = self._tokenizer.tokenize(' ' + prefix_b)
        suffix_tokens = self._tokenizer.tokenize(' ' + suffix)

        tokens_a = prefix_a_tokens + suffix_tokens
        tokens_b = prefix_b_tokens + suffix_tokens

        eval_mask_a = np.array([0] * len(prefix_a_tokens) + [1] * len(suffix_tokens))
        eval_mask_b = np.array([0] * len(prefix_b_tokens) + [1] * len(suffix_tokens))

        metadata = {
            'prefix_a': [t.text for t in prefix_a_tokens],
            'prefix_b': [t.text for t in prefix_b_tokens],
            'suffix': [t.text for t in suffix_tokens],
        }

        fields = {
            'tokens_a': TextField(tokens_a, token_indexers=self._token_indexers),
            'tokens_b': TextField(tokens_b, token_indexers=self._token_indexers),
            'eval_mask_a': ArrayField(eval_mask_a, dtype=bool),
            'eval_mask_b': ArrayField(eval_mask_b, dtype=bool),
            'metadata': MetadataField(metadata),
        }

        return Instance(fields)
