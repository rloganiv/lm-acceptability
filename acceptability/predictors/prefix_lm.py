from typing import Dict

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.predictors import Predictor


@Predictor.register('prefix_lm')
class PrefixLmPredictor(Predictor):
    def predict(self, prefix: str, suffix: str) -> JsonDict:
        return self.predict_json({'prefix': prefix, 'suffix': suffix})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) ->  Instance:
        prefix = json_dict['prefix']
        suffix = json_dict['suffix']
        return self._dataset_reader.text_to_instance(prefix=prefix, suffix=suffix)
