from typing import Dict

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.predictors import Predictor


@Predictor.register('prefix')
class PrefixPredictor(Predictor):
    def predict(
        self,
        prefix_a: str,
        prefix_b: str,
        suffix: str
    ) -> JsonDict:
        return self.predict_json({
            'prefix_a': prefix_a,
            'prefix_b': prefix_b,
            'suffix': suffix
        })

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) ->  Instance:
        prefix_a = json_dict['prefix_a']
        prefix_b = json_dict['prefix_b']
        suffix = json_dict['suffix']
        return self._dataset_reader.text_to_instance(
            prefix_a=prefix_a,
            prefix_b=prefix_b,
            suffix=suffix
        )
