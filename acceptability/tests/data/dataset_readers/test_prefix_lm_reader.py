from unittest import TestCase

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.vocabulary import Vocabulary

from acceptability.data.dataset_readers import PrefixLmReader


class TestPrefixLmReader(TestCase):
    def test_vanilla_text_to_instance(self):
        vocab = Vocabulary()
        vocab.add_tokens_to_namespace(
            [
                'This',
                'is',
                'a',
                'difficult',
                'test'
            ],
            namespace='tokens'
        )
        reader = PrefixLmReader()
        instance = reader.text_to_instance(prefix='This is a', suffix='difficult test')
        instance.index_fields(vocab)
        tensor_dict = instance.as_tensor_dict(instance.get_padding_lengths())

        tokens = instance['tokens']
        self.assertListEqual(
            [t.text for t in tokens],
            ['This', 'is', 'a', 'difficult', 'test']
        )

        token_ids = tensor_dict['tokens']['tokens']['tokens']
        self.assertListEqual(token_ids.tolist(), [2, 3, 4, 5, 6])

        eval_mask = tensor_dict['eval_mask']
        self.assertListEqual(eval_mask.tolist(), [0, 0, 0, 1, 1])

        metadata = tensor_dict['metadata']
        self.assertListEqual(metadata['prefix'],  ['This', 'is', 'a'])
        self.assertListEqual(metadata['suffix'],  ['difficult', 'test'])

    def test_gpt2_text_to_instance(self):
        tokenizer = PretrainedTransformerTokenizer(
            model_name='gpt2',
            add_special_tokens=False
        )
        token_indexers={
            'tokens': PretrainedTransformerIndexer(model_name='gpt2')
        }
        reader = PrefixLmReader(
            tokenizer=tokenizer,
            token_indexers=token_indexers
        )
        instance = reader.text_to_instance(prefix='This is a', suffix='difficult test')
        instance.index_fields(Vocabulary())
        tensor_dict = instance.as_tensor_dict(instance.get_padding_lengths())

        tokens = instance['tokens']
        self.assertListEqual(
            [t.text for t in tokens],
            ['ĠThis', 'Ġis', 'Ġa', 'Ġdifficult', 'Ġtest']
        )

        eval_mask = tensor_dict['eval_mask']
        self.assertListEqual(eval_mask.tolist(), [0, 0, 0, 1, 1])

        metadata = tensor_dict['metadata']
        self.assertListEqual(metadata['prefix'], ['ĠThis', 'Ġis', 'Ġa'])
        self.assertListEqual(metadata['suffix'], ['Ġdifficult', 'Ġtest'])
