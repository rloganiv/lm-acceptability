from unittest import TestCase

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

        # Check that default tokenizer works as expected
        tokens = instance['tokens']
        self.assertListEqual([t.text for t in tokens], ['This', 'is', 'a', 'difficult', 'test'])

        # Check that default token_indexer works as expected
        token_ids = tensor_dict['tokens']['tokens']['tokens']
        self.assertListEqual(token_ids.tolist(), [2, 3, 4, 5, 6])

        # Check that eval mask works as expected
        eval_mask = tensor_dict['eval_mask']
        self.assertListEqual(eval_mask.tolist(), [0, 0, 0, 1, 1])
