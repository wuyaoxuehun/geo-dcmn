'''
this is adaptation of original, for every choice with its own context
'''

import json
import logging

logger = logging.getLogger(__name__)


class SwagExample(object):
    """A single training/test example for the SWAG dataset."""

    def __init__(self,
                 ques_id,
                 context_sentence,
                 background,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label=None):
        self.ques_id = ques_id
        self.context_sentence = context_sentence
        self.background = background
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3
        ]
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 ques_id,
                 choices_features,
                 label
                 ):
        self.example_id = example_id
        self.ques_id = ques_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'doc_len': doc_len,
                'ques_len': ques_len,
                'option_len': option_len,
            }
            for input_ids, input_mask, segment_ids, doc_len, ques_len, option_len in choices_features
        ]
        self.label = label


def file_reader(paths, p_num=3):
    data = []
    for path in paths:
        with open(path, 'r', encoding='utf8') as f:
            data += json.load(f)
    examples = []
    for idx, instance in enumerate(data):
        contexts = []
        for opt in list('abcd'):
            paras = instance['paragraph_' + opt]
            paras_gold = [para['paragraph'] for para in paras[:p_num]]
            if paras_gold:
                contexts.append(''.join(paras_gold))
            else:
                contexts.append('。')

        background = instance['background']
        question_text = instance['question']

        if instance['answer'] not in list('ABCD'):
            print(instance['answer'])
            continue
        if instance.get('A', instance.get('a')) \
                and instance.get('A', instance.get('a')) \
                and instance.get('A', instance.get('a')) \
                and instance.get('A', instance.get('a')):
            examples.append(SwagExample(
                ques_id=instance['id'],
                context_sentence=contexts,
                background=background,
                start_ending=question_text,
                ending_0=instance.get('A', instance.get('a')),
                ending_1=instance.get('B', instance.get('b')),
                ending_2=instance.get('C', instance.get('c')),
                ending_3=instance.get('D', instance.get('d')),
                label=ord(instance['answer']) - ord('A')
            ))

    print(len(examples))
    return examples


def read_swag_examples(input_file, p_num):
    examples = file_reader(input_file, p_num)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    # outputs.
    features = []
    from tqdm import tqdm
    token_nums = []

    for example_index, example in tqdm(enumerate(examples)):

        choices_features = []
        # sentence_index = [0] * 100
        for ending_index, (context, ending) in enumerate(zip(example.context_sentence, example.endings)):
            context_tokens = tokenizer.tokenize(context)
            background_tokens = tokenizer.tokenize(example.background)
            start_ending_tokens = tokenizer.tokenize(example.start_ending)
            ending_tokens = tokenizer.tokenize(ending)
            token_nums.append(len(context_tokens) + len(background_tokens) + len(start_ending_tokens) + len(ending_tokens))

            # ending_tokens = start_ending_tokens + ending_tokens
            _truncate_seq_quadruple(context_tokens, background_tokens, start_ending_tokens, ending_tokens, max_seq_length - 3)

            start_ending_tokens = background_tokens + start_ending_tokens
            doc_len = len(context_tokens)
            option_len = len(ending_tokens)
            ques_len = len(start_ending_tokens)

            context_tokens_choice = context_tokens + start_ending_tokens
            try:
                encoded_tokens = tokenizer.encode_plus(context_tokens_choice, ending_tokens,
                                                       max_length=max_seq_length,
                                                       pad_to_max_length=True)
            except Exception as e:
                print(e)
                break
            input_ids = encoded_tokens['input_ids']
            input_mask = encoded_tokens['attention_mask']
            segment_ids = encoded_tokens['token_type_ids']

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            # assert (doc_len + ques_len + option_len) <= max_seq_length
            if (doc_len + ques_len + option_len) > max_seq_length:
                print(doc_len, ques_len, option_len, len(context_tokens_choice), len(ending_tokens))
                assert (doc_len + ques_len + option_len) <= max_seq_length - 3
            choices_features.append((input_ids, input_mask, segment_ids, doc_len, ques_len, option_len))
        else:
            label = example.label

            features.append(
                InputFeatures(
                    example_id=example_index,
                    ques_id=int(example.ques_id),
                    choices_features=choices_features,
                    label=label
                )
            )
    import numpy as np
    from scipy import stats
    print(f'average_tokens={np.mean(token_nums)}\n'
          f'tokens_std={np.std(token_nums, ddof=1)}\n'
          f'90%={np.percentile(token_nums, 90)}\n'
          f'{max_seq_length}={stats.percentileofscore(token_nums, max_seq_length)}\n')

    return features


def _truncate_seq_quadruple(tokens_a, tokens_b, tokens_c, tokens_d, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c) + len(tokens_d)

        if total_length <= max_length:
            break
        if len(tokens_a) >= max(len(tokens_b), len(tokens_c)):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_c):
            # only truncate the beginning of backgroud+question
            tokens_b.pop()
        elif len(tokens_c) >= len(tokens_d):
            tokens_c.pop()
        else:
            tokens_d.pop()


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]
