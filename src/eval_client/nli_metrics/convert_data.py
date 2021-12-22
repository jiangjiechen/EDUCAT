# -*- coding:utf-8  -*-


import os
import logging
import random
try:
    from .src.utils import init_logger, read_json_lines, save_json_lines
except:
    from src.utils import init_logger, read_json_lines, save_json_lines


logger = logging.getLogger(__name__)
data_dir = os.path.join(os.environ['PJ_HOME'], 'data/TimeTravel')


def convert(filename):
    data = []
    for line in read_json_lines(filename):
        premise = line['premise']
        initial = line['initial']
        original_ending = line['original_ending']
        original_ending = original_ending if isinstance(original_ending, str) else ' '.join(original_ending)
        counterfactual = line['counterfactual']
        if line.get('edited_ending'):
            edited_ending = ' '.join(line['edited_ending'])
        else:
            edited_ending = ' '.join(line['edited_endings'][0])
        # entail: 2, refute: 0
        data.append({'text_a': premise + ' ' + initial, 'text_b': original_ending, 'label': 2})
        data.append({'text_a': premise + ' ' + counterfactual, 'text_b': original_ending, 'label': 0})
        data.append({'text_a': premise + ' ' + initial, 'text_b': edited_ending, 'label': 0})
        data.append({'text_a': premise + ' ' + counterfactual, 'text_b': edited_ending, 'label': 2})
    logger.info('data size: {}'.format(len(data)))
    return data


def main():
    init_logger(logging.INFO)
    for role in ['train_supervised_large', 'dev_data', 'test_data']:
        data = convert(f'{data_dir}/{role}.json')
        if 'dev' in role:
            rrole = 'eval'
        elif 'test' in role:
            rrole = 'test'
        else:
            random.shuffle(data)
            rrole = 'train'
        os.makedirs(f'{data_dir}/nli_metrics', exist_ok=True)
        save_json_lines(data, f'{data_dir}/nli_metrics/data_{rrole}.json')


if __name__ == '__main__':
    main()
