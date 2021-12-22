# -*- coding: utf-8 -*-


import time
import copy
import torch
import numpy as np
import cjjpy as cjj
import subprocess
import pseudo_multiproc_toolkit as pmp
from config import config as cfg
import os
from eval_client import metrics


def multirun(config):
    st = time.time()

    sliced_dataset = pmp.slice_dataset(config.data_path, config.pool_size)
    sliced_output = pmp.slice_filenames(config.output_file, config.pool_size)
    sliced_log_files = []

    children = []
    os.makedirs(os.path.dirname(config.output_file), exist_ok=True)
    for i, (data_path, output_file) in enumerate(zip(sliced_dataset, sliced_output)):
        if torch.cuda.is_available():
            device_id = i % torch.cuda.device_count()
            device = 'cuda:%d' % device_id
        else:
            device = 'cpu'
        config_i = copy.copy(config)
        config_i.data_path = data_path
        config_i.output_file = output_file
        config_i.device = device
        config_i.rank_0 = 1 if i == 0 else 0
        script = f'python3 counterfactual_rewrite.py {pmp.args_to_shell(config_i)}'
        log_file = cjj.ChangeFileFormat(output_file, f".log.{output_file.split('.')[-1]}")
        sliced_log_files.append(log_file)
        with open(log_file, 'w') as flog:
            child = subprocess.Popen(script, shell=True, stdout=flog, stderr=flog)
        children.append(child)
        print(f'* Starting process {i} by: {script}')

    while True:
        time.sleep(5)
        wait_flag = False
        for i, child in enumerate(children):
            if child.poll() is None:
                wait_flag = True
        if not wait_flag:
            break

    output_file = pmp.union_multiproc_files(sliced_output, overwrite=True)
    log_file = pmp.union_multiproc_files(sliced_log_files, overwrite=True)
    pmp.clean_multiproc_files(sliced_output)
    pmp.clean_multiproc_files(sliced_dataset)
    pmp.clean_multiproc_files(sliced_log_files)
    assert output_file == config.output_file

    results = metrics.evaluate(config.output_file, config.data_path, ['bleu', 'entailscore', 'bertscore'])
    output_json = metrics.assemble(config.output_file, config.data_path)
    msg = f'Multi EduCat completed in {cjj.TimeClock(time.time() - st)}: {output_json}. Results: {results}'
    print(msg)
    if config.lark:
        cjj.lark(msg)


if __name__ == "__main__":
    config = cfg()
    np.random.seed(config.seed)
    multirun(config)