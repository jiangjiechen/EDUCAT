import sys
import os
import json
from typing import List
import numpy as np
import rouge
import re
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from bert_score import score as bert_score
import nltk
import torch
import cjjpy as cjj
try:
    from nli_metrics.entail_score import EntailScore
except:
    from .nli_metrics.entail_score import EntailScore

nltk.download('punkt')


def _clean_text(txt):
    return txt.lower()


def read_lines(filename):
    lines = []
    with open(filename, "r") as f:
        for line in f:
            l = line.strip()
            if len(re.sub(r'[^\w\s]', '', l)) == 0:
                lines.append("")
            else:
                lines.append(l)
    return lines


def read_jsonl_lines(filename):
    with open(filename) as f:
        return [json.loads(line) for line in f]


def process_data(pred_endings_file, gold_file):
    pred_endings = read_lines(filename=pred_endings_file)
    gold_records = read_jsonl_lines(gold_file)

    instances = []
    for pe, record in zip(pred_endings, gold_records):
        instance = CFRInstance(
            original_context=record["initial"],
            cf_context=record["counterfactual"],
            predicted_ending=pe,
            original_ending=' '.join(record["original_ending"]),
            gold_cf_endings=[' '.join(_ge) for _ge in record["edited_endings"]]
        )
        instances.append(instance)
    return instances


class CFRInstance(object):

    def __init__(self,
                 original_context: str,
                 cf_context: str,
                 original_ending: str,
                 predicted_ending: str,
                 gold_cf_endings: List[str],
                 ):
        self.original_context = original_context
        self.cf_context = cf_context

        self.predicted_ending = predicted_ending
        self.original_ending = original_ending
        self.gold_cf_endings = gold_cf_endings

        self.original_context_tokens = word_tokenize(original_context)
        self.original_ending_tokens = word_tokenize(original_ending)
        self.cf_context_tokens = word_tokenize(cf_context)
        self.predicted_ending_tokens = word_tokenize(predicted_ending)
        self.gold_cf_endings_tokens = [word_tokenize(x) for x in gold_cf_endings]


def eval_nli(pred_endings_file, gold_file):
    model_path = f'{os.environ["PJ_HOME"]}/models/nli_metrics/roberta-large'
    nli_scorer = EntailScore(model_path)

    pred_endings = read_lines(filename=pred_endings_file)
    gold_records = read_jsonl_lines(gold_file)
    for js, pred in zip(gold_records, pred_endings):
        js['generated_endings'] = pred

    score, score_by_instance = nli_scorer.score_from_jsonl(gold_records)
    return {'entail_score': score, 'entail_score_by_instance': score_by_instance}


def eval_bleu(pred_endings_file, gold_file):
    instances = process_data(pred_endings_file, gold_file)

    references = []
    hypotheses = []
    for instance in instances:
        references.append(instance.gold_cf_endings_tokens)
        hypotheses.append(instance.predicted_ending_tokens)

    corpus_bleu_scores = corpus_bleu(
        references, hypotheses, smoothing_function=SmoothingFunction().method4
    )

    sentence_bleu_scores = []
    total_skipped = 0
    for r, h in zip(references, hypotheses):
        if len(h) == 0:
            sentence_bleu_scores.append(0)
            continue
        else:
            try:
                sentence_bleu_scores.append(
                    sentence_bleu(r, h, smoothing_function=SmoothingFunction().method4))
            except:
                sentence_bleu_scores.append(0.0)
                total_skipped += 1

    print("Total skipped = {}".format(total_skipped))

    metrics = {
        'corpus_bleu': corpus_bleu_scores,
        'mean_sentence_bleu': np.mean(sentence_bleu_scores),
        'sentence_bleu_by_instance': sentence_bleu_scores
    }
    return metrics


def eval_rouge(pred_endings_file, gold_file):
    instances = process_data(pred_endings_file, gold_file)

    references = []
    hypotheses = []

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                            max_n=4,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=True,
                            apply_best=False,
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=True)

    by_instance = []
    for instance in instances:
        _r = [_clean_text(g) for g in instance.gold_cf_endings]
        _h = _clean_text(instance.predicted_ending)
        references.append(_r)
        hypotheses.append(_h)
        try:
            by_instance.append(evaluator.get_scores(_h, _r))
        except:
            by_instance.append({})

    scores = evaluator.get_scores(hypotheses, references)
    return {'rouge_all': scores,
            'rouge_by_instance': by_instance
            }


def eval_bert_score(pred_endings_file, gold_file, bert_model="bert-base-uncased"):
    instances = process_data(pred_endings_file, gold_file)

    references = []
    hypotheses = []
    for instance in instances:
        for gold_end in instance.gold_cf_endings:
            # clean_reference = _clean_text(instance.original_context + ' ' + instance.original_ending)
            clean_reference = _clean_text(gold_end)
            clean_hypothesis = _clean_text(instance.predicted_ending)
            if len(clean_hypothesis) == 0:
                continue
            references.append(clean_reference)
            hypotheses.append(clean_hypothesis)

    P, R, F1 = bert_score(hypotheses, references, model_type=bert_model,
                          device='cuda' if torch.cuda.is_available() else 'cpu')
    return {
        "bert_score_P": P.mean().item(),
        "bert_score_R": R.mean().item(),
        "bert_score_F1": F1.mean().item(),
        "bert_score_P_by_instance": [float(f) for f in list(P.numpy())],
        "bert_score_R_by_instance": [float(f) for f in list(R.numpy())],
        "bert_score_F1_by_instance": [float(f) for f in list(F1.numpy())],
    }


def evaluate(pred_file, gold_file, metrics, bert_model='bert-base-uncased'):
    results = {}
    if 'bleu' in metrics:
        results.update(eval_bleu(pred_file, gold_file))
    if 'rouge' in metrics:
        results.update(eval_rouge(pred_file, gold_file))
        rscore = results.pop('rouge_all')
        results['rouge-l'] = rscore['rouge-l']
    if 'bertscore' in metrics:
        results.update(eval_bert_score(pred_file, gold_file, bert_model=bert_model))
    if 'entailscore' in metrics:
        results.update(eval_nli(pred_file, gold_file))
        if 'bleu' in metrics:
            bleu = results['corpus_bleu']
            ents = results['entail_score']
            results.update(
                {'hm': (2 * bleu * ents) / (bleu + ents)}
            )
    for k in list(results.keys()):
        if 'instance' in k:
            results.pop(k)
    return results


def assemble(pred_file, gold_file):
    output_file = cjj.ChangeFileFormat(pred_file, '.json')
    with open(pred_file) as fp, open(gold_file) as fg, open(output_file, 'w') as fo:
        preds = fp.readlines()
        gold = fg.readlines()
        for line1, line2 in zip(preds, gold):
            js = json.loads(line2)
            js['generated_endings'] = line1.strip()
            fo.write(json.dumps(js) + '\n')
    return output_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_txt', '-p', help='output txt file name. '
                                               'e.g python3 metrics.py -o demo.txt -i demo.json')
    parser.add_argument('--input_json', '-i', help='input json file name. '
                                              'e.g python3 metrics.py -o demo.txt -i demo.json')
    parser.add_argument('--only_assemble', '-a', action='store_true', help='Only run assemble')
    parser.add_argument('--metrics', '-m', nargs='+', default=['bertscore', 'bleu', 'rouge', 'entailscore'])
    args = parser.parse_args()

    if not args.only_assemble:
        results = evaluate(args.pred_txt, args.input_json, args.metrics)
        print(results)

    output_file = assemble(args.pred_txt, args.input_json)
    print("Written into ", output_file)
