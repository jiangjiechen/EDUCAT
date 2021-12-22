# -*- coding: utf-8 -*-


import os
import logging
import time
import numpy as np
import cjjpy as cjj

from datareader import DataReader
from config import config as cfg
from utils import sample_from_dist, normalize, just_acc, mask_sentence, annealing
from plm_scorer import BERT_Scorer
from stationary import StaDist
from eval_client.metrics import evaluate
import traceback


class EduCat():
    def __init__(self, config, bert_scorer: BERT_Scorer, stationary: StaDist):
        self.output_dir = os.path.dirname(config.output_file)
        os.makedirs(self.output_dir, exist_ok=True)
        logging_level = logging.WARNING if config.pool_size > 1 else logging.INFO
        self.logger = cjj.init_logger(cjj.ChangeFileFormat(config.output_file, '.log'),
                                      log_file_level=logging_level, from_scratch=True)
        self.tokenizer = bert_scorer.tokenizer
        self.mlm = bert_scorer  # could be roberta
        self.stationary = stationary
        self.all_samples = []
        self.accepted_samples = []
        self.id2act = {0: 'replace', 1: 'insert', 2: 'delete'}

        self.search_size = config.search_size
        self.sample_time = config.sample_time
        self.causal_token_finding = config.causal_token_finding
        self.action_prob = config.action_prob
        self.action_threshold = config.action_threshold
        self.step_size = config.step_size
        self.restrict_constr = config.restrict_constr
        self.threshold = config.threshold
        self.drop_time = config.drop_time
        self.min_length = config.min_length
        self.action_threshold = config.action_threshold
        self.annealing_parameter = config.annealing_parameter
        self.just_acc_rate = config.just_acc_rate
        self.lark = config.lark
        self.coherence_type = config.coherence_type

    def _load_data(self, data_path):
        self.dataset = DataReader(self.tokenizer)
        self.dataset.load(data_path)

    def sample(self, data_path, output_file):
        self._load_data(data_path)
        st = time.time()
        self.all_samples = []
        self.accepted_samples = []
        f = open(output_file, 'w')
        for sen_id in range(len(self.dataset)):
            generated_endings = []
            counterfactual = self.dataset.data['counterfactual'][sen_id]
            premise = self.dataset.data['premise'][sen_id]
            initial = self.dataset.data['initial'][sen_id]
            prev_orig_endings = {'text': '', 'token_ids': ''}
            prev_edited_endings = {'text': '', 'token_ids': ''}

            for i, ending in enumerate(self.dataset.data['original_ending'][sen_id]):
                sampled_ending, prob = self.sample_educat(sen_id + 0.1 * i, premise, initial,
                                                          counterfactual, ending,
                                                          prev_orig_endings, prev_edited_endings)
                _, text = self.dataset.tokenize(sampled_ending)
                text = text[len(premise['text']) + len(counterfactual['text']) + len(prev_edited_endings['text']):]
                gen_ending = {
                    'text': text,
                    'token_ids': self.tokenizer.encode(text, add_special_tokens=False)
                }
                generated_endings.append(text.strip())
                prev_orig_endings = self.dataset.append_data(prev_orig_endings, ending)
                prev_edited_endings = self.dataset.append_data(prev_edited_endings, gen_ending)

            generated_endings = " ".join(generated_endings)
            self.logger.warning(f'{sen_id} [final] {generated_endings}')
            f.write(generated_endings + '\n')

        self.logger.warning(f'Accept rate: {round(len(self.accepted_samples) / len(self.all_samples), 6) * 100}%, '
                            f'{len(self.accepted_samples)}/{len(self.all_samples)} accepted.')
        cost_time = time.time() - st
        self.logger.warning(f'Cost {cjj.TimeClock(cost_time)}, '
                            f'{cjj.TimeClock(cost_time / len(self.dataset))} per case in average.')
        f.close()
        print(f'EduCat completed: {output_file}')

    def sample_educat(self, data_i, premise, initial, counterfactual, ending, prev_orig_endings, prev_edited_endings):
        '''
        :param data_i:
        :param premise:
        :param initial:
        :param counterfactual:
        :param ending: current original ending to be edited
        :param prev_orig_endings: previous original endings
        :param prev_edited_endings: previous generated endings
        :return:
        '''
        fixed_prefix = premise['text'] + counterfactual['text'] + prev_edited_endings['text']  # with blank
        fixed_token_ids = self.tokenizer.encode(fixed_prefix, add_special_tokens=False)

        input_text = fixed_prefix + ending['text']
        input_ids = fixed_token_ids + self.tokenizer.encode(ending['text'], add_special_tokens=False)

        old_prob = self.stationary.fluency(input_text)
        old_prob *= self.stationary.coherence_constraint(premise['text'],
                                                             initial['text'] + prev_orig_endings['text'],
                                                             counterfactual['text'] + prev_edited_endings['text'],
                                                             ending['text'])

        prev_inds = []
        outputs = []
        output_p = []  # output sentences
        sequence_length = len(ending['token_ids'])

        for iter in range(self.sample_time):
            gen_ending_token_ids = input_ids[len(fixed_token_ids):]

            if self.causal_token_finding:
                pos_set = self._find_position_causal(gen_ending_token_ids, prev_inds, premise['text'],
                                                     initial['text'] + prev_orig_endings['text'],
                                                     counterfactual['text'] + prev_edited_endings['text'],
                                                     self.step_size)
                action_set = [sample_from_dist(self.action_prob, size=None) for i in range(len(pos_set))]
            else:
                pos_set = self._find_position_random(sequence_length, prev_inds, self.step_size)
                action_set = [sample_from_dist(self.action_prob, size=None) for i in range(len(pos_set))]

            # last token (i.e. period) only support replacement (i.e. action:0)
            for i, pos in enumerate(pos_set):
                if pos == sequence_length - 1:
                    action_set[i] = 0

            # add prefix: premise + counterfactual, which will not be edited.
            pos_set = [x + len(fixed_token_ids) for x in pos_set]
            masked_sent, adjusted_pos_set = mask_sentence(input_ids, pos_set, action_set,
                                                          self.tokenizer.mask_token_id)
            prev_inds = pos_set

            proposal_prob = 1.0  # Q(x'|x)
            proposal_prob_reverse = 1.0  # Q(x|x')
            input_ids_tmp = np.array(masked_sent)  # copy
            sequence_length_tmp = sequence_length

            for step_i in range(len(pos_set)):
                ind = adjusted_pos_set[step_i]
                ind_old = pos_set[step_i]
                action = action_set[step_i]

                if sequence_length_tmp < self.min_length and action == 2:
                    # Avoiding over deleting
                    action = sample_from_dist([0.8, 0.2], size=None)

                # Only compute sentence level score upon completing, in case step_size > 1.
                if self.restrict_constr:
                    use_constr = step_i == len(pos_set) - 1
                else:
                    use_constr = True

                if action == 0:
                    input_ids_tmp, proposal_prob, proposal_prob_reverse = \
                        self._replace(input_ids, input_ids_tmp, ind, ind_old,
                                      proposal_prob, proposal_prob_reverse, use_constr)
                elif action == 1:
                    input_ids_tmp, proposal_prob, proposal_prob_reverse = \
                        self._insert(input_ids, input_ids_tmp, ind, ind_old,
                                     proposal_prob, proposal_prob_reverse, use_constr)
                    sequence_length_tmp += 1
                elif action == 2:
                    input_ids_tmp, proposal_prob, proposal_prob_reverse = \
                        self._delete(input_ids, input_ids_tmp, ind, ind_old,
                                     proposal_prob, proposal_prob_reverse, use_constr)
                    sequence_length_tmp -= 1
                else:
                    pass

            input_text_tmp = self.tokenizer.decode(input_ids_tmp)
            ending_temp = input_text_tmp[len(fixed_prefix):]
            new_prob = self.stationary.fluency(input_text_tmp)
            new_prob *= self.stationary.coherence_constraint(premise['text'],
                                                                 initial['text'] + prev_orig_endings['text'],
                                                                 counterfactual['text'] + prev_edited_endings['text'],
                                                                 ending_temp)

            self.all_samples.append({
                'sampled_text': input_text_tmp,
                'new_prob': new_prob
            })

            if proposal_prob == 0.0 or old_prob == 0.0:
                alpha_star = 1.0
            else:
                if self.annealing_parameter < 1:
                    alpha_star = (proposal_prob_reverse / proposal_prob) * (new_prob / old_prob) ** annealing(
                        parameter=self.annealing_parameter,
                        num_iter=iter)
                else:
                    alpha_star = (proposal_prob_reverse / proposal_prob) * (new_prob / old_prob)

            alpha = min(1, alpha_star)

            for i in range(self.step_size):
                alter_word = self.tokenizer.decode([input_ids[pos_set[i]]]).strip()
                self.logger.info(f"No.{data_i}: [{self.id2act[action_set[i]]} {alter_word}] {input_text_tmp}")
            self.logger.info(f"[{iter}/{self.sample_time}] "
                             f"alpha: {alpha}, "
                             f"old_prob: {old_prob}, "
                             f"proposal_prob: {proposal_prob}, "
                             f"new_prob: {new_prob}, "
                             f"proposal_prob_reverse: {proposal_prob_reverse}")

            if sample_from_dist([alpha, 1 - alpha], size=None) == 0 \
                    and (new_prob > old_prob * self.threshold or just_acc(self.just_acc_rate) == 0):

                if self.tokenizer.decode(input_ids_tmp) != self.tokenizer.decode(input_ids):
                    self.logger.warning(f'No.{data_i}: [accepted] {input_text_tmp}')
                    self.accepted_samples.append({
                        'sampled_text': input_text_tmp,
                        'new_prob': new_prob
                    })

                    if iter > self.drop_time:
                        if self.tokenizer.decode(input_ids_tmp) not in output_p:
                            rank_prob = new_prob
                            outputs.append({
                                'sampled_text': input_text_tmp,
                                'new_prob': rank_prob
                            })
                        if outputs != []:
                            output_p.append(input_text_tmp)

                input_ids = input_ids_tmp
                sequence_length = sequence_length_tmp
                old_prob = new_prob

        outputs_s = []
        for num in range(self.min_length, 0, -1):
            outputs_s = [x for x in outputs if len(x['sampled_text'].split()) >= num]
            if outputs_s != []:
                break
        if outputs_s == []:
            outputs_s = [{
                'sampled_text': self.tokenizer.decode(input_ids),
                'new_prob': 0.
            }]
        outputs_s = sorted(outputs_s, key=lambda x: x['new_prob'])[::-1]  # rank samples after burn_in_time 
        return outputs_s[0]['sampled_text'], outputs_s[0]['new_prob']

    def _propose(self, input_ids_tmp, ind, mode=0):
        '''
        proposal with masked input ids
        :param input_ids_tmp:
        :param ind:
        :return: probability of the masked token
        '''
        prob_mask = self.mlm.mask_score(input_ids_tmp, ind, mode)
        return prob_mask

    def _replace(self, input_ids, input_ids_tmp, ind, ind_old,
                 proposal_prob, proposal_prob_reverse, use_constr):
        # action: 0
        prob_mask = self._propose(input_ids_tmp, ind)
        input_candidate, prob_candidate, reverse_candidate_idx = \
            self.generate_candidate_input_with_mask(input_ids_tmp,
                                                    None,
                                                    ind,
                                                    prob_mask,
                                                    self.search_size,
                                                    old_tok=input_ids[ind_old],
                                                    mode=0)

        prob_candidate_norm = normalize(prob_candidate)
        prob_candidate_ind = sample_from_dist(prob_candidate_norm, size=None)
        input_ids_tmp = input_candidate[prob_candidate_ind]  # changed
        proposal_prob *= prob_candidate_norm[prob_candidate_ind]  # Q(x'|x)
        proposal_prob_reverse *= prob_candidate_norm[reverse_candidate_idx]  # Q(x|x')
        # sequence_length_tmp += 0
        self.logger.info(f"> Replace: "
                         f"g(x'|x) = {prob_candidate_norm[prob_candidate_ind]}, "
                         f"g(x|x') = {prob_candidate_norm[reverse_candidate_idx]}")
        return input_ids_tmp, proposal_prob, proposal_prob_reverse

    def _insert(self, input_ids, input_ids_tmp, ind, ind_old,
                proposal_prob, proposal_prob_reverse, use_constr):
        # action: 1
        prob_mask = self._propose(input_ids_tmp, ind, mode=0)
        input_candidate, prob_candidate, reverse_candidate_idx = \
            self.generate_candidate_input_with_mask(input_ids_tmp,
                                                    None,
                                                    ind,
                                                    prob_mask,
                                                    self.search_size,
                                                    mode=1,
                                                    old_tok=input_ids[ind_old])

        prob_candidate_norm = normalize(prob_candidate)
        prob_candidate_ind = sample_from_dist(prob_candidate_norm, size=None)
        input_ids_tmp = input_candidate[prob_candidate_ind]
        proposal_prob *= prob_candidate_norm[prob_candidate_ind]  # Q(x'|x)
        proposal_prob_reverse *= 1.0  # Q(x|x'), reverse action is deleting
        # sequence_length_tmp += 1
        self.logger.info(f"> Insert: "
                         f"g(x'|x) = {prob_candidate_norm[prob_candidate_ind]}, "
                         f"g(x|x') = 1.0")
        return input_ids_tmp, proposal_prob, proposal_prob_reverse

    def _delete(self, input_ids, input_ids_tmp, ind, ind_old,
                proposal_prob, proposal_prob_reverse, use_constr):
        # action: 2
        input_ids_for_del = np.concatenate([input_ids_tmp[:ind],
                                            [self.tokenizer.mask_token_id],
                                            input_ids_tmp[ind:]])
        # add mask, for evaluating reverse probability
        prob_mask = self._propose(input_ids_for_del, ind, mode=0)
        input_candidate, prob_candidate, reverse_candidate_idx = \
            self.generate_candidate_input_with_mask(input_ids_for_del,
                                                    None,
                                                    ind,
                                                    prob_mask,
                                                    self.search_size,
                                                    mode=0,
                                                    old_tok=input_ids[ind_old])

        prob_candidate_norm = normalize(prob_candidate)

        proposal_prob *= 1.0  # Q(x'|x)
        proposal_prob_reverse *= prob_candidate_norm[reverse_candidate_idx]  # Q(x|x'), reverse action is inserting
        # sequence_length_tmp -= 1
        self.logger.info(f"> Delete: "
                         f"g(x'|x) = 1.0, "
                         f"g(x|x') = {prob_candidate_norm[reverse_candidate_idx]}")
        return input_ids_tmp, proposal_prob, proposal_prob_reverse

    def generate_candidate_input_with_mask(self, input_ids, sequence_length, ind, prob,
                                           search_size, old_tok=-1, mode=0):
        '''
        get top-k candidates. difference:
        1. index is already adjusted
        2. input is already masked
        3. for inserting: mask is already added, only consider replace
        4. when ind == 0, constraint on start word
        5. return new normalized prob

        :param input_ids: token ids with mask
        :param sequence_length:
        :param ind: mask position
        :param prob: mlm prob
        :param search_size: top k
        :param old_tok: masked token id
        :param mode: 0
        :return:
            input_candidate: filled token ids [] * search_size
            prob_candidate: mlm prob for filled candidate tokens [] * search_size
            reverse_candidate_idx: id of raw sentence in prob_candidate
        '''

        prob_tmp = np.array(prob)
        # if ind == 0:
        # 	for i in range(len(prob_tmp)):
        # 		if i not in start_word_ids:
        # 			prob_tmp[i] = -np.inf
        # 	search_size = len(start_word_ids)
        sorted_prob_tmp = np.argsort(prob_tmp)
        _tok_candidate = sorted_prob_tmp[-search_size:]

        # eliminate special tokens
        tok_candidate = []
        del_cnt = 0
        for tok in _tok_candidate:
            if tok not in self.mlm.special_ids:
                tok_candidate.append(tok)
            else:
                # fill with new token id
                while True:
                    del_cnt += 1
                    new_tok = sorted_prob_tmp[-(search_size + del_cnt)]
                    if new_tok not in self.mlm.special_ids:
                        tok_candidate.append(new_tok)
                        break

        assert search_size == len(tok_candidate), (search_size, len(tok_candidate))

        input_candidate = np.array([input_ids] * search_size)
        for i in range(search_size):
            input_candidate[i][ind] = tok_candidate[i]

        # dealing with reverse proposal
        reverse_candidate_idx = -1
        if mode == 0:
            if old_tok in tok_candidate:
                for reverse_candidate_idx in range(len(tok_candidate)):
                    if old_tok == tok_candidate[reverse_candidate_idx]:
                        break
            if reverse_candidate_idx < 0 or reverse_candidate_idx >= len(input_candidate):
                reverse_candidate_idx = len(input_candidate)
                reverse_candidate = np.array(input_ids)
                reverse_candidate[ind] = old_tok
                input_candidate = np.concatenate([input_candidate, [reverse_candidate]], axis=0)
                tok_candidate = np.concatenate([tok_candidate, [old_tok]])

        prob_candidate = np.array([prob[tok] for tok in tok_candidate])

        return input_candidate, prob_candidate, reverse_candidate_idx

    def _find_position_random(self, seq_len, prev_inds, step_size=1):
        '''
        :param seq_len: sequence length
        :param prev_inds: already found positions
        :param step_size: number of positions found per sample time
        :return: [1, 2, 3]
        '''
        candidates = []
        if step_size >= seq_len:
            self.logger.warning('Warning: too short sequence length')
            if len(prev_inds) >= seq_len:
                candidates = list(range(seq_len))
            else:
                for ind in range(seq_len):
                    if ind not in prev_inds:
                        candidates.append(ind)
            np.random.shuffle(candidates)
            return [candidates[0]]

        if step_size + len(prev_inds) >= seq_len:
            candidates = list(range(seq_len))
        else:
            for ind in range(seq_len):
                if ind not in prev_inds:
                    candidates.append(ind)
        np.random.shuffle(candidates)
        pos_set = candidates[:step_size]
        pos_set = sorted(pos_set, reverse=True)  # descending order, for avoiding conflicts
        return pos_set

    def _find_position_causal(self, gen_ending_token_ids, prev_inds, premise,
                              initial, counterfactual, step_size=1):
        seq_len = len(gen_ending_token_ids)
        candidates = []
        if step_size >= seq_len:
            self.logger.warning('Warning: too short sequence length')
            if len(prev_inds) >= seq_len:
                candidates = list(range(seq_len))
            else:
                for ind in range(seq_len):
                    if ind not in prev_inds:
                        candidates.append(ind)
            np.random.shuffle(candidates)
            return [candidates[0]]

        tokenized_ending = self.tokenizer.convert_ids_to_tokens(gen_ending_token_ids)
        importance = self.stationary.cause_effect_position_finder(premise, initial, counterfactual,
                                                                  tokenized_ending, normalize=True)

        assert len(importance) == len(gen_ending_token_ids), (len(importance), len(gen_ending_token_ids))
        pos_set = sample_from_dist(importance, size=step_size, replace=False)
        return pos_set


if __name__ == '__main__':
    cfg = cfg()
    np.random.seed(cfg.seed)
    try:
        bert_scorer = BERT_Scorer(cfg.mlm_path, cfg.device)
        stationary = StaDist(cfg.gpt2_path, cfg.coherence_type, cfg.constraint_model_path, cfg.device)
        educat = EduCat(cfg, bert_scorer=bert_scorer, stationary=stationary)
        educat.sample(cfg.data_path, cfg.output_file)
        if cfg.pool_size == 1:
            results = evaluate(cfg.output_file, cfg.data_path, ['entailscore', 'bleu', 'bertscore'])
            print(results)
    except Exception as e:
        exc = traceback.format_exc()
        msg = f'[Error]: {e}.\n[Traceback]: {exc}'
        print(msg)
        if cfg.lark and cfg.rank_0: cjj.lark(msg)
