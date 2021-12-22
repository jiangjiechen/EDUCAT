# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F
from plm_scorer import GPT2_Scorer
import numpy as np
from eval_client.nli_metrics.entail_score import EntailScore


class StaDist():
    def __init__(self, gpt2_path, coherence_type, constraint_model_path=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.gpt2_scorer = GPT2_Scorer(gpt2_path, self.device)
        self.tokenizer = self.gpt2_scorer.tokenizer
        self.coherence_type = coherence_type
        if coherence_type == 'ents':
            self.ents = EntailScore(constraint_model_path, device=device)

    def fluency(self, text: str, **kwargs):
        return self.gpt2_scorer.sent_score(text, **kwargs)

    def cause_effect_position_finder(self, premise: str, initial: str, counterfactual: str,
                                     original_ending_tokens: list, normalize=False):
        """
        Return tokenized ending by GPT-2 & importance
        This is important to maintain the tokenization coherence between GPT-2 & RoBERTa
        return: {'token': score} or [score, score]
        """
        init_prob = self._sample_prob(premise + initial, original_ending_tokens)
        count_prob = self._sample_prob(premise + counterfactual, original_ending_tokens)
        assert len(init_prob) == len(count_prob), "the length of init_prob and count_prob should be equal!"

        importance = []
        for idx in range(len(original_ending_tokens)):
            importance.append(init_prob[idx] / count_prob[idx])

        if normalize:
            importance = F.softmax(torch.tensor(importance), dim=-1).numpy()

        return importance

    def _sample_prob(self, prefix, postfix_tokens):
        """
        prefix: premise + initial / premise + counter
        postfix: original ending
        return: list of probability for each token in orig_ending
        """
        start_ids = self.tokenizer.encode_plus(prefix, return_tensors="pt")["input_ids"].squeeze().to(self.device)
        end_ids = self.tokenizer.convert_tokens_to_ids(postfix_tokens)
        end_ids = torch.tensor(end_ids, dtype=torch.int).to(self.device)

        opt = self.gpt2_scorer.gpt2_model(torch.cat((start_ids, end_ids), dim=0), return_dict=True)
        probs = F.softmax(opt.logits[:, :], dim=1)  # probs now is of shape seq_len * vocab_size

        # opt.logits of shape (1, sen_len, vocab_size)
        res = [0] * len(end_ids)
        for i in range(len(start_ids), len(start_ids) + len(end_ids)):
            cur_id = end_ids[i - len(start_ids)].item()
            res[i - len(start_ids)] = probs[i - 1, cur_id].item()

        return res

    def keyword_constraint(self, s, keywords):
        for k in keywords:
            if k not in s:
                return 0.
        return 1.

    def coherence_constraint(self, premise, initial, counterfactual, edited_ending):
        if self.coherence_type == 'ents':
            return self._entail_constraint(premise, initial, counterfactual, edited_ending)
        elif self.coherence_type == 'lm':
            return self._lm_constraint(premise, initial, counterfactual, edited_ending)
        else:
            return 1

    def _lm_constraint(self, premise, initial, counterfactual, edited_ending):
        init_prob = self._get_sent_prob(premise + initial)
        count_prob = self._get_sent_prob(premise + counterfactual)
        orig_prob = self._get_sent_prob(premise + initial + edited_ending) / init_prob
        edit_prob = self._get_sent_prob(premise + counterfactual + edited_ending) / count_prob

        ratio = torch.tensor(edit_prob / orig_prob)
        return F.sigmoid(ratio).item()

    def _entail_constraint(self, premise, initial, counterfactual, edited_ending):
        ent_initial = self.ents.score_from_list(premise + initial, edited_ending)[0]
        ent_counterfactual = self.ents.score_from_list(premise + counterfactual, edited_ending)[0]
        ratio = ent_counterfactual / ent_initial
        return ratio

    def _get_sent_prob(self, s: str):
        input_ids = self.tokenizer.encode_plus(self.tokenizer.eos_token + " " + s.strip(),
                                               return_tensors="pt")["input_ids"].to(self.device)
        opt = self.gpt2_scorer.gpt2_model(input_ids, labels=input_ids)
        return np.exp(-1 * opt.loss.item() * (len(input_ids[0]) - 1))
