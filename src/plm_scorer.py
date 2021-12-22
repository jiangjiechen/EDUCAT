import torch
import numpy as np
import math
from transformers import (
    BertTokenizer, BertForMaskedLM,
    GPT2Tokenizer, GPT2LMHeadModel,
    RobertaTokenizer, RobertaForMaskedLM
)

MAX_LEN = 100


class BERT_Scorer:
    def __init__(self, pretrained='bert-base-uncased', device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if 'roberta' in pretrained:
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained)
            self.bert_model = RobertaForMaskedLM.from_pretrained(pretrained).to(self.device)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(pretrained)
            self.bert_model = BertForMaskedLM.from_pretrained(pretrained).to(self.device)
        self.mask_id = self.tokenizer.mask_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.special_tokens = set(self.tokenizer.all_special_tokens)
        self.special_ids = set(self.tokenizer.all_special_ids)

    def mask_score(self, sent_ids, mask_idx, mode=0, log_prob=False, maxlen=MAX_LEN):
        if maxlen:
            if mask_idx > maxlen:
                dist = int(mask_idx - maxlen / 2)
                sent_ids = sent_ids[dist:dist + maxlen]
                mask_idx -= dist
        sent_ids = np.concatenate([[self.cls_id], sent_ids, [self.sep_id]])
        mask_idx += 1
        if mode == 0:
            masked_sent_ids = np.array(sent_ids)
            masked_sent_ids[mask_idx] = self.mask_id
        else:
            masked_sent_ids = np.concatenate([sent_ids[:mask_idx], [self.mask_id], sent_ids[mask_idx:]])

        sent_len = len(masked_sent_ids)
        if maxlen is not None:
            sent_len = min(sent_len, maxlen + 2)
        input_tensor = torch.tensor(masked_sent_ids[:sent_len]).unsqueeze(0).to(self.device)
        outputs = self.bert_model(input_tensor)
        prediction_scores = outputs[0]
        if log_prob:
            log_pred_probs = torch.log_softmax(prediction_scores, dim=2)
            return log_pred_probs[0][mask_idx].detach().cpu().numpy()
        else:
            pred_probs = torch.softmax(prediction_scores, dim=2)
            return pred_probs[0][mask_idx].detach().cpu().numpy()

    def sent_score(self, line, maxlen=MAX_LEN, log_prob=False, ignore_idx=-1, ppl=False):
        if type(line) == str:
            sent_ids = self.tokenizer.encode(line.strip(), add_special_tokens=False)
        else:
            sent_ids = line
        if len(sent_ids) == 0:
            if log_prob:
                return -math.inf
            else:
                return 0.0
        sent_ids = np.concatenate([[self.cls_id], sent_ids, [self.sep_id]])
        sent_len = len(sent_ids)
        if maxlen is not None:
            sent_len = min(sent_len, maxlen + 2)
        input_tensor = torch.tensor((sent_len - 2) * [sent_ids[:sent_len]]).to(self.device)
        for idx in range(sent_len - 2):
            input_tensor[idx][idx + 1] = self.mask_id
        outputs = self.bert_model(input_tensor)
        prediction_scores = outputs[0]
        log_pred_probs = torch.log_softmax(prediction_scores, dim=2)
        sent_log_prob = 0.0
        for idx in range(sent_len - 2):
            tok_ind = idx + 1
            tok = sent_ids[tok_ind]
            if tok_ind != ignore_idx:
                sent_log_prob += log_pred_probs[idx][tok_ind][tok].item()
        if ppl:
            ppl_val = math.pow(math.exp(sent_log_prob), -1 / (sent_len - 1))
            return ppl_val
        elif log_prob:
            return sent_log_prob
        else:
            return math.exp(sent_log_prob)

    def multi_mask_score(self, sent_ids, mask_idx_set, mode=0, log_prob=False, maxlen=None, output_ind=None):
        if output_ind is None:
            output_ind = min(mask_idx_set)
        if maxlen:
            assert maxlen >= max(mask_idx_set)
        if mode == 0:
            masked_sent_ids = np.array(sent_ids)
            for mask_idx in mask_idx_set:
                masked_sent_ids[mask_idx] = self.mask_id
        else:
            raise NotImplementedError

        sent_len = len(masked_sent_ids)
        if maxlen is not None:
            sent_len = min(sent_len, maxlen)
        input_tensor = torch.tensor(masked_sent_ids[:sent_len]).unsqueeze(0).to(self.device)
        outputs = self.bert_model(input_tensor)
        prediction_scores = outputs[0]
        if log_prob:
            log_pred_probs = torch.log_softmax(prediction_scores, dim=2)
            return log_pred_probs[0][output_ind].detach().cpu().numpy()
        else:
            pred_probs = torch.softmax(prediction_scores, dim=2)
            return pred_probs[0][output_ind].detach().cpu().numpy()

    def id2sent(self, ids):
        return self.tokenizer.decode(ids)

    def close(self):
        pass


class GPT2_Scorer:
    def __init__(self, pretrained='gpt2', device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained)
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(pretrained, return_dict=True).to(self.device)
        self.pad_id = self.tokenizer.eos_token_id

    def sent_score(self, line, maxlen=None, log_prob=False, ppl=False):
        id_line = self.tokenizer.encode(line.strip(), add_special_tokens=False)
        id_line = [self.pad_id] + id_line + [self.pad_id]
        sent_len = len(id_line)
        if maxlen is not None:
            sent_len = min(sent_len, maxlen + 2)
        input_tensor = torch.tensor(id_line[:sent_len]).unsqueeze(0).to(self.device)
        try:
            outputs = self.gpt2_model(input_tensor, labels=input_tensor)
        except RuntimeError:
            print('RuntimeError in GPT2_Scorer.sent_score, input line:', line)
            return 0.0
        loss, prediction_scores = outputs[:2]
        # log_pred_probs = torch.log_softmax(prediction_scores[0], dim=-1)
        # sent_log_prob = 0.0
        # for idx in range(sent_len - 1):
        #     tok = id_line[idx+1]
        #     sent_log_prob += log_pred_probs[idx][tok].item()
        sent_log_prob = -loss.item() * (sent_len - 1)
        if ppl:
            ppl_val = math.pow(math.exp(sent_log_prob), -1 / (sent_len - 1))
            return ppl_val
        elif log_prob:
            return sent_log_prob, sent_len - 1
        else:
            return math.exp(sent_log_prob)

    def sent_score_batch(self, lines, maxlen=None, log_prob=False, batch_size=100):
        if not torch.cuda.is_available():
            batch_size = 16
        cnt = len(lines)
        sent_ids = [[self.pad_id] + self.tokenizer.encode(line.strip(), add_special_tokens=False) + [self.pad_id] for
                    line in lines]
        batches = [sent_ids[i:i + batch_size] if i + batch_size <= cnt else sent_ids[i:] for i in
                   range(0, cnt, batch_size)]
        results = []
        for batch_sent_ids in batches:
            bsize = len(batch_sent_ids)
            sent_max_len = max([len(s) for s in batch_sent_ids])
            if maxlen is not None:
                sent_max_len = min(sent_max_len, maxlen + 2)
            input_tensor = torch.zeros(bsize, sent_max_len, dtype=torch.long).fill_(self.pad_id)
            for idx, sent_id in enumerate(batch_sent_ids):
                sent_len = min(len(sent_id), sent_max_len)
                input_tensor[idx][:sent_len] = torch.tensor(sent_id[:sent_len])
            input_tensor = input_tensor.to(self.device)
            outputs = self.gpt2_model(input_tensor, labels=input_tensor)
            loss, prediction_scores = outputs[:2]
            for idx, sent_id in enumerate(batch_sent_ids):
                sent_len = min(len(sent_id), sent_max_len)
                pred_log_probs = torch.log_softmax(prediction_scores, dim=2)
                sent_log_prob = 0.0
                for tok_id in range(sent_len - 1):
                    sent_log_prob += pred_log_probs[idx][tok_id][sent_id[tok_id + 1]].item()
                if log_prob:
                    results.append(sent_log_prob)
                else:
                    results.append(math.exp(sent_log_prob))
        return results


if __name__ == '__main__':
    import os, cjjpy as cjj

    prefix = os.environ.get('PJ_HOME', cjj.AbsParentDir(__file__, '....'))
    bert_dir = f'{prefix}/models/bert'
    gpt2_dir = f'{prefix}/models/gpt2'
    bert_scorer = BERT_Scorer(bert_dir)
    # gpt2_scorer = GPT2_Scorer(gpt2_dir)

    # ids = bert_scorer.tokenizer.convert_tokens_to_ids('I eat apple'.split())
    # print(ids)
    # s = bert_scorer.mask_score(ids, 1)
    # print(s)

    s = bert_scorer.sent_score("Sally was riding her bike. "
                               # "She met friends who showed her a new bike path. "
                               "She was only on her bike for a few minutes before she got a flat tire."
                               "She quickly patched up her bike with some tape.")
    print(s)

    s = bert_scorer.sent_score("Sally was riding her bike. "
                               "She met friends who showed her a new bike path. "
                               # "She was only on her bike for a few minutes before she got a flat tire."
                               "She quickly patched up her bike with some tape.")
    print(s)