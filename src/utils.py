
import numpy as np
import pandas as pd
import numpy as np
import torch
from typing import Dict, List
from dataclasses import dataclass


def _get_performance(ranks):
    ranks = np.array(ranks, dtype=np.float32)
    out = dict()
    out['mr'] = ranks.mean(axis=0)
    out['mrr'] = (1. / ranks).mean(axis=0)
    out['hit1'] = np.sum(ranks == 1, axis=0) / len(ranks)
    out['hit3'] = np.sum(ranks <= 3, axis=0) / len(ranks)
    out['hit10'] = np.sum(ranks <= 10, axis=0) / len(ranks)
    return out


def get_performance(model, tail_ranks, head_ranks):
    tail_out = _get_performance(tail_ranks)
    head_out = _get_performance(head_ranks)
    mr = np.array([tail_out['mr'], head_out['mr']])
    mrr = np.array([tail_out['mrr'], head_out['mrr']])
    hit1 = np.array([tail_out['hit1'], head_out['hit1']])
    hit3 = np.array([tail_out['hit3'], head_out['hit3']])
    hit10 = np.array([tail_out['hit10'], head_out['hit10']])
    perf = {'mrr': mrr, 'mr': mr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    perf = pd.DataFrame(perf, index=['tail ranking', 'head ranking'])
    perf.loc['mean ranking'] = perf.mean(axis=0)
    for hit in ['hit@1', 'hit@3', 'hit@5', 'hit@10']:
        if hit in list(perf.columns):
            perf[hit] = perf[hit].apply(lambda x: '%.2f%%' % (x * 100))
    return perf


class RunEval:
    def __init__(self, configs, model, tokenizer, ent_name_list, target_ids, trie, ext_neigs_0, ext_neigs_2):
        self.configs = configs
        self.ent_name_list = ent_name_list
        self.target_ids = target_ids
        self.configs = configs
        self.model = model
        self.tokenizer = tokenizer
        self.trie = trie
        self.ext_neigs_0 = ext_neigs_0
        self.ext_neigs_2 = ext_neigs_2

    @torch.no_grad()
    def validation_step(self, batched_data, dataset_idx):
        input_ids = batched_data['input_ids'].to('cuda')
        attention_mask = batched_data['attention_mask'].to('cuda')
        labels = batched_data['labels']
        labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)
        neighboors_embeddings=batched_data['neighboors_embeddings'].to('cuda')
        neighboors_embeddings_mask=batched_data['neighboors_embeddings_mask'].to('cuda')
        target_ent_embeddings=batched_data['target_ent_embeddings'].to('cuda')
        triple_id = batched_data['triplet'].numpy()

        old_seqs = []
        ranks = torch.randint(self.configs.num_beams + 1, self.configs.n_ent, (len(labels),))
        for i in range(self.configs.num_beams):
          outputs = self.model.generate(
              input_ids=input_ids,
              attention_mask=attention_mask,
              return_dict_in_generate=True,
              max_length=512,
              prefix_allowed_tokens_fn=lambda batch_idx, m_input_ids: self._next_candidate(batch_idx, m_input_ids, triple_id, dataset_idx, old_seqs),
              neighboors_embeddings=neighboors_embeddings,
              neighboors_embeddings_mask=neighboors_embeddings_mask,
              target_ent_embeddings=target_ent_embeddings,
          )
          pred = outputs.sequences.cpu()
          old_seqs.append(pred)
          pred = pred[:,1:]
          if pred.shape[1] > labels.shape[1]:
            pred = pred[:,:labels.shape[1]]
          else:
            cut_labels = labels[:,:pred.shape[1]]
          cut_labels = labels
          seq_match = (pred==cut_labels).all(1)
          new_ranks = torch.where(~seq_match, ranks, i+1)
          ranks = torch.min(ranks, new_ranks)

        ranks = ranks.tolist()
        out = {'ranks': ranks}
        return out


    def _next_candidate(self, batch_idx, input_ids, triple_id, dataset_idx, old_seqs=None):
        input_ids = input_ids.cpu()
        if input_ids[-1] == 0 and len(input_ids) != 1:
            return [0]
        pred_ids = self.target_ids[triple_id[batch_idx][dataset_idx]]
        pred_id = int(pred_ids[len(input_ids)])
        ext_neigs = self.ext_neigs_2 if dataset_idx == 0 else self.ext_neigs_0
        all_gt_ids = torch.cat(get_neigs(triple_id[batch_idx][2-dataset_idx], triple_id[batch_idx][1]), ext_neigs)

        all_gt_seq = torch.index_select(self.target_ids, 0, all_gt_ids)
        all_gt_seq_mask = (all_gt_seq[:, :len(input_ids)]==input_ids).all(1)
        all_gt_seq_tokens = all_gt_seq[:, len(input_ids)][all_gt_seq_mask]
        if len(old_seqs) > 0:
          old_seq = torch.nn.utils.rnn.pad_sequence([x[batch_idx] for x in old_seqs], batch_first=True, padding_value=0)
          if old_seq.shape[1] > len(input_ids):
            old_seq_mask = (old_seq[:, :len(input_ids)]==input_ids).all(1)
            old_seq_tokens = old_seq[:, len(input_ids)][old_seq_mask]
          else:
            old_seq_tokens = torch.tensor([], dtype=torch.int64)
        else:
          old_seq_tokens = torch.tensor([], dtype=torch.int64)
        all_gt_seq_tokens = set(torch.cat([all_gt_seq_tokens, old_seq_tokens]).tolist())
        pred_id = int(pred_ids[len(input_ids)])
        next_tokens = set(self.trie.get(input_ids.tolist())).difference(all_gt_seq_tokens)
        if pred_id in all_gt_seq_tokens:
          next_tokens.add(pred_id)
        if len(next_tokens) == 0:
          return [0]
        return list(next_tokens)

    def validation_epoch_end(self, outs):
        pred_tail_out, pred_head_out = outs
        agg_tail_out, agg_head_out = dict(), dict()
        for out in pred_tail_out:
            for key, value in out.items():
                if key in agg_tail_out:
                    agg_tail_out[key] += value
                else:
                    agg_tail_out[key] = value
        for out in pred_head_out:
            for key, value in out.items():
                if key in agg_head_out:
                    agg_head_out[key] += value
                else:
                    agg_head_out[key] = value
        tail_ranks, head_ranks = agg_tail_out['ranks'], agg_head_out['ranks']
        del agg_tail_out['ranks']
        del agg_head_out['ranks']
        perf = get_performance(self, head_ranks, tail_ranks)
        return perf

# get all ground truth
def get_neigs(ent_id, rel_id, ext_get_neigs):
  n_train = ext_get_neigs['train'].__getitem__(ent_id, rel_id)
  n_valid = ext_get_neigs['valid'].__getitem__(ent_id, rel_id)
  n_test = ext_get_neigs['test'].__getitem__(ent_id, rel_id)
  return [n_train, n_valid, n_test]


class DataCollatorForSeq2Seq:
    model= None
    padding= True
    max_length= None
    pad_to_multiple_of=None
    label_pad_token_id= -100
    data_names = None
    def __init__(self, tokenizer, model=None, label_pad_token_id=-100,data_names=None):
        self.tokenizer = tokenizer
        self.model = model
        self.data_names = data_names
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        features2 = {}
        for name in self.data_names:
          if name == 'triplet':
            continue
          if name in ['labels','filter_id']:
            padding_value=self.label_pad_token_id
          else:
            padding_value=self.tokenizer.pad_token_id
          x_features = [feature[name] for feature in features]
          features2[name] = torch.nn.utils.rnn.pad_sequence(x_features, batch_first=True, padding_value=padding_value)
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features2["labels"])
            features2["decoder_input_ids"] = decoder_input_ids
        return features2
    


"""## run eval"""

from torch.nn.utils.rnn import pad_sequence

class EvalDataCollatorForSeq2Seq:
    model= None
    padding= True
    max_length= None
    pad_to_multiple_of=None
    label_pad_token_id= -100
    data_names = None
    def __init__(self, tokenizer, model=None, padding=True, max_length=None, pad_to_multiple_of=None, label_pad_token_id=-100,data_names=None):
        self.tokenizer = tokenizer
        self.model = model
        self.data_names = data_names
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        features2 = {}
        for name in self.data_names:
        #   if name == 'triplet':
        #     continue
          if name in ['labels','filter_id']:
            padding_value=self.label_pad_token_id
          else:
            padding_value=self.tokenizer.pad_token_id
          x_features = [feature[name] for feature in features]
          features2[name] = torch.nn.utils.rnn.pad_sequence(x_features, batch_first=True, padding_value=padding_value)
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features2["labels"])
            features2["decoder_input_ids"] = decoder_input_ids
        return features2
    
class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1
        self.append_trie = None
        self.bos_token_id = None
    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id
    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1
    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id)
    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie
    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])
    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            if len(output) == 0:
                return [0]
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return [0]
    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(prefix_sequence + [next_token], trie_dict[next_token])
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)
    def __len__(self):
        return self.len
    def __getitem__(self, value):
        return self.get(value)
    


@dataclass
class EvalConfigs:
    num_beams: int = 1
    num_beam_groups: int = 1
    num_return_sequences: int = 1
    max_length: int = 30
    n_ent: int = 14541
    n_rel: int = 237