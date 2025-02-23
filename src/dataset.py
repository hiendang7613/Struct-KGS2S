import locale
import torch
import numpy as np
from torch.utils.data import Dataset
import random
from .utils import _tokenize


def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

class Hop1Index:
    def __init__(self, triples, num_entities, key_col=0, max_context_size=64):
        self.max_context_size = max_context_size
        self.shuffle = False
        self.key_col = key_col
        self.triples = triples[triples[:, key_col].argsort()]
        keys, values_offset = np.unique(
            self.triples[:, key_col], axis=0, return_index=True
        )
        values_offset = np.append(values_offset, len(self.triples))
        self.keys = keys
        self.values_offset = values_offset

        self.key_to_start = -1 * np.ones(num_entities, dtype=int)
        self.key_to_start[keys] = values_offset[:-1]
        self.key_to_end = -1 * np.ones(num_entities, dtype=int)
        self.key_to_end[keys] = values_offset[1:]

    def __getitem__(self, item, rel_id=None):
        start = self.key_to_start[item]
        end = self.key_to_end[item]
        context = self.triples[start:end, [1, 2 - self.key_col]]
        if rel_id is not None:
            context = context[context[:,0] == rel_id][:,1]
        if len(context) > self.max_context_size:
            ids = np.random.choice(len(context), self.max_context_size, replace=False)
            context = context[ids]
        if self.shuffle:
            np.random.shuffle(context)
        return context

    def get_context(self, item, rel_id=None):
        return self.__getitem__(item, rel_id)




class KGCDataset(Dataset):
    def __init__(self, num_ents=14541, structal_model=None, kg_data=None, tokenizer=None):
        self.num_ents = num_ents
        self.structal_model = structal_model
        # Fb15k wn18rr
        self.id_triplets ={
            'train': kg_data['train_triplet_id'],
            'valid': kg_data['valid_triplet_id'],
            'test': kg_data['test_triplet_id']
        }
        self.tokens_triplets ={
            'train': kg_data['train_triplet_tokens'],
            'valid': kg_data['valid_triplet_tokens'],
            'test': kg_data['test_triplet_tokens']
        }
        self.decs_triplets ={
            'train': kg_data['train_triplet_decs'],
            'valid': kg_data['valid_triplet_decs'],
            'test': kg_data['test_triplet_decs']
        }

        self.get_neigs_0 ={
            'train': Hop1Index(self.id_triplets['train'], self.num_ents, 0),
            'valid': Hop1Index(self.id_triplets['valid'], self.num_ents, 0),
            'test': Hop1Index(self.id_triplets['test'], self.num_ents, 0)
        }
        self.get_neigs_2 ={
            'train': Hop1Index(self.id_triplets['train'], self.num_ents, 2),
            'valid': Hop1Index(self.id_triplets['valid'], self.num_ents, 2),
            'test': Hop1Index(self.id_triplets['test'], self.num_ents, 2)
        }

        self.mask_token = _tokenize('<extra_id_90>')
        self.eos_token = torch.tensor([tokenizer.eos_token_id])
        self.zero_neig_embedding = torch.zeros([512])

        self.predict_head_token = _tokenize('predict head :')
        self.predict_tail_token = _tokenize('predict tail :')
        self.start_decs_token = _tokenize('[')
        self.end_decs_token = _tokenize(']')
        self.inversion_token = _tokenize('inversion of ')
        self.empty_token = torch.tensor([], dtype=torch.int)
        self.set_ent_id = set(range(self.num_ents))
        self.p_dropout = 0. # 0.2 when training
        self.max_neg_ent_index = 14505
        self.struct_emb_dim = 700

    def __getitem__(self, idx):
        return self.get(idx, split=self.split)
    def __len__(self, split='train'):
        return len(self.tokens_triplets[split])

    def get(self, idx: int, split: str = "train", full_mask_part_idx=None):
        head_lbl, relation, tail_lbl = self.tokens_triplets[split][idx]
        head_id, rel_id, tail_id = self.id_triplets[split][idx]
        head_decs, tail_decs = self.decs_triplets[split][idx]

        if full_mask_part_idx is None:
          full_mask_part_idx = 2 if random.randint(0, 1) else 0

        inversion = False

        if full_mask_part_idx:
          source = [
              self.predict_tail_token if not inversion else self.predict_head_token,
              head_lbl,
              self.start_decs_token,
              head_decs,
              self.end_decs_token,
              self.inversion_token if inversion else self.empty_token,
              relation,
          ]
          target = [tail_lbl]
          neighboors_0 = self.get_neigs_0[split][head_id]
          neighboors_0 = neighboors_0[(neighboors_0[:,0]!=rel_id) | (neighboors_0[:,1]!=tail_id)]
          neighboors_2 = self.get_neigs_2[split][head_id]
          neighboors_2 = neighboors_2[(neighboors_2[:,0]!=rel_id) | (neighboors_2[:,1]!=tail_id)]
        else:
          source = [
              self.predict_head_token if not inversion else self.predict_tail_token,
              tail_lbl,
              self.start_decs_token,
              tail_decs,
              self.end_decs_token,
              self.inversion_token if inversion else self.empty_token,
              relation,
          ]
          target = [head_lbl]
          neighboors_0 = self.get_neigs_0[split][tail_id]
          neighboors_0 = neighboors_0[(neighboors_0[:,0]!=rel_id) | (neighboors_0[:,1]!=head_id)]
          neighboors_2 = self.get_neigs_2[split][tail_id]
          neighboors_2 = neighboors_2[(neighboors_2[:,0]!=rel_id) | (neighboors_2[:,1]!=head_id)]

        target_ent_embeddings = []
        neighboors_embeddings = []
        for rel_n_id, ent_n_id in neighboors_0:
          if ent_n_id >= self.max_neg_ent_index:
            continue
          ent_n_embedding = self.structal_model.entity_embedding[ent_n_id]
          rel_n_embedding = self.structal_model.relation_embedding[rel_n_id]
          target_ent_embedding = self.structal_model(ent_n_id, rel_n_id)
          neighboors_embeddings.append(torch.cat([ent_n_embedding, rel_n_embedding]))
          target_ent_embeddings.append(target_ent_embedding)
        for rel_n_id, ent_n_id in neighboors_2:
          if ent_n_id >= self.max_neg_ent_index:
            continue
          ent_n_embedding = self.structal_model.entity_embedding[ent_n_id]
          rel_n_embedding = self.structal_model.relation_embedding[rel_n_id]
          target_ent_embedding = self.structal_model(ent_n_id, rel_n_id)
          neighboors_embeddings.append(torch.cat([ent_n_embedding, -rel_n_embedding]))
          target_ent_embeddings.append(target_ent_embedding)

        if len(neighboors_embeddings):
          neighboors_embeddings = torch.stack(neighboors_embeddings)
          target_ent_embeddings = torch.stack(target_ent_embeddings)
          neighboors_embeddings_mask = torch.ones(len(neighboors_embeddings))
        else:
          neighboors_embeddings_mask = torch.zeros([1])
          neighboors_embeddings = torch.zeros([1, self.struct_emb_dim*2])
          target_ent_embeddings = torch.zeros([1, self.struct_emb_dim])


        source.append(self.eos_token)
        target.append(self.eos_token)
        source = torch.cat(source)
        target = torch.cat(target)

        attention_mask = torch.ones_like(source)
        rand = torch.rand_like(attention_mask.float())
        dropout = torch.logical_not(rand < self.p_dropout).long()
        dropout[(source == self.start_decs_token[0]) | (source == self.end_decs_token[0])] = 1
        dropout[:4]=1
        inversion_len = len(self.inversion_token if inversion else self.empty_token)
        relation_len = len(relation)
        dropout[-relation_len-inversion_len:-relation_len]=1
        attention_mask = attention_mask * dropout


        output = {
            "input_ids": source,
            "attention_mask": attention_mask,
            "labels": target,
            'neighboors_embeddings': neighboors_embeddings,
            'neighboors_embeddings_mask': neighboors_embeddings_mask,
            'target_ent_embeddings': target_ent_embeddings,
            'triplet': self.id_triplets[split][idx],
        }
        return output
    

class SplitDatasetWrapper:
    def __init__(self, dataset, split, full_mask_part_idx=None):
        self.dataset = dataset
        self.split = split
        self.full_mask_part_idx = full_mask_part_idx
    def __getitem__(self, idx):
        return self.dataset.get(idx, self.split, self.full_mask_part_idx)
    def __len__(self):
        return self.dataset.__len__(split=self.split)






