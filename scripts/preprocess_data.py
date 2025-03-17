from transformers import T5Tokenizer
from tqdm.auto import tqdm
import os
import torch
import numpy as np
from src.utils import Trie, _tokenize



def get_ent2id(root, dataset):
  path = os.path.join(root, dataset, "entities.txt")

  ent2id = {}
  with open(path, "r") as f:
    total_lines = sum(1 for _ in f)
    f.seek(0)  # Reset file pointer to the beginning
    for i, line in tqdm(enumerate(f), total=total_lines, desc="Processing lines"):
      ent = line.strip().split('\t')[0]
      ent2id[ent] = int(i)
  return ent2id

def get_rel2id(root, dataset):
    path = os.path.join(root, dataset, "relations.txt")
    
    rel2id = {}
    with open(path, "r") as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)  # Reset file pointer to the beginning
        for i, line in tqdm(enumerate(f), total=total_lines, desc="Processing lines"):
            rel = line.strip().split('\t')[0]
            rel2id[rel] = int(i)
    return rel2id


def get_ent2text(root, dataset):
    path = os.path.join(root, dataset, "entity2text.txt")

    ent2text = {}
    with open(path, "r") as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)  # Reset file pointer to the beginning
        for i, line in tqdm(enumerate(f), total=total_lines, desc="Processing lines"):
            ent, text = line.strip().split('\t')
            ent2text[ent] = _tokenize(text)
    return ent2text

def get_rel2text(root, dataset):
    path = os.path.join(root, dataset, "relation2text.txt")

    rel2text = {}
    with open(path, "r") as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)  # Reset file pointer to the beginning
        for i, line in tqdm(enumerate(f), total=total_lines, desc="Processing lines"):
            rel, text = line.strip().split('\t')
            rel2text[rel] = _tokenize(text)
    return rel2text


def get_ent2decs(root, dataset, max_decs = 64):
    path = os.path.join(root, dataset, "entity2textlong.txt")

    ent2decs = {}
    with open(path, "r") as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)  # Reset file pointer to the beginning
        for i, line in tqdm(enumerate(f), total=total_lines, desc="Processing lines"):
            ent, text = line.strip().split('\t')
            ent2decs[ent] = _tokenize(text)[:max_decs]
    return ent2decs


list_set_filename = {
    'fb15k-237': 
        {
            'train': 'train.tsv',
            'valid': 'dev.tsv',
            'test': 'test.tsv'
        }
}

def get_triplets_data(root, dataset, split, ent2id, rel2id, ent2text, ent2decs, rel2text):
    set_filename = list_set_filename[dataset][split]
    path = os.path.join(root, dataset, set_filename)
    
    triplet_id = []
    triplet_tokens = []
    triplet_decs = []
    with open(path, "r") as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)  # Reset file pointer to the beginning
        for i, line in tqdm(enumerate(f), total=total_lines, desc="Processing lines"):
            head, relation, tail = line.strip().split('\t')
            head_id = ent2id[head]
            relation_id = rel2id[relation]
            tail_id = ent2id[tail]
            head_tokens = ent2text[head]
            relation_tokens = rel2text[relation]
            tail_tokens = ent2text[tail]
            head_decs = ent2decs[head]
            tail_decs = ent2decs[tail]
            triplet_id.append((head_id, relation_id, tail_id))
            triplet_tokens.append((head_tokens, relation_tokens, tail_tokens))
            triplet_decs.append((head_decs, tail_decs))
    triplet_id = np.array(triplet_id)
    return triplet_id, triplet_tokens, triplet_decs

def get_entid2text(ent2id, ent2text):
    entid2text = [0]*len(ent2id)
    for ent in tqdm(ent2id):
        entid2text[ent2id[ent]] = [0] + ent2text[ent].tolist() + [1]
    return entid2text

def get_ent_name_decode_list(tokenizer, entid2text):
    ent_name_decode_list = []
    for target in tqdm(entid2text):
        ent_name_decode_list.append(tokenizer.decode(target[1:-1]))
    return ent_name_decode_list

# run
if __name__ == "__main__":
    root = '/Users/apple/Struct-KGS2S/'
    raw_data_dir = os.path.join(root, 'data/raw') 
    dataset = 'fb15k-237'  #wn18rr

    tokenizer = T5Tokenizer.from_pretrained('t5-small', padding=True)
    _tokenize.tokenizer = tokenizer

    ent2id = get_ent2id(raw_data_dir, dataset)
    rel2id = get_rel2id(raw_data_dir, dataset)
    ent2text = get_ent2text(raw_data_dir, dataset)
    ent2decs = get_ent2decs(raw_data_dir, dataset)
    rel2text = get_rel2text(raw_data_dir, dataset)        
    entid2text = get_entid2text(ent2id, ent2text)
    ent_name_decode_list = get_ent_name_decode_list(tokenizer, entid2text)

    train_triplet_id, train_triplet_tokens, train_triplet_decs = get_triplets_data(raw_data_dir, dataset, 'train', ent2id, rel2id, ent2text, ent2decs, rel2text)
    valid_triplet_id, valid_triplet_tokens, valid_triplet_decs = get_triplets_data(raw_data_dir, dataset, 'valid', ent2id, rel2id, ent2text, ent2decs, rel2text)
    test_triplet_id, test_triplet_tokens, test_triplet_decs = get_triplets_data(raw_data_dir, dataset, 'test', ent2id, rel2id, ent2text, ent2decs, rel2text)

    pretrained_struct_emb_dir = os.path.join(root, 'pretrained', dataset)
    struct_ent_emb = torch.load(os.path.join(pretrained_struct_emb_dir, 'struct_ent_emb.pt'))
    struct_rel_emb = torch.load(os.path.join(pretrained_struct_emb_dir, 'struct_rel_emb.pt'))

    kg_data = {
        "train_triplet_id":train_triplet_id,
        "train_triplet_tokens":train_triplet_tokens,
        "train_triplet_decs":train_triplet_decs,
        "valid_triplet_id":valid_triplet_id,
        "valid_triplet_tokens":valid_triplet_tokens,
        "valid_triplet_decs":valid_triplet_decs,
        "test_triplet_id":test_triplet_id,
        "test_triplet_tokens":test_triplet_tokens,
        "test_triplet_decs":test_triplet_decs,
        "struct_ent_emb":struct_ent_emb,
        "struct_rel_emb":struct_rel_emb,
        "ent_name_decode_list":ent_name_decode_list,
        "entid2text":entid2text,
    }

    kg_data_path = os.path.join(root, 'data/processed', dataset, 'kg_data.pt')
    os.makedirs(os.path.dirname(kg_data_path), exist_ok=True)
    torch.save(kg_data, kg_data_path)
