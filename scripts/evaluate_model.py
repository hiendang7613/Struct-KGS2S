
import os
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from src.utils import EvalConfigs, RunEval
from transformers import T5Tokenizer
from src.dataset import SplitDatasetWrapper
from src.utils import Hop1Index
from src.utils import EvalDataCollatorForSeq2Seq
from src.dataset import KGCDataset, SplitDatasetWrapper
from .struct_models.rotatE import RotatE




dataset_name = 'fb15k-237'
root = '/Users/apple/Struct-KGS2S/'


num_return_sequences = 1
max_length = 30 
n_ent = 14541
n_rel = 237
device = 'cpu' 

model_path = '/Users/apple/Struct-KGS2S/models/saved_models/structkgs2s_fb15k237.pt'
model_state_dict = torch.load(model_path, map_location=torch.device('mps'))

from transformers import AutoConfig
from src.model import StructKGS2S

# init model
ckpt_name ='t5-small'

config = AutoConfig.from_pretrained(ckpt_name)
config.struct_d_model = 700

model = StructKGS2S.from_pretrained(ckpt_name, config=config)
model.load_state_dict(model_state_dict)

model.to(device)
tokenizer = T5Tokenizer.from_pretrained('t5-small', padding=True)

configs = EvalConfigs(
    num_return_sequences = num_return_sequences,
    max_length = max_length,
    n_ent = n_ent,
    n_rel = n_rel,
)

# init datasets
kg_data_path = os.path.join(root, 'data/processed', dataset_name, 'kg_data.pt')
kg_data = torch.load(kg_data_path)
ent_name_decode_list = kg_data['ent_name_decode_list']
entid2text = kg_data['entid2text']
target_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in entid2text], batch_first=True, padding_value=0)
entity_embedding = kg_data['struct_ent_emb']
relation_embedding = kg_data['struct_rel_emb']
trie = kg_data['trie']

rotatE = RotatE(k=350, entity_embedding=entity_embedding, relation_embedding=relation_embedding, max_rel_size=n_rel)
dataset = KGCDataset(num_ents=14541, structal_model=rotatE, kg_data=kg_data, tokenizer=tokenizer)


ext_neigs_0 ={
    'train': Hop1Index(kg_data['train_triplet_id'], dataset.num_ents, 0, max_context_size=1e10),
    'valid': Hop1Index(kg_data['valid_triplet_id'], dataset.num_ents, 0, max_context_size=1e10),
    'test': Hop1Index(kg_data['test_triplet_id'], dataset.num_ents, 0, max_context_size=1e10),
}

ext_neigs_2 ={
    'train': Hop1Index(kg_data['train_triplet_id'], dataset.num_ents, 2, max_context_size=1e10),
    'valid': Hop1Index(kg_data['valid_triplet_id'], dataset.num_ents, 2, max_context_size=1e10),
    'test': Hop1Index(kg_data['test_triplet_id'], dataset.num_ents, 2, max_context_size=1e10),
}


head_test_dataset = SplitDatasetWrapper(dataset, split="test", full_mask_part_idx=0)
tail_test_dataset = SplitDatasetWrapper(dataset, split="test", full_mask_part_idx=2)

eval_data_collator = EvalDataCollatorForSeq2Seq(tokenizer, model=model, data_names=list(head_test_dataset[0].keys()))


tail_data_loader = DataLoader(tail_test_dataset, batch_size=2, shuffle=False, collate_fn=eval_data_collator, )
head_data_loader = DataLoader(head_test_dataset, batch_size=2, shuffle=False, collate_fn=eval_data_collator, )

runEval = RunEval(configs, model, tokenizer, ent_name_decode_list, target_ids, trie, ext_neigs_0, ext_neigs_2, device=device)

head_list_result = []
for data in tqdm(head_data_loader):
    rank_rs = runEval.validation_step(data, 0)
    head_list_result.append(rank_rs)
    
tail_list_result = []
for data in tqdm(tail_data_loader):
    rank_rs = runEval.validation_step(data, 2)
    tail_list_result.append(rank_rs)

kq = runEval.validation_epoch_end((head_list_result, tail_list_result))

print(kq)

