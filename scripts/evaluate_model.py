
import os
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from src.utils import EvalConfigs, RunEval
from transformers import T5Tokenizer
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
from src.dataset import SplitDatasetWrapper
from src.utils import Hop1Index
from src.utils import EvalDataCollatorForSeq2Seq
from src.dataset import KGCDataset, SplitDatasetWrapper
from .struct_models.rotatE import RotatE


dataset_name = 'fb15k-237'
root = '/Users/apple/structs2s/'
max_rel_size = 237


num_return_sequences = 1
max_length = 30 
n_ent = 14541
n_rel = 237

model_path = '/Users/apple/structs2s/models/saved_models/kgt5_rotatE_x11.pt'
model = torch.load(model_path)
model.eval()
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
entity_embedding = kg_data['RotatE_ent_emb']
relation_embedding = kg_data['RotatE_rel_emb']

rotatE = RotatE(k=350, entity_embedding=entity_embedding, relation_embedding=relation_embedding, max_rel_size=max_rel_size)
dataset = KGCDataset(num_ents=14541, structal_model=rotatE, kg_data=kg_data)


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


tail_data_loader = DataLoader(tail_test_dataset, batch_size=64, shuffle=False, collate_fn=eval_data_collator, num_workers=8, pin_memory=True)
head_data_loader = DataLoader(head_test_dataset, batch_size=64, shuffle=False, collate_fn=eval_data_collator, num_workers=8, pin_memory=True)

runEval = RunEval(configs, model, tokenizer, ent_name_decode_list, target_ids, ext_neigs_0, ext_neigs_2)
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

