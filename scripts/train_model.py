import os
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, T5Tokenizer
from .struct_models.rotatE import RotatE
from ..src.model import StructKGS2S
from ..src.utils import DataCollatorForSeq2Seq
from ..src.dataset import KGCDataset, SplitDatasetWrapper
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


if __name__ == "__main__":

    dataset_name = 'fb15k-237'
    root = '/Users/apple/structs2s/'
    max_rel_size = 237
    
    # init datasets
    kg_data_path = os.path.join(root, 'data/processed', dataset_name, 'kg_data.pt')
    kg_data = torch.load(kg_data_path)
    rotatE = RotatE(k=350, entity_embedding=kg_data['RotatE_ent_emb'], relation_embedding=kg_data['RotatE_rel_emb'], max_rel_size=max_rel_size)
    dataset = KGCDataset(num_ents=14541, structal_model=rotatE, kg_data=kg_data)
    train_dataset = SplitDatasetWrapper(dataset, split="train")
    valid_dataset = SplitDatasetWrapper(dataset, split="valid")

    tokenizer = T5Tokenizer.from_pretrained('t5-small', padding=True)
    
    # init model
    ckpt_name ='t5-small'
    model = StructKGS2S.from_pretrained(ckpt_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, data_names=list(train_dataset[0].keys()))

    # training arguments
    batch_size= 32*4
    num_train_epochs = 1000
    learning_rate = 1e-4
    torch_compile = True

    args = Seq2SeqTrainingArguments(
        "kgs2s-rotatE",
        dataloader_num_workers=8,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy='epoch',
        learning_rate=learning_rate,
        torch_compile=torch_compile,
        fp16=True,
        tf32=True,
        report_to='none',
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    # training
    trainer.train()

