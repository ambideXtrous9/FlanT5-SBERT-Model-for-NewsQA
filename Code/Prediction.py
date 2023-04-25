import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

import pytorch_lightning as pl
import nltk
nltk.download('wordnet')
nltk.download('wordnet_ic')
nltk.download('punkt')

from nltk.translate.meteor_score import meteor_score as meteor
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk.translate.bleu_score import SmoothingFunction
from nltk import word_tokenize
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
torch.set_float32_matmul_precision('high')
from torch.optim import AdamW
import argparse
from Model import NQAModel,compute_bleu_scores,compute_meteor_scores,compute_rouge_scores

from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM)

pl.seed_everything (42)



parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='google/flan-t5-base', help="model for training")
parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
args = parser.parse_args()

print("\n============================================================================================\n")

print("Model Name:", args.model_name)
print("Batch size:", args.batch_size)

print("\n============================================================================================\n")


path = 'NewsQA_SPAN.feather'

df = pd.read_feather(path)

MODEL_NAME = args.model_name



bert_model = SentenceTransformer('all-MiniLM-L12-v2')

corpus_embeddings = bert_model.encode(df['question'],
                                      batch_size = 8192,
                                      show_progress_bar=True,
                                      convert_to_tensor=True)




 
def predict_context(question):
    question_embedding = bert_model.encode(question,convert_to_tensor=True)
    result = util.semantic_search(question_embedding,corpus_embeddings)[0][0]
    top_context = df.iloc[result['corpus_id']]['paragraph']
    return top_context



class NQADataset(Dataset):
  def __init__(self,data ,model_name ,source_max_token_len : int = 400,target_max_token_len : int = 32):

    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.data = data
    self.source_max_token_len = source_max_token_len
    self.target_max_token_len = target_max_token_len

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self,index : int):
    data_row = self.data.iloc[index]
    
    question = data_row['question']
    
    paragraph = predict_context(question)

    source_encoding = self.tokenizer(
        question,
        paragraph,
        max_length = self.source_max_token_len,
        padding = "max_length",
        truncation = "only_second",
        return_attention_mask = True,
        add_special_tokens = True,
        return_tensors = "pt")
    
    return dict(
        answer = data_row['answer'],
        input_ids = source_encoding['input_ids'].flatten(),
        attention_mask = source_encoding['attention_mask'].flatten())



test_dataset = NQADataset(data=df,model_name=args.model_name)

test_dataloader = DataLoader(test_dataset,batch_size = args.batch_size)


class MyModel(NQAModel):
    def __init__(self, MODEL_NAME, lr,tokenizer):
        super().__init__(MODEL_NAME, lr)
        self.tokenizer = tokenizer
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        answer = batch['answer']

        # Generate predictions from the model
        predictions = self.model.generate(input_ids=input_ids, 
                                          attention_mask=attention_mask,
                                          num_beams=1,
                                          max_length=32,
                                          repetition_penalty=2.5,
                                          length_penalty=1.0,
                                          early_stopping=True,
                                          use_cache=True)

        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        meteor_score = compute_meteor_scores(predictions, answer)
        blue_score = compute_bleu_scores(predictions, answer,self.chencherry.method2)
        rogue_score = compute_rouge_scores(predictions, answer)

        self.log_dict({"test_METEOR": meteor_score,
                       "test_BLUE": blue_score,
                       "test_ROGUE": rogue_score},
                      prog_bar=True,
                      logger=True)
        return None



tokenizer = AutoTokenizer.from_pretrained(args.model_name)

cppath = 'FlanT5-BaseFinal.ckpt'
trained_model = MyModel.load_from_checkpoint(cppath,MODEL_NAME=MODEL_NAME,lr=0.0001,tokenizer=tokenizer)
trained_model.freeze()

trainer = pl.Trainer(devices=-1, accelerator="gpu")

trainer.test(model=trained_model, dataloaders=test_dataloader)
