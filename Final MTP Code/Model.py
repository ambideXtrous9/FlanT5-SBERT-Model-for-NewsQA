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

import torch
torch.set_float32_matmul_precision('high')
from torch.optim import AdamW

from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM)


def compute_meteor_scores(predictions, answers):
        scores = []
        for pred, ans in zip(predictions, answers):
            pred_tokens = word_tokenize(pred)
            ans_tokens = word_tokenize(ans)
            score = meteor([ans_tokens], pred_tokens, gamma=0)
            scores.append(score)
        return sum(scores) / len(scores)


def compute_bleu_scores(predictions, answers,smoothing_function):
    scores = []
    for pred, ans in zip(predictions, answers):
        pred_tokens = word_tokenize(pred)
        ans_tokens = word_tokenize(ans)
        score = sentence_bleu([ans_tokens], pred_tokens,smoothing_function=smoothing_function)
        scores.append(score)
    return sum(scores) / len(scores)

def compute_rouge_scores(predictions, answers):
    rouge = Rouge()
    scores = []
    for pred, ans in zip(predictions, answers):
        try:
            score = rouge.get_scores(pred, ans)[0]['rouge-1']['f']
        except ValueError:
            pass  # if pred is empty, just move on to the next pair of predictions and answers
        else:
            scores.append(score)
    return sum(scores) / len(scores) if scores else 0



class NQAModel(pl.LightningModule):
  def __init__(self,MODEL_NAME,lr):
    super().__init__()

    self.MODEL_NAME = MODEL_NAME
    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME,return_dict=True)
    self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
    self.lr = lr
    self.chencherry = SmoothingFunction() # for BLUE-SCORE

  def forward(self,input_ids,attention_mask,labels=None):
    output = self.model(
        input_ids = input_ids,
        attention_mask = attention_mask,
        labels = labels)
    
    return output.loss, output.logits

  def training_step(self,batch,batch_idx):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    loss, outputs = self(input_ids,attention_mask,labels)
    self.log("train_loss",loss,prog_bar=True,logger=True)
    return loss

  def validation_step(self,batch,batch_idx):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    answer = batch['answer']
    loss, outputs = self(input_ids,attention_mask,labels)

    # Generate predictions from the model
    predictions = self.model.generate(input_ids=input_ids, 
                                      attention_mask=attention_mask,
                                      num_beams = 1,
                                      max_length = 32,
                                      repetition_penalty = 2.5,
                                      length_penalty = 1.0,
                                      early_stopping = True,
                                      use_cache = True)
    
    predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

    meteor_score = compute_meteor_scores(predictions, answer)
    blue_score = compute_bleu_scores(predictions, answer,self.chencherry.method2)
    rogue_score = compute_rouge_scores(predictions, answer)

    self.log_dict({"val_loss" : loss,
                   "val_METEOR" : meteor_score,
                   "val_BLUE" : blue_score,
                   "val_ROGUE" : rogue_score},prog_bar=True,logger=True)

    return loss

  def test_step(self,batch,batch_idx):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    answer = batch['answer']
    loss, outputs = self(input_ids,attention_mask,labels)

    # Generate predictions from the model
    predictions = self.model.generate(input_ids=input_ids, 
                                      attention_mask=attention_mask,
                                      num_beams = 1,
                                      max_length = 32,
                                      repetition_penalty = 2.5,
                                      length_penalty = 1.0,
                                      early_stopping = True,
                                      use_cache = True)
    
    predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

    meteor_score = compute_meteor_scores(predictions, answer)
    blue_score = compute_bleu_scores(predictions, answer,self.chencherry.method2)
    rogue_score = compute_rouge_scores(predictions, answer)

    self.log_dict({"test_loss" : loss,
                   "test_METEOR" : meteor_score,
                   "test_BLUE" : blue_score,
                   "test_ROGUE" : rogue_score},prog_bar=True,logger=True)
    return loss

  def configure_optimizers(self):
    return AdamW(self.parameters(),lr = self.lr)
