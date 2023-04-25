import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from Model import NQAModel
from DataModule import NQADataModule

pl.seed_everything (42)



parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='google/flan-t5-base', help="model for training")
parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
parser.add_argument("--epochs", type=int, default=10, help="number of epochs for training")
args = parser.parse_args()

print("\n============================================================================================\n")

print("Model Name:", args.model_name)
print("Batch size:", args.batch_size)
print("Epochs:", args.epochs)

print("\n============================================================================================\n")

"""# **Dataset**"""

path = 'NewsQA_SPAN.feather'


df = pd.read_feather(path)


"""# **Tokenization**"""

MODEL_NAME = args.model_name

BATCH_SIZE = args.batch_size
N_EPOCHS = args.epochs

train_df, val_df = train_test_split(df,test_size=0.2)
val_df, test_df = train_test_split(val_df,test_size=0.5)

data_module = NQADataModule(train_df,val_df,test_df,MODEL_NAME,batch_size = BATCH_SIZE)
data_module.setup()


model = NQAModel(MODEL_NAME=MODEL_NAME,lr=0.0001)

checkpoint_callback = ModelCheckpoint(
    dirpath = 'checkpoints',
    filename = 'FlanT5-Base',
    save_top_k = 1,
    verbose = True,
    monitor = 'val_METEOR',
    mode = 'max'
)

trainer = pl.Trainer(devices=-1, accelerator="gpu",
    callbacks=[checkpoint_callback],
    max_epochs = N_EPOCHS
)

trainer.fit(model,data_module)

trainer.test(model, data_module)

