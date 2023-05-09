import tkinter as tk
from tkinter import Entry
import matplotlib
import pandas as pd
matplotlib.use('Agg')
from transformers import AutoTokenizer as Tokenizer
from Model import NQAModel
from sentence_transformers import SentenceTransformer, util
import torch


path = 'NewsQA_SPAN.feather'
df = pd.read_feather(path)


MODEL_NAME = 'google/flan-t5-base'
tokenizer = Tokenizer.from_pretrained(MODEL_NAME)

MODEL_PATH = 'FlanT5-BaseFinal.ckpt'
trained_model = NQAModel.load_from_checkpoint(MODEL_PATH,MODEL_NAME=MODEL_NAME,lr=0.0001)
trained_model.freeze()



bert_model = SentenceTransformer('SBERT/')

corpus_embeddings = torch.load('corpus_embeddings.pt').to('cuda')



 
def predict_context(question):
    question_embedding = bert_model.encode(question,convert_to_tensor=True)
    result = util.semantic_search(question_embedding,corpus_embeddings)[0][0]
    top_context = df.iloc[result['corpus_id']]['paragraph']
    return top_context


def generate_ans(question):
    source_encoding = tokenizer(
        question,
        predict_context(question),
        max_length = 200,
        padding = "max_length",
        truncation = "only_second",
        return_attention_mask = True,
        add_special_tokens = True,
        return_tensors = "pt")
    
    generated_ids = trained_model.model.generate(
        input_ids = source_encoding['input_ids'],
        attention_mask = source_encoding['attention_mask'],
        num_beams = 1,
        max_length = 20,
        repetition_penalty = 2.5,
        length_penalty = 1.0,
        early_stopping = True,
        use_cache = True)
    
    return tokenizer.decode(generated_ids[0],skip_special_tokens=True,clean_up_tokenization_spaces=True)
    

def get_answer():
    # get the question from the user
    question = question_entry.get()
    # pass the question to your model and get the answer
    answer = generate_ans(question)
    # display the answer in the GUI
    answer_label.config(text=answer)

# create the main window
root = tk.Tk()
root.geometry("640x480")
root.title("Question Answering")

# create a label for the question
question_label = tk.Label(root, text="Enter your question:", font=("Helvetica", 16))
question_label.pack()

# create a text entry for the question
question_entry = Entry(root, bd=10, width=40, font=("Helvetica", 20), background='lightblue')
question_entry.pack()

# create a submit button
submit_button = tk.Button(root, text="Submit", font=("Helvetica", 16), bg='lightgreen', relief='solid', bd=3, activebackground='darkgreen', command=get_answer)
submit_button.pack()


# create a label for the answer
answer_label = tk.Label(root, text="", font=("Helvetica", 16))
answer_label.pack()

# start the GUI
root.mainloop()