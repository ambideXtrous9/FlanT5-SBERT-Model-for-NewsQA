from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
import pandas as pd
from transformers import AutoTokenizer as Tokenizer
from Model import NQAModel
from sentence_transformers import SentenceTransformer, util
import torch
import sys
import random
from kivy.uix.image import AsyncImage


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


class MyLayout(BoxLayout):
    
    def generate_answer(self, instance):
        # Add your code to generate the answer here
        question = self.question_input.text
        answer = generate_ans(question)
        self.answer_label.text = "Answer: " + answer
    
    def exit_program(self, instance):
        sys.exit()
    
    def __init__(self, **kwargs):
        super(MyLayout, self).__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = 50
        self.spacing = 20
        
        self.gif_image = AsyncImage(source='img.jpeg')
        self.add_widget(self.gif_image)
        
        self.question_input = TextInput(multiline=True,font_size=30)
        self.add_widget(self.question_input)
        
        self.submit_button = Button(text="Submit", font_size=20)
        self.submit_button.bind(on_press=self.generate_answer)
        self.add_widget(self.submit_button)
        
        self.answer_label = Label(text="Answer: ", font_size=30)
        self.add_widget(self.answer_label)
        
        self.exit_button = Button(text="Exit", font_size=20)
        self.exit_button.bind(on_press=self.exit_program)
        self.add_widget(self.exit_button)



class MyKivyApp(App):
    title = "QA-Assistant"
    def build(self):
        layout = MyLayout()
        layout.size = (800, 600)
        return layout


if __name__ == "__main__":
    MyKivyApp().run()
