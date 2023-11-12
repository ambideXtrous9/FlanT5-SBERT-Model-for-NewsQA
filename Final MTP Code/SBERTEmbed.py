from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

path = 'NewsQA_SPAN.feather'
df = pd.read_feather(path)


bert_model = SentenceTransformer('SBERT/')

corpus_embeddings = bert_model.encode(df['question'],
                                      batch_size = 1024,
                                      show_progress_bar=True,
                                      convert_to_tensor=True)



torch.save(corpus_embeddings, 'corpus_embeddings.pt')
