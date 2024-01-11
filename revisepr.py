from vnlp import StemmerAnalyzer
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

stemmer = StemmerAnalyzer()

client = chromadb.PersistentClient(path="/nlp_project")
collectionr = client.get_collection("owpr_1")

#collectionr.delete()

sentence = str(stemmer.predict('Uçuşum iptal edilirse, iade veya başka bir uçuş için nasıl bir süreç izlemem gerekecek?'))
model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
sample_embedding = model.encode(sentence)
upd = sample_embedding.tolist()

found = collectionr.query(query_embeddings = upd, n_results = 3)
print(found)