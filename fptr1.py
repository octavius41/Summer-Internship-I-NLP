from vnlp import StemmerAnalyzer
from vnlp import StopwordRemover
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

sampler = pd.read_csv('sample_data.csv',encoding = "utf-8")
np_sampler = sampler.iloc[:,:-1].values

konu_sütunu = sampler.iloc[:,0].values
soru_sütunu = sampler.iloc[:,1].values
cevap_sütunu = sampler.iloc[:,2].values
l_soru_sütunu = soru_sütunu.tolist()
stemmed_soru_sütunu = []
count = len(konu_sütunu)

stemmer = StemmerAnalyzer()
for i in range(0,count):
    stemmed_soru_sütunu.append(str(stemmer.predict(l_soru_sütunu[i])))

keys = []
for i in range(0,count):
    keys.append(str(i))

trb_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
client = chromadb.PersistentClient(path="/nlp_project")
collection = client.get_or_create_collection("owpr_1", embedding_function = trb_ef)

temp = konu_sütunu[0]
for i in range(0,count):
    if (temp != konu_sütunu[i]):
        temp = konu_sütunu[i]
    collection.add(documents = stemmed_soru_sütunu[i], metadatas = {"konu" : temp}, ids = keys[i])

sentence = str(stemmer.predict('Uçuşum iptal edilirse, iade veya başka bir uçuş için nasıl bir süreç izlemem gerekecek?'))
model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
sample_embedding = model.encode(sentence)
upd = sample_embedding.tolist()

found = collection.query(query_embeddings = upd, n_results = 3)


f_konu = found['metadatas'][0][0]['konu']

f_indices=[]
for i in range(0,3):
        f_indices.append(int(found['ids'][0][i]))

print(f_indices)












