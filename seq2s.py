from vnlp import StemmerAnalyzer
import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from vnlp import StopwordRemover


sampler = pd.read_csv('sample_data.csv',encoding = "utf-8")
np_sampler = sampler.iloc[:,:-1].values

konu_sütunu = sampler.iloc[:,0].values
soru_sütunu = sampler.iloc[:,1].values
cevap_sütunu = sampler.iloc[:,2].values
count = len(konu_sütunu)
stemmed_soru_sütunu = []



stp_rmv = StopwordRemover()
stemmer = StemmerAnalyzer()
"""""
dum = stemmer.predict("Üniversite sınavlarına ama canla başla fakat çalışıyorlardı.")
print(stp_rmv.drop_stop_words(dum))          #true formation#
"""""

def stem_stop_remover(sentence):

    temp = stemmer.predict(str(sentence))
    final = stp_rmv.drop_stop_words(temp)
    return final

print(stemmed_soru_sütunu[2])
print(soru_sütunu[2])
whole = ' '
sents_wo_stpw=[]

def puzzle_one(sentence):
    whole = ' '
    for i in sentence:
        whole += ' '+i
    return whole

sents_wo_stpw.append(whole)
print(sents_wo_stpw[0])















