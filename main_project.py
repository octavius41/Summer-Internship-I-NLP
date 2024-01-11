import pandas as pd
import handy_f as hf

sampler = pd.read_csv('main_data.csv',encoding = "utf-8")
konu_sutunu = sampler.iloc[:,0].values
soru_sutunu = sampler.iloc[:,1].values
cevap_sutunu = sampler.iloc[:,2].values

stemmed_cevap_sutunu = []

#calling embedded collection of questions
collection = hf.collection_get("main_1")

soru = 'Rezervasyonumu yaptıktan sonra ödeme için ne kadar sürem var ve hangi yöntemleri kullanabilirim?'

#querying user input
found = hf.sorgula(soru,"main_1")

#gathering most related answers under one string
large_answers_text = hf.get_answers(found,cevap_sutunu)

#selecting the most related sentences from complete answers
extraction_sum = hf.extraction(large_answers_text)

# detection of overused words in sentences like common forwarding sentences "detay bilgi ödeme sayfa ziyaret et" needs to be removed with vnlp stopword detection in further studies

#combining question and extracted sentences keeping the question as a flag in order to question work as a seed text for generation
rnn_train = soru + extraction_sum

#creating the sequences of sentences moving word by word
sequences = hf.formation(rnn_train)

#encoding sequences
encoded_seqs,vocab_size,tokenizer = hf.encoding(sequences)

#forming the input layer and output layer as we created sequences (x+1) like where x belongs to input layer 1 belongs to output layer
X_train = hf.in_x(encoded_seqs)
y_train = hf.categorized_y(encoded_seqs,vocab_size)
seq_length = X_train.shape[1]

#creating rnn model
model = hf.get_model(vocab_size,seq_length)

#training rnn
hf.train_model(model,X_train,y_train)

#result
resume = hf.generation(model,tokenizer,seq_length,soru,10)

print(resume)



