import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

par = "Uzun süren araştırmalar sonrası anlaşılmıştır ki bir çalışma taslağı veya şablonuna bakarken taslağın barındığı içeriğin anlamının etkisiyle okuyucunun dikkati dağılır. Lorem Ipsum ise, 'buraya içerik gelecek, buraya içerik gelecek' kısmının yerine konularak, bir düzen halinde sembolik içerik olarak yerleştirildiğinden gerçek içeriğe daha yakın bir sonuç verir. Günümüzde pek çok masaüstü yazılımı ve web tasarımcısı örnek yazı için Lorem Ipsum kullanmaktadır."
sents = nltk.sent_tokenize(par)
wordp = nltk.word_tokenize(sents[2])
print(len(wordp))

stemmer = PorterStemmer()
stop_words = set(stopwords.words('turkish'))

for i in range(len(sents)):
    words = nltk.word_tokenize(sents[i])
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    sents[i] = ' '.join(words)

print(len(words))
print(sents)

# lemmetization is same and it takes the true roots of the words or in a more approtiate way you can apply by changing stem to lemm calls