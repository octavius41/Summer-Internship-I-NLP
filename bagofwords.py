import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

wnlem = WordNetLemmatizer()
portst = PorterStemmer()

stpwrds = set(stopwords.words('english'))
par = "Delightful unreserved impossible few estimating men favourable see entreaties. She propriety immediate was improving. He or entrance humoured likewise moderate. Much nor game son say feel. Fat make met can must form into gate. Me we offending prevailed discovery."
corpus = []
sents = nltk.sent_tokenize(par)

for i in range(len(sents)):
    rev = re.sub('[^a-zA-Z]+',' ',sents[i])
    rev = rev.lower()
    rev = rev.split()
    rev = [wnlem.lemmatize(word) for word in rev if not word in stpwrds]
    rev = ' '.join(rev)
    corpus.append(rev)

from sklearn.feature_extraction.text import CountVectorizer
cvz = CountVectorizer()
x = cvz.fit_transform(corpus).toarray()
print(x)