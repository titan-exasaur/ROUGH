import nltk
# nltk.download()
# nltk.download('punkt')

paragraph = """After listening to all of you, I feel that the number of Lakhpati Didis
            in the country is going to increase significantly. When people read and hear your stories, 
            they will feel inspired. You should share your experiences with others—how it has been, 
            how self-reliant you have become, and how much you can support your entire family. Moreover, 
            your empowerment brings significant changes to the environment around you. Do you know what my goal is? 
            You see, 1 crore Didis have already become Lakhpati Didis, and I aim to make 3 crore Lakhpati Didis. 
            So, you must help explain this to others. Will you do that?"""

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []

for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()

print(X)

import pandas as pd
df = pd.DataFrame(X)
df.to_csv("bag_of_words.csv", index = False)
