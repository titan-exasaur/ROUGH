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

sentences = nltk.sent_tokenize(paragraph)#sentence tokenization

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

stemmer = PorterStemmer()
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)

print(sentences)