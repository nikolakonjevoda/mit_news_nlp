"""
- Top 10 reprezentacji słów (tworzenie bow) - wykres

- Chmurę słów w postaci top_n słów - wykres

- Analizę emocji - wykres

- Analizę min. 3 tematów ze zbioru słów - wykres

"""


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

#import articles
articles = pd.read_csv('articles.csv')
print(articles.head())

#create a corpus
corpus = articles['Text'].tolist()
print(corpus[0])

#create CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
cvec = count_vectorizer.fit_transform(corpus)

#assign tokens
tokens=count_vectorizer.get_feature_names_out()

#create occurence array
array = cvec.toarray()

#create bow
bow = pd.DataFrame(array, columns=tokens)
print(bow.head())

#plot the top ten items in bow
word_count = bow.sum(axis=0)
word_count_sorted = word_count.sort_values(ascending=False)
topten = word_count_sorted[:10]
topten.plot(kind='bar')
plt.xlabel('Top 10 words')
plt.ylabel('Quantity')
plt.show()

#create wordcloud

from wordcloud import WordCloud 

#create dict for wordcloud
word_count_dict = dict(zip(count_vectorizer.get_feature_names_out(), word_count))
#initiate the wordcloud
mit_wrdcld = WordCloud(background_color='white').generate_from_frequencies(word_count_dict)
plt.axis("off")
plt.imshow(mit_wrdcld)


#analysing sentiment
from textblob import TextBlob

#sentiment for whole document
sent = TextBlob(str(corpus))
print(sent.sentiment)

#sentiment per article
import nltk
import text2emotion as te
nltk.download('omw-1.4')
emotions = [te.get_emotion(str(c)) for c in corpus]
print(emotions)

#save to pickle to not loose the data if there is some software error - text2emotion runs for a couple of hours
import pickle

with open('emotions.pkl', 'wb') as f:
    pickle.dump(emotions, f)

#create df of emotions
df_emotions = pd.DataFrame(emotions)
print(df_emotions.head())

#match emotions with the titles
article_emotion = pd.concat([articles['Title'], df_emotions], axis=1)
article_emotion.head()

#plot a boxplot of emotions
plt.boxplot(df_emotions, labels=(df_emotions.columns))
plt.ylabel('text2emotion value')
plt.show()


#LDA
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Instantiate a tokenizer that captures only word characters (ignores punctuation and special characters)
tokenizer = RegexpTokenizer(r'\w+')

# Preprocessing
stop_words = set(stopwords.words('english')) 

# tokenize, convert to lower case, and remove punctuation and special characters
texts = []
for document in corpus:
    tokenized_document = tokenizer.tokenize(document)
    texts.append([word.lower() for word in tokenized_document if word.lower() not in stop_words])

# Create a dictionary from the texts
dictionary = corpora.Dictionary(texts)

# Create a corpus from the dictionary
corp = [dictionary.doc2bow(text) for text in texts]

# Train the LDA model
lda_model = models.LdaModel(corp, num_topics=10, id2word=dictionary, passes=15)

# Print the topics
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)
