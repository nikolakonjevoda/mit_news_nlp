"""
- Top 10 reprezentacji słów (tworzenie bow) - wykres

- Chmurę słów w postaci top_n słów - wykres

- Analizę emocji - wykres

- Analizę min. 3 tematów ze zbioru słów - wykres

- oraz środowiska czyli zapis pliku w postaci "XXX.Rdata" (czyli skrypt + dane, na których pracowaliście).

Przed tworzeniem wykresów dane powinny być odczyszczone z niepotrzebnych elementów, tak jak pokazane zostało to na zajęciach praktycznie na każdym z warsztatów, przed pracą z danymi.


Będę zwracał uwagę na poprawność wykonania skryptu oraz wykresów. Szczegółów dot. wyboru danych nie będę oceniał, możecie mieć własne zbiory danych.

Każdy dodatkowy wykres, pogłębiona ,wnikliwa analiza mile widziana.
"""

#create bow

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

#assign our tokens
tokens=count_vectorizer.get_feature_names_out()

#create our occurence array
array = cvec.toarray()

#create bow
bow = pd.DataFrame(array, columns=tokens)
print(bow.head())

#plot the top ten items in bow
word_count = bow.sum(axis=0)
word_count_sorted = word_count.sort_values(ascending=False)
topten = word_count_sorted[:10]
topten.plot(kind='bar')
plt.show()

#create wordcloud

from wordcloud import WordCloud 

#create dict for wordcloud
word_count_dict = dict(zip(count_vectorizer.get_feature_names_out(), word_count))
#initiate the wordcloud
mit_wrdcld = WordCloud(background_color='white').generate_from_frequencies(word_count_dict)
plt.axis("off")
plt.imshow(mit_wrdcld)
