import requests
from bs4 import BeautifulSoup
import pandas as pd

#create empty list to store urls
articleurls = {}
 
#make a request to get link to each of the articles 
for x in range(67):
    page = requests.get(
        ("https://news.mit.edu/topic/artificial-intelligence2?page=" + str(x))
    )
    soup = BeautifulSoup(page.content, 'html.parser')

    #extract title of article
    articles = soup.find_all('article', {'class' : 'term-page--news-article--item', 'role' :'article'})
    
    #extract the link of the article
    for article in articles:
        #find all a tags and loop to make sure there is only one link per article
        articlenm = article.find_all('a', {'class':'term-page--news-article--item--title--link'})
        for a in articlenm:
            articleurls[a.text] = a['href']
            

#create a df out of extracted data and inspect it
articledf = pd.DataFrame(list(articleurls.items()), columns=['Title', 'URL'])
print(articledf.head())

#initiate list to store
article_texts = [] 
#variables that will be used to show progress
notify=0
n_articles = len(articledf)

#make a request to get text of each article 
for url in articledf['URL']:
    link = 'https://news.mit.edu' + url
    print(link)
    page = requests.get(link)
    soup = BeautifulSoup(page.content, 'html.parser')

    #get the main content of the article page
    article_content = soup.find('div', class_='paragraph paragraph--type--content-block-text paragraph--view-mode--default')
    if article_content is not None:
        text = article_content.get_text()
    else:
        text = ''

    article_texts.append(text)
    notify+=1
    print(str(notify) + ' / ' + str(n_articles))
    print(text[:4])

#add the texts as a new column to the dataframe
articledf['Text'] = article_texts
print(articledf.head())

#save the df as csv to desired path
articledf.to_csv('articles.csv', index=False) 