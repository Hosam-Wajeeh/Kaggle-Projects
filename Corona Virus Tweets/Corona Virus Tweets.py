# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 20:08:06 2022

@author: hosam
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('mode.chained_assignment', None)
palette=sns.color_palette('magma')
sns.set(palette=palette)

train_data=pd.read_csv('Corona_NLP_train.csv', encoding='latin-1')
test_data=pd.read_csv('Corona_NLP_test.csv')


train_data.info()


#Top 10 Countries that had the highest tweets
plt.figure(figsize=(12,6))
plt.title('Top 10 Countries with highest Tweets')
countries =sns.countplot(x='Location', data=train_data, order=train_data['Location'].value_counts().index[:10], palette=palette)
countries.set_xticklabels(countries.get_xticklabels(), rotation=45)
plt.show() 


#Pie Chart for the Sentiments Percentage
plt.figure(figsize=(10,6))
plt.pie(train_data['Sentiment'].value_counts(), labels=train_data['Sentiment'].unique(), autopct='%.1f%%', textprops={'color':"w"})
plt.legend(loc='upper right')
plt.axis('equal')
plt.show()


#WordCloud for the Sentiments
from wordcloud import WordCloud
for label, cmap in zip(['Positive', 'Negative', 'Neutral', 'Extremely Positive', 'Extremely Negative'],
                       ['winter', 'autumn', 'magma', 'viridis', 'plasma']):
    text = train_data.query('Sentiment == @label')['OriginalTweet'].str.cat(sep=' ')
    plt.figure(figsize=(10, 6))
    wc = WordCloud(width=1000, height=600, background_color="#f8f8f8", colormap=cmap)
    wc.generate_from_text(text)
    plt.imshow(wc)
    plt.axis("off")
    plt.title(f"Words Commonly Used in ${label}$ Messages", size=20)
    plt.show()


#Only intrested in the OriginalTweet and Sentiment Columns
train_df= train_data[['OriginalTweet','Sentiment']]
test_df= test_data[['OriginalTweet','Sentiment']]


#Checking for null values
train_df.isnull().sum()
test_df.isnull().sum()

#Mapping the target column
target_mapping={'Extremely Negative':0, 'Negative':0, 'Neutral':1,
                'Positive':2, 'Extremely Positive':2}
train_df['SentimentMapped']=train_df['Sentiment'].map(lambda x:target_mapping[x])
test_df['SentimentMapped']=test_df['Sentiment'].map(lambda x:target_mapping[x])



#Cleaning the text
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
#69000==>31750

def preprocessor(data):
    corpus = []
    for i in range(len(data)):
        #remove urls
        tweet1= re.sub(r'http\S+', ' ', data['OriginalTweet'][i])
        #remove html tags
        tweet2 = re.sub(r'<.*?>',' ', tweet1) 
        #remove digits
        tweet3 = re.sub(r'\d+',' ', tweet2)
        #remove hashtags
        tweet4 = re.sub(r'#\w+',' ', tweet3)
        review = re.sub('[^a-zA-Z]', ' ', tweet4)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if word not in all_stopwords]
        review = ' '.join(review)
        corpus.append(review)
    return corpus  


    
corpus_train=preprocessor(train_df)
corpus_test=preprocessor(test_df)



#Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=10000) 
X_train = cv.fit_transform(corpus_train).toarray()
y_train = train_df['SentimentMapped']
X_test = cv.transform(corpus_test).toarray()
y_test = test_df['SentimentMapped']


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(random_state=0, early_stopping=True, verbose=2)
mlp.fit(X_train, y_train)
y_pred_mlp=mlp.predict(X_test)
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
print('MLP Accuracy:', accuracy_score(y_test, y_pred_mlp))
print(classification_report(y_test, y_pred_mlp))
sns.heatmap(cm_mlp, annot=True, fmt='g', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('MLP Confusion Matrix')
plt.show()
#MLP Training Accuracy: 0.9015477318560634
#MLP Testing  Accuracy: 0.8101632438125329




'''
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_mnb=mnb.predict(X_test)
cm_mnb = confusion_matrix(y_test, y_pred_mnb)
print('Multinomial NB Accuracy:', accuracy_score(y_test, y_pred_mnb))
print(classification_report(y_test, y_pred_mnb))
sns.heatmap(cm_mnb, annot=True, fmt='g', cbar=False, cmap='BuPu')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('')
plt.show()
#Multinomial NB Accuracy: 0.6898367561874671



from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('Guassian Accuracy:', accuracy_score(y_test, y_pred))
print(cm)
#Guassian Accuracy: 0.3385992627698789


from xgboost import XGBClassifier
xgb = XGBClassifier(use_label_encoder=False, verbosity=0, random_state=0, n_job=-1)
xgb.fit(X_train,y_train)
y_pred_xgb=xgb.predict(X_test)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print('XGBoost Accuracy:', accuracy_score(y_test, y_pred_xgb))
print(cm_xgb)
#XGBoost Accuracy: 0.7696155871511322
'''
'''
MLP Accuracy: 0.6421800947867299
              precision    recall  f1-score   support

           0       0.72      0.61      0.66       592
           1       0.60      0.59      0.60      1041
           2       0.68      0.72      0.70       619
           3       0.57      0.64      0.61       947
           4       0.74      0.68      0.70       599

    accuracy                           0.64      3798
   macro avg       0.66      0.65      0.65      3798
weighted avg       0.65      0.64      0.64      3798

'''

