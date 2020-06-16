"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
import re
import nltk

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
#pip install wordCloud
# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	def remove_twitter_handles(tweet, pattern):
		r = re.findall(pattern, tweet)
		for text in r:
			tweet = re.sub(text, '', tweet)
		return tweet

	raw['clean_tweet'] = np.vectorize(remove_twitter_handles)(raw['message'], "@[\w]*") 


	st.title("Insights on people's peception on Climate Change ")

	st.subheader("Sentiment Data")

	plt.figure(figsize=(12,6))
	sns.countplot(x='sentiment',data=raw, palette='CMRmap')
	plt.title('Number of Tweets per Class', fontsize=20)
	plt.xlabel('Number of Tweets', fontsize=14)
	plt.ylabel('Class', fontsize=14)
	plt.show()
	st.pyplot()

	st.subheader("Understanding Common Words in the Positive Tweets")
	df_pro = raw[raw.sentiment==1]
	words = ' '.join([text for text in raw['clean_tweet']])
	#text= (' '.join(df_positive['stemmed_tweet']))
	wordcloud = WordCloud(width = 1000, height = 500).generate(words)
	plt.figure(figsize=(15,10))
	plt.imshow(wordcloud)
	plt.axis('off')
	plt.show()
	st.pyplot()

	st.subheader("Understanding Common Words in Neutral Tweets")
	df_neutral = raw[raw.sentiment==0]
	text= (' '.join(df_neutral['clean_tweet']))
	wordcloud = WordCloud(width = 1000, height = 500).generate(text)
	plt.figure(figsize=(15,10))
	plt.imshow(wordcloud)
	plt.axis('off')
	plt.show()
	st.pyplot()

	st.subheader("Understanding Common Words in Factual Tweets")
	df_factual = raw[raw.sentiment==2]
	text= (' '.join(df_factual['clean_tweet']))
	wordcloud = WordCloud(width = 1000, height = 500).generate(text)
	plt.figure(figsize=(15,10))
	plt.imshow(wordcloud)
	plt.axis('off')
	plt.show()
	st.pyplot()

	st.subheader("Understanding Relationship of Hashtags and Sentiment of Tweet")
	pro_hashtags = []
	for message in df_pro['message']:
		hashtag = re.findall(r"#(\w+)", message)
		pro_hashtags.append(hashtag)

	pro_hashtags = sum(pro_hashtags,[])
	a = nltk.FreqDist(pro_hashtags)
	d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

	# selecting top 10 most frequent hashtags     
	d = d.nlargest(columns="Count", n = 10) 
	plt.figure(figsize=(10,5))
	ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
	plt.setp(ax.get_xticklabels(),rotation='vertical', fontsize=10)
	plt.title('Top 10 Hashtags in "Pro" Tweets', fontsize=14)
	plt.show()
	st.pyplot()
	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
