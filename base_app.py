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
#nltk.download('stopwords')
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
#pip install wordCloud
 #import nltk
 #nltk.download('stopwords')
 #nltk.download('wordnet')
# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	st.sidebar.title("Multiclass Classification Web App")
	st.sidebar.markdown("Is the tweet News , Pro , Neutral or Anti climate-change ? ")
	#Removing Twitter Handles
	#@st.cache(persist=True)
	def remove_twitter_handles(tweet, pattern):
		r = re.findall(pattern, tweet)
		for text in r:
			tweet = re.sub(text, '', tweet)
		return tweet

	raw['clean_tweet'] = np.vectorize(remove_twitter_handles)(raw['message'], "@[\w]*") 

	#Removing Stopwords
	stop_words = nltk.corpus.stopwords.words('english')
	raw['tidy_tweet'] = raw['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))
	#Text Normalization
	#@st.cache(persist=True)
	def tokenizing(text):
		text = re.split('\W+', text)
		return text

	raw['tokenized_tweet'] = raw['tidy_tweet'].apply(lambda x: tokenizing(x))

	tokens = raw['tokenized_tweet']

	lemmatizer = WordNetLemmatizer()

	tokens = tokens.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])

	raw['lemmatized_tweet'] = tokens
	
    # Split the data
	X = raw['lemmatized_tweet']
	y = raw['sentiment']
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

	X_train = X_train.str.join(', ')
	X_test = X_test.str.join(', ')
	vectorizer = TfidfVectorizer()
	X_train_tfidf = vectorizer.fit_transform(X_train)

	def plot_metrics(metrics_list):
		if 'Confusion Matrix' in metrics_list:
			st.subheader("Confusion Matrix")
			plot_confusion_matrix(model, X_test, y_test, display_labels=class_names)
			st.pyplot()

		if 'ROC Curve' in metrics_list:
			st.subheader("ROC Curve")
			plot_roc_curve(model, X_test, y_test)
			st.pyplot() 

		if 'Precision-Recall Curve' in metrics_list:
			st.subheader("Precision-Recall Curve")
			plot_precision_recall_curve(model, X_test, y_test)
			st.pyplot()

	class_names = ['News','Pro','Neutral','Anti']
	st.sidebar.subheader("Choose Classifier")
	Classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression","Random Forest"))
    
	if Classifier == 'Support Vector Machine (SVM)':
		st.sidebar.subheader("Model Hyperparameters")
		C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0, step=0.01, key='C')
		kernel = st.sidebar.radio("Kernel",("rbf","linear"),key='kernel')
		gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale","auto"),key ='gamma')
        
		metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

		if st.sidebar.button("Classify", key="classify"):
			st.subheader("Support Vector Machine (SVM) Results")
			model = SVC(C=C, kernel=kernel, gamma=gamma)
			#model.fit(x_train, y_train)
			model.fit(X_train_tfidf,y_train)

			text_classifier = Pipeline([
    			('bow',CountVectorizer()),  # strings to token integer counts
    			('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    			('classifier',SVC(C=C, kernel=kernel, gamma=gamma)),  # train on TF-IDF vectors w/ Linear Support Vector Classifier
			])
			text_classifier.fit(X_train, y_train)
			accuracy = text_classifier.score(X_test, y_test)
			y_pred  = text_classifier.predict(X_test)
			st.write("Accuracy: ", accuracy.round(2))
			#st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
			#st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)	

	if Classifier == 'Logistic Regression':
		st.sidebar.subheader("Model Hyperparameters")
		C = st.sidebar.number_input("C (Regularization parameter)",0.01,100.0, step=0.01, key='C_LR')
		max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        
		metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

		if st.sidebar.button("Classify", key="classify"):
			st.subheader("Logistic Regression Results")
			model = LogisticRegression(C=C, max_iter=max_iter)
			model.fit(X_train_tfidf, y_train)

			text_classifier = Pipeline([
    			('bow',CountVectorizer()),  # strings to token integer counts
    			('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    			('classifier',LogisticRegression(C=C, max_iter=max_iter)),  # train on TF-IDF vectors w/ Linear Support Vector Classifier
				])
			text_classifier.fit(X_train, y_train)
			accuracy = text_classifier.score(X_test, y_test)
			y_pred  = model.predict(X_test)
			st.write("Accuracy: ", accuracy.round(2))
            	#st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            	#st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)


	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information","EDA"]
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
	# Building out the EDA page	
	if selection == "EDA":
		#st.title("Insights on people's peception on Climate Change ")

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

		st.subheader("Understanding Common Words in the Negative Tweets")
		df_anti = raw[raw.sentiment==-1]
		text= (' '.join(df_anti['clean_tweet']))
		wordcloud = WordCloud(width = 1000, height = 500).generate(text)
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
		plt.setp(ax.get_xticklabels(),rotation=17, fontsize=10)
		plt.title('Top 10 Hashtags in "Pro" Tweets', fontsize=14)
		plt.show()
		st.pyplot()
		print('\n')
		print('\n')
		#st.subheader("Understanding Relationship of Hashtags and Sentiment of Tweet")
		anti_hashtags = []
		for message in df_anti['message']:
			hashtag = re.findall(r"#(\w+)", message)
			anti_hashtags.append(hashtag)

		anti_hashtags = sum(anti_hashtags,[])


		a = nltk.FreqDist(anti_hashtags)
		d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

		# selecting top 20 most frequent hashtags     
		d = d.nlargest(columns="Count", n = 10) 
		plt.figure(figsize=(10,5))
		ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
		plt.setp(ax.get_xticklabels(),rotation=17, fontsize=10)
		plt.title('Top 10 Hashtags in "Anti" Tweets', fontsize=14)
		plt.show()
		st.pyplot()
		print('\n')

		neutral_hashtags = []
		for message in df_neutral['message']:
			hashtag = re.findall(r"#(\w+)", message)
			neutral_hashtags.append(hashtag)

		neutral_hashtags = sum(neutral_hashtags,[])


		a = nltk.FreqDist(neutral_hashtags)
		d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

		# selecting top 20 most frequent hashtags     
		d = d.nlargest(columns="Count", n = 10) 
		plt.figure(figsize=(10,5))
		ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
		plt.setp(ax.get_xticklabels(),rotation=17, fontsize=10)
		plt.title('Top 10 Hashtags in Neutral Tweets', fontsize=14)
		plt.show()
		st.pyplot()

		print("\n")
		factual_hashtags = []
		for message in df_factual['message']:
			hashtag = re.findall(r"#(\w+)", message)
			factual_hashtags.append(hashtag)

		factual_hashtags = sum(factual_hashtags,[])


		a = nltk.FreqDist(factual_hashtags)
		d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

		# selecting top 20 most frequent hashtags     
		d = d.nlargest(columns="Count", n = 10) 
		plt.figure(figsize=(10,5))
		ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
		plt.setp(ax.get_xticklabels(),rotation=17, fontsize=10)
		plt.title('Top 10 Hashtags in Factual Tweets', fontsize=14)
		plt.show()
		st.pyplot()	
		
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
