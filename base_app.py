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
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from itertools import cycle
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
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
pd.options.mode.chained_assignment = None 
from gensim import models
from gensim.models import word2vec

#pip install wordCloud
 #import nltk
 #nltk.download('stopwords')
 #nltk.download('wordnet')
 #nltk.download('stopwords')
# Vectorizer
#news_vectorizer = open("resources/tfidfvect.pkl","rb")
#tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
 @st.cache(persist=True)
raw = pd.read_csv("resources/train.csv")
 @st.cache(persist=True)
data_v = raw.copy()
m = pd.read_csv("model.csv")
# Load clean dataset
 @st.cache(persist=True)
clean_data = pd.read_csv("clean_data.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	st.sidebar.title("Multiclass Classification Web App")
	st.sidebar.markdown("Is the tweet News , Pro , Neutral or Anti climate-change ? ")


    # Split the data
	X = clean_data['lemmatized_tweet']
	y =clean_data['sentiment']
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

	#X_train = X_train.str.join(', ')
	#X_test = X_test.str.join(', ')
	vectorizer = TfidfVectorizer()
	#X_train1 = X_train.tolist()
	#X_train1 = ", ".join(X_train1)
	#X_train_tfidf = vectorizer.fit_transform(X_train)

	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(X_train)
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

	X_test_counts = count_vect.transform(X_test)
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)
	#X_test_tfidf = vectorizer.fit_transform(X_test)
	STOP_WORDS = nltk.corpus.stopwords.words()
	@st.cache(persist=True)
	def clean_sentence(val):
    	#"remove chars that are not letters or numbers, downcase, then remove stop words"
		regex = re.compile('([^\s\w]|_)+')
		sentence = regex.sub('', val).lower()
		sentence = sentence.split(" ")
    
		for word in list(sentence):
			if word in STOP_WORDS:
				sentence.remove(word)  
            
		sentence = " ".join(sentence)
		return sentence
 	@st.cache(persist=True)
	def clean_dataframe(data):
    	#"drop nans, then apply 'clean_sentence' function "
		data = data.dropna(how="any")
    
		for col in ['message']:
			data[col] = data[col].apply(clean_sentence)
    
		return data
	data_v1 = clean_dataframe(data_v)	
	@st.cache(persist=True)
	def build_corpus(data):
    	#"Creates a list of lists containing words from each sentence"
		corpus = []
		for col in ['message']:
			for sentence in data[col].iteritems():
				word_list = sentence[1].split(" ")
				corpus.append(word_list)
            
		return corpus

	corpus = build_corpus(data_v1)  
	modell = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)
	model11 = word2vec.Word2Vec(corpus, size=100, window=20, min_count=500, workers=4)
	@st.cache(persist=True)
	def tsne_plot(model):
		#Creates and TSNE model and plots it
		labels = []
		tokens = []

		for word in model.wv.vocab:
			tokens.append(model[word])
			labels.append(word)
    
		tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
		new_values = tsne_model.fit_transform(tokens)

		x = []
		y = []
		for value in new_values:
			x.append(value[0])
			y.append(value[1])
        
		plt.figure(figsize=(13, 7)) 
		for i in range(len(x)):
			plt.scatter(x[i],y[i])
			plt.annotate(labels[i],
                     xy=(x[i], y[i]),xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
		st.pyplot()	
	@st.cache(persist=True)
	def plot_metrics(metrics_list):
		if 'Confusion Matrix' in metrics_list:
			st.subheader("Confusion Matrix")
			#X_test1 = model.transform(X_test)
			#cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
			#cnf_matrix
			class_names = [-1, 0, 1, 2]# name  of classes
			#fig, ax = plt.subplots()
			#tick_marks = np.arange(len(class_names))
			#plt.xticks(tick_marks, class_names)
			#plt.yticks(tick_marks, class_names)
			# create heatmap
			#sns.heatmap(pd.DataFrame(metrics.confusion_matrix(np.array(y_test),np.array(y_pred))), annot=True, cmap="YlGnBu" ,fmt='g')
			#ax.xaxis.set_label_position("top")
			#plt.tight_layout()
			#plt.title('Confusion matrix', y=1.1)
			#plt.ylabel('Actual label')
			#plt.xlabel('Predicted label')
			plot_confusion_matrix(model, X_test_tfidf, y_test, display_labels=class_names,cmap=plt.cm.Blues,normalize='true')
			#plot_confusion_matrix(classifier, X_test, y_test,
			st.pyplot()

		if 'ROC Curve' in metrics_list:
			st.subheader("ROC Curve")
			y_score = model.fit(X_train_tfidf, y_train).decision_function(X_test_tfidf)
			class_names = [-1, 0, 1, 2]# name  of classes
			# roc curve
			fpr = dict()
			tpr = dict()

			for i in range(len(class_names)):
				fpr[i], tpr[i], _ = roc_curve(y_test[i], y_score[i])
				plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))

			plt.xlabel("false positive rate")
			plt.ylabel("true positive rate")
			plt.legend(loc="best")
			plt.title("ROC curve")
			#plt.show()
			#plot_roc_curve(model, X_test_tfidf, y_test,cmap=plt.cm.Blues,normalize='true' )
			st.pyplot() 

		if 'Precision-Recall Curve' in metrics_list:
			st.subheader("Precision-Recall Curve")
			plot_precision_recall_curve(model, X_test_tfidf, y_test)
			st.pyplot()

	class_names = [-1, 0, 1, 2]
	st.sidebar.subheader("Choose Classifier")
	Classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression","Random Forest","Linear Support Vector","K-nearest neighbours"))
    
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
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)	

	if Classifier == 'Logistic Regression':
		st.sidebar.subheader("Model Hyperparameters")
		C = st.sidebar.number_input("C (Regularization parameter)",1.0,100.0, step=0.01, key='C_LR')
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
			y_pred  = text_classifier.predict(X_test)
			st.write("Accuracy: ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,average='micro').round(2))	
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,average='micro').round(2))
			plot_metrics(metrics)
			# #if st.checkbox('Let\'s  play:'):
			# 	# try different Values of C and record testing accuracy
			# C_range = list(range(1,50))
			# scores = []
			# for c in C_range:
			# 	model = LogisticRegression(C=c)
			# 	model.fit(X_train_tfidf, y_train)
			# 	y_pred  = text_classifier.predict(X_test)
			# 	class_names = [-1,0,1,2]
			# 	scores.append(precision_score(y_test, y_pred, labels=class_names,average='micro').round(2))
			# plt.plot(C_range, scores)
			# plt.xlabel("Value of C for LogisticRegression")
			# plt.ylabel("Testing Accuracy")
			# st.pyplot()
	if Classifier == 'Random Forest':
		st.sidebar.subheader("Model Hyperparameters")
		n_estimators =  st.sidebar.number_input("The number of trees in the forest",100,5000,step =10, key="n_estimators")
		max_depth = st.sidebar.number_input("The maximum depth of the tree",1,20,step=1,key="max_depth")
		bootstrap = st.sidebar.radio("Bootstrap samples when building trees",("True","False"), key="bootstrap")
        
		metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

		if st.sidebar.button("Classify", key="classify"):
			st.subheader("LRandom Forest Results")
			model =RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap,n_jobs=-1)
			model.fit(X_train_tfidf, y_train)
			text_classifier = Pipeline([
				('bow',CountVectorizer()),  # strings to token integer counts
				('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
				('classifier',RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap,n_jobs=-1)),  # train on TF-IDF vectors w/ Linear Support Vector Classifier
				])
			text_classifier.fit(X_train, y_train)
			accuracy = text_classifier.score(X_test, y_test)
			y_pred  = text_classifier.predict(X_test)
			st.write("Accuracy: ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,average='micro').round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,average='micro').round(2))
			plot_metrics(metrics)

	if Classifier == 'Linear Support Vector':
		st.sidebar.subheader("Model Hyperparameters")
		C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0, step=0.01, key='C')
        
		metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

		if st.sidebar.button("Classify", key="classify"):
			st.subheader("Linear Support Vector Classifier(LSVC) Results")
			model = LinearSVC(C=C, multi_class='ovr')
			#model.fit(x_train, y_train)
			model.fit(X_train_tfidf,y_train)

			text_classifier = Pipeline([
				('bow',CountVectorizer()),  # strings to token integer counts
				('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
				('classifier',LinearSVC(C=C,multi_class='ovr')),  # train on TF-IDF vectors w/ Linear Support Vector Classifier
			])
			text_classifier.fit(X_train, y_train)
			accuracy = text_classifier.score(X_test, y_test)
			y_pred  = text_classifier.predict(X_test)
			st.write("Accuracy: ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,average='micro').round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,average='micro').round(2))
			plot_metrics(metrics)


	if Classifier == 'K-nearest neighbours':
		st.sidebar.subheader("Model Hyperparameters")
		n_neighbors = st.sidebar.slider("Number of nearest neighbours", 1, 50, key='n_neighbors')
        
		metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

		if st.sidebar.button("Classify", key="classify"):
			st.subheader("K-nearest neighbours(KNN) Results")
			model = KNeighborsClassifier(n_neighbors =n_neighbors)
			model.fit(X_train_tfidf,y_train)

			text_classifier = Pipeline([
				('bow',CountVectorizer()),  # strings to token integer counts
				('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
				('classifier',KNeighborsClassifier(n_neighbors =n_neighbors)),  # train on TF-IDF vectors w/ K-nearest neighbours
			])
			text_classifier.fit(X_train, y_train)
			accuracy = text_classifier.score(X_test, y_test)
			y_pred  = text_classifier.predict(X_test)
			st.write("Accuracy: ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,average='micro').round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,average='micro').round(2))
			plot_metrics(metrics)
	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["About Machine Learning App","Instruction of use","Prediction", "Information","EDA","TSNE plot"]
	selection = st.sidebar.selectbox("Choose Option", options)
	if selection == "TSNE plot":
		if st.checkbox('Show TSNE plot'):
			#md = np.array(m)
			tsne_plot(modell)
			tsne_plot(model11)

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
		all_ml_models = ["Linear Support Vector","Random Forest","LogisticRegression","K-nearest neighbour"]
		model_choice = st.selectbox("Choose ML Model",all_ml_models)

		if st.button("Classify"):
			# Transforming user input with vectorizer
			#X_train = tweet_text.join(', ')
			#vectorizer = TfidfVectorizer()
			#count_vect = CountVectorizer(lowercase=False)
			#user_text = count_vect.transform(tweet_text)
			#tfidf_user_text = vectorizer.transform(tweet_text)
			if model_choice == 'Linear Support Vector':
				model = LinearSVC(C=1, multi_class='ovr')
				#model.fit(x_train, y_train)
				model.fit(X_train_tfidf,y_train)
				text_classifier = Pipeline([
					('bow',CountVectorizer(lowercase=False)),  # strings to token integer counts
					('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
					('classifier',LinearSVC()),  # train on TF-IDF vectors w/ Linear Support Vector Classifier
				])
				text_classifier.fit(X_train, y_train)
				vect_text = tweet_text
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				#predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				#prediction = predictor.predict(vect_text)
				#predictor = joblib.load(open(os.path.join("resources/Linear_Support_Vector_Classifier_model.pkl"),"rb"))
				#predictor = joblib.load(open(os.path.join("resources/Linear_Support1.pkl"),"rb"))
				#model = joblib.load('model_question_topic.pkl')
				prediction = text_classifier.predict(pd.Series(tweet_text.split(",")))
				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
			elif  model_choice == 'Random Forest':
				model =RandomForestClassifier()
				model.fit(X_train_tfidf, y_train)
				text_classifier = Pipeline([
						('bow',CountVectorizer(lowercase=False)),  # strings to token integer counts
						('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
						('classifier',RandomForestClassifier()),  # train on TF-IDF vectors w/ Linear Support Vector Classifier
				])
				text_classifier.fit(X_train, y_train)			
				prediction = text_classifier.predict(pd.Series(tweet_text.split(",")))


			st.success("Text Categorized as: {}".format(prediction))

	# Building out the EDA page	
	if selection == "EDA":
		#st.title("Insights on people's peception on Climate Change ")

		st.subheader("Sentiment Data")
		if st.checkbox('Show sentiment data'):
			plt.figure(figsize=(12,6))
			sns.countplot(x='sentiment',data=clean_data, palette='CMRmap')
			plt.title('Number of Tweets per Class', fontsize=20)
			plt.xlabel('Number of Tweets', fontsize=14)
			plt.ylabel('Class', fontsize=14)
			st.pyplot()

		st.subheader("Understanding Common Words in the Positive Tweets")
		if st.checkbox('Show Positive Tweets'):
			df_pro = clean_data[clean_data.sentiment==1]
			words = ' '.join([text for text in clean_data['clean_tweet']])
			#text= (' '.join(df_positive['stemmed_tweet']))
			wordcloud = WordCloud(width = 1000, height = 500).generate(words)
			plt.figure(figsize=(15,10))
			plt.imshow(wordcloud)
			plt.axis('off')
			st.pyplot()

		st.subheader("Understanding Common Words in the Negative Tweets")
		if st.checkbox('Show Negative Tweets'):
			df_anti = clean_data[clean_data.sentiment==-1]
			text= (' '.join(df_anti['clean_tweet']))
			wordcloud = WordCloud(width = 1000, height = 500).generate(text)
			plt.figure(figsize=(15,10))
			plt.imshow(wordcloud)
			plt.axis('off')
			st.pyplot()

		st.subheader("Understanding Common Words in Neutral Tweets")
		if st.checkbox('Show Neutral Tweets'):
			df_neutral = clean_data[clean_data.sentiment==0]
			text= (' '.join(df_neutral['clean_tweet']))
			wordcloud = WordCloud(width = 1000, height = 500).generate(text)
			plt.figure(figsize=(15,10))
			plt.imshow(wordcloud)
			plt.axis('off')
			st.pyplot()

		st.subheader("Understanding Common Words in Factual Tweets")
		if st.checkbox('Show Factual Tweets'):
			df_factual = clean_data[clean_data.sentiment==2]
			text= (' '.join(df_factual['clean_tweet']))
			wordcloud = WordCloud(width = 1000, height = 500).generate(text)
			plt.figure(figsize=(15,10))
			plt.imshow(wordcloud)
			plt.axis('off')
			st.pyplot()


		st.subheader("Understanding Relationship of Hashtags and Sentiment of Tweet")
		if st.checkbox('Show Pro hashtags Tweets'):
			df_pro = clean_data[clean_data.sentiment==1]
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
			st.pyplot()

		#st.subheader("Understanding Relationship of Hashtags and Sentiment of Tweet")
		if st.checkbox('Show Anti hashtags Tweets'):
			df_anti = clean_data[clean_data.sentiment==-1]
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
			st.pyplot()

		if st.checkbox('Show Neutral hashtags Tweets'):
			df_neutral = clean_data[clean_data.sentiment==0]
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
			st.pyplot()

		if st.checkbox('Show Factual hashtags Tweets'):
			df_factual = clean_data[clean_data.sentiment==2]
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
			st.pyplot()	
		
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
