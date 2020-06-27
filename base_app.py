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
#@st.cache(persist=True)
raw = pd.read_csv("resources/train.csv")
#@st.cache(persist=True)
data_v = raw.copy()
m = pd.read_csv("model.csv")
# Load clean dataset
#@st.cache(persist=True)
clean_data = pd.read_csv("clean_data.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	st.sidebar.title("Multiclass Classification Web App")
	options = ["Prediction", "Information","EDA","TSNE plot","About Machine Learning App","Instruction of use"]
	selection = st.sidebar.selectbox("Choose Option", options)
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
	#@st.cache(persist=True)
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
	#@st.cache(persist=True)
	def clean_dataframe(data):
    	#"drop nans, then apply 'clean_sentence' function "
		data = data.dropna(how="any")
    
		for col in ['message']:
			data[col] = data[col].apply(clean_sentence)
    
		return data
	data_v1 = clean_dataframe(data_v)	
	#@st.cache(persist=True)
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
	#@st.cache(persist=True)
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
	#@st.cache(persist=True)
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
	Classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression","Linear Support Vector","Random Forest"))
    
	# if Classifier == 'Support Vector Machine (SVM)':
	# 	st.sidebar.subheader("Model Hyperparameters")
	# 	C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0, step=0.01, key='C')
	# 	kernel = st.sidebar.radio("Kernel",("rbf","linear"),key='kernel')
	# 	gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale","auto"),key ='gamma')
        
	# 	metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

	# 	if st.sidebar.button("Classify", key="classify"):
	# 		st.subheader("Support Vector Machine (SVM) Results")
	# 		model = SVC(C=C, kernel=kernel, gamma=gamma)
	# 		#model.fit(x_train, y_train)
	# 		model.fit(X_train_tfidf,y_train)

	# 		text_classifier = Pipeline([
    # 			('bow',CountVectorizer()),  # strings to token integer counts
    # 			('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    # 			('classifier',SVC(C=C, kernel=kernel, gamma=gamma)),  # train on TF-IDF vectors w/ Linear Support Vector Classifier
	# 		])
	# 		text_classifier.fit(X_train, y_train)
	# 		accuracy = text_classifier.score(X_test, y_test)
	# 		y_pred  = text_classifier.predict(X_test)
	# 		st.write("Accuracy: ", accuracy.round(2))
	# 		st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
	# 		st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
	# 		plot_metrics(metrics)	

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


	# if Classifier == 'K-nearest neighbours':
	# 	st.sidebar.subheader("Model Hyperparameters")
	# 	n_neighbors = st.sidebar.slider("Number of nearest neighbours", 1, 50, key='n_neighbors')
        
	# 	metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

	# 	if st.sidebar.button("Classify", key="classify"):
	# 		st.subheader("K-nearest neighbours(KNN) Results")
	# 		model = KNeighborsClassifier(n_neighbors =n_neighbors)
	# 		model.fit(X_train_tfidf,y_train)

	# 		text_classifier = Pipeline([
	# 			('bow',CountVectorizer()),  # strings to token integer counts
	# 			('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
	# 			('classifier',KNeighborsClassifier(n_neighbors =n_neighbors)),  # train on TF-IDF vectors w/ K-nearest neighbours
	# 		])
	# 		text_classifier.fit(X_train, y_train)
	# 		accuracy = text_classifier.score(X_test, y_test)
	# 		y_pred  = text_classifier.predict(X_test)
	# 		st.write("Accuracy: ", accuracy.round(2))
	# 		st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,average='micro').round(2))
	# 		st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,average='micro').round(2))
	# 		plot_metrics(metrics)
	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
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
		all_ml_models = ["Linear Support Vector","LogisticRegression","K-nearest neighbour"]
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
			elif  model_choice == 'K-nearest neighbour':
				model = KNeighborsClassifier(n_neighbors = 3)
				model.fit(X_train_tfidf, y_train)
				text_classifier = Pipeline([
						('bow',CountVectorizer(lowercase=False)),  # strings to token integer counts
						('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
						('classifier',KNeighborsClassifier(n_neighbors = 3)),  # train on TF-IDF vectors w/ Linear Support Vector Classifier
				])
				text_classifier.fit(X_train, y_train)			
				
				prediction = text_classifier.predict(pd.Series(tweet_text.split(",")))

			elif model_choice == 'LogisticRegression':
				model =  LogisticRegression(C=1, max_iter=10)
				model.fit(X_train_tfidf,y_train)
				text_classifier = Pipeline([
						('bow',CountVectorizer(lowercase=False)),  # strings to token integer counts
						('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
						('classifier',LogisticRegression(C=1, max_iter=10)),  # train on TF-IDF vectors w/ Linear Support Vector Classifier
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
	if selection == "About Machine Learning App":
		st.title("Welcome to the climate change Classification Machine Learning App")
		st.subheader('Machine Learning')
		st.markdown('<p>Machine learning (ML) is the study of computer algorithms that improve automatically through experience.It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so.Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or infeasible to develop conventional algorithms to perform the needed tasks. </p>', unsafe_allow_html=True)
		st.subheader('Machine Learning Algorithms')
		st.markdown('<p>A machine learning (ML) algorithm is essentially a process or sets of procedures that helps a model adapt to the data given an objective. An ML algorithm normally specifies the way the data is transformed from input to output and how the model learns the appropriate mapping from input to output. </p>', unsafe_allow_html=True)
		st.subheader('Climate change tweet Classification')		
		st.markdown('<p>This Application is used for classify tweets into four different categories.The tweet can be News , Pro ,Neutral or Anti . The four categories are described as follows:</p>', unsafe_allow_html=True)	
		st.markdown('<p>2(News): the tweet links to factual news about climate change </p>', unsafe_allow_html=True)
		st.markdown('<p>1(Pro): the tweet supports the belief of man-made climate change </p>', unsafe_allow_html=True)
		st.markdown('<p>0(Neutral): the tweet neither supports nor refutes the belief of man-made climate change </p>', unsafe_allow_html=True)
		st.markdown('<p>-1(Anti): the tweet does not believe in man-made climate change </p>', unsafe_allow_html=True)
		st.markdown('<p>We use machine learning algorithms to classify the tweets so for this application we use the following algorithms to classify tweets: </p>', unsafe_allow_html=True)
		st.markdown('<p><strong>LogisticRegression</strong> </p>', unsafe_allow_html=True)	
		st.markdown('<p>Logistic regression is a supervised learning classification algorithm used to predict the probability of a target variable. The nature of target or dependent variable is dichotomous, which means there would be only two possible classes. </p>', unsafe_allow_html=True)
		st.markdown('<p><strong>K-nearest neighbour</strong></p>', unsafe_allow_html=True)
		st.markdown('<p>K-nearest neighbors (KNN) algorithm is a type of supervised ML algorithm which can be used for both classification as well as regression predictive problems. However, it is mainly used for classification predictive problems in industry. </p>', unsafe_allow_html=True)	
		st.markdown('<p><strong>Random forest</strong></p>', unsafe_allow_html=True)	
		st.markdown('<p>Random forest is a supervised learning algorithm which is used for both classification as well as regression. But however, it is mainly used for classification problems. As we know that a forest is made up of trees and more trees means more robust forest. Similarly, random forest algorithm creates decision trees on data samples and then gets the prediction from each of them and finally selects the best solution by means of voting. It is an ensemble method which is better than a single decision tree because it reduces the over-fitting by averaging the result. </p>', unsafe_allow_html=True)	
		st.markdown('<p><strong>Linear support vector classifier</strong></p>', unsafe_allow_html=True)	
		st.markdown('<p>Support vector machines (SVMs) are powerful yet flexible supervised machine learning algorithms which are used both for classification and regression. But generally, they are used in classification problems. Lately, they are extremely popular because of their ability to handle multiple continuous and categorical variables. </p>', unsafe_allow_html=True)	
		st.markdown('<p>For more information about building data Apps Please go to :<a href="https://www.streamlit.io/">Here</a></p>', unsafe_allow_html=True)	
		st.markdown('<p> </p>', unsafe_allow_html=True)	

	if selection == "Instruction of use":
		st.title("Instructions")
		st.markdown('When the application opens the first page you will see is the prediction page.')
		st.image('images/prediction_p.png', width=600)
		st.markdown(' Here as a user you will put your tweet on the text box and then choose a machine learning algorithm to classify the tweet and then when it has been selected then you can press the classify button ')
		st.image('images/algorithms_p.png', width=600)
		st.markdown('You can find more about machine learning and the algorithm used in this application on the "About Machine Learning App" page.\
		On the top right you will see an arrow and when you click it you will see a sidebar .\
		The sidebar has a title then a selection box .\
		When you click the selection box you will see all the pages on the app the you can select the page you want to see.')
		st.image('images/algorithms.png', width=600)
		st.markdown('The EDA page is for all the insights we found on the dataset we used.\
		We have another page for t-sne plot of the words you can find on the dataset we used.\
		Below the selection box we have a section for the algorithms and their hyperparameters and you can change the hyperparameters to see how the Accuracy, Recall , confusion matrix , then after you change the hyperparameters you can plot the confusion matrix and then press the classify button to see the recall , accuracy and confusion matrix.', unsafe_allow_html=True)		
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
