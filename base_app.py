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
import emoji
# Before you can run this script on your local computer please download the following 
#pip install wordCloud
#nltk.download('stopwords')
#nltk.download('wordnet')
#pip install emoji --upgrade
# pip install gensim
# pip install nltk

# Load your raw data
raw = pd.read_csv("resources/train.csv")
# copy raw data
data_v = raw.copy()
# Load clean dataset
clean_data = pd.read_csv("clean_data.csv")
# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	st.sidebar.title("Multiclass Classification Web App")
	options = ["Prediction", "Information","EDA","About Machine Learning App","Instruction of use","Emojis"]
	selection = st.sidebar.selectbox("Choose Option", options)
    # Split the data
	X = clean_data['lemmatized_tweet']
	y =clean_data['sentiment']
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

	# vectorize the data
	vectorizer = TfidfVectorizer()

	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(X_train)
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

	X_test_counts = count_vect.transform(X_test)
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)

	def plot_metrics(metrics_list):
		if 'Confusion Matrix' in metrics_list:
			st.subheader("Confusion Matrix")
			class_names = [-1, 0, 1, 2]# name  of classes
			plot_confusion_matrix(model, X_test_tfidf, y_test, display_labels=class_names,cmap=plt.cm.Blues,normalize='true')
			st.pyplot()
	

	class_names = [-1, 0, 1, 2]
	st.sidebar.subheader("Choose Classifier")
	Classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression","Linear Support Vector","Random Forest"))
    
	

	if Classifier == 'Logistic Regression':
		st.sidebar.subheader("Model Hyperparameters")
		C = st.sidebar.number_input("C (Regularization parameter)",1.0,100.0, step=0.01, key='C_LR')
		max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

		option_s = ['Confusion Matrix']
		metrics = st.sidebar.selectbox("Plot Confusion Matrix",option_s)

		if st.sidebar.button("Classify", key="classify"):
			st.subheader("Logistic Regression Results")
			model = LogisticRegression(C=C, max_iter=max_iter)
			model.fit(X_train_tfidf, y_train)

			text_classifier = Pipeline([
    			('bow',CountVectorizer()),  # strings to token integer counts
    			('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    			('classifier',LogisticRegression(C=C, max_iter=max_iter)),  # train on TF-IDF vectors w/ LogisticRegression
				])
			text_classifier.fit(X_train, y_train)
			accuracy = text_classifier.score(X_test, y_test)
			y_pred  = text_classifier.predict(X_test)
			st.write("Accuracy: ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,average='micro').round(2))	
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,average='micro').round(2))
			plot_metrics(metrics)

	if Classifier == 'Random Forest':
		st.sidebar.subheader("Model Hyperparameters")
		n_estimators =  st.sidebar.number_input("The number of trees in the forest",100,5000,step =10, key="n_estimators")
		max_depth = st.sidebar.number_input("The maximum depth of the tree",1,20,step=1,key="max_depth")
		bootstrap = st.sidebar.radio("Bootstrap samples when building trees",("True","False"), key="bootstrap")
		option_s = ['Confusion Matrix']
		metrics = st.sidebar.selectbox("Plot confusion matrix",option_s)

		if st.sidebar.button("Classify", key="classify"):
			st.subheader("LRandom Forest Results")
			model =RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap,n_jobs=-1)
			model.fit(X_train_tfidf, y_train)
			text_classifier = Pipeline([
				('bow',CountVectorizer()),  # strings to token integer counts
				('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
				('classifier',RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap,n_jobs=-1)),  # train on TF-IDF vectors w/ Random Forest
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
		option_s = ['Confusion Matrix']
		metrics = st.sidebar.selectbox("Plot confusion matrix",option_s)

		if st.sidebar.button("Classify", key="classify"):
			st.subheader("Linear Support Vector Classifier(LSVC) Results")
			model = LinearSVC(C=C, multi_class='ovr')
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

	

	# Creating sidebar with selection box -
	if selection == "Emojis":
		if st.checkbox('Show some emojis in the data'):
			emojis_list = [':sun::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::United_States::trade_mark::face_with_tears_of_joy::face_with_tears_of_joy::United_States::United_States:\
 						  :zipper-mouth_face::zipper-mouth_face::party_popper::thinking_face::face_with_tears_of_joy::face_with_tears_of_joy::trade_mark::trade_mark::copyright::fire::trade_mark:\
						  :squinting_face_with_tongue::face_with_tongue::face_with_tongue::trade_mark::trade_mark::trade_mark::oncoming_fist_light_skin_tone::middle_finger_light_skin_tone:\
 						  :trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::sparkler::sparkler::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark:',':trade_mark:\
 						  :trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::snowflake::trade_mark::nail_polish_medium_skin_tone:\
 						  :left_arrow::trade_mark::trade_mark::grinning_face::grinning_face_with_sweat::police_car_light::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark:\
 						  :rolling_on_the_floor_laughing::rolling_on_the_floor_laughing::rolling_on_the_floor_laughing::trade_mark::trade_mark::trade_mark::trade_mark::thinking_face::smiling_face_with_sunglasses::trade_mark:\
						  :face_with_tears_of_joy::face_with_tears_of_joy::face_with_tears_of_joy::face_with_tears_of_joy:\
 						  :trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark:\
 						  :ear_light_skin_tone::pile_of_poo::ear_light_skin_tone::part_alternation_mark::exclamation_mark::trade_mark::trade_mark::trade_mark::trade_mark::hugging_face::trade_mark::clown_face::shuffle_tracks_button:\
						  :trade_mark::trade_mark::trade_mark::anxious_face_with_sweat::astonished_face::confused_face:\
 						  :angry_face::angry_face::angry_face::angry_face::angry_face::angry_face::person_wearing_turban::thumbs_down::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark:\
						  :trade_mark::face_with_rolling_eyes::female_sign::winking_face_with_tongue::trade_mark::snowflake::snowflake::snowflake::trade_mark::trade_mark::thinking_face::face_with_tears_of_joy::trade_mark:\
 						  :trade_mark::movie_camera::backhand_index_pointing_right_light_skin_tone::face_with_tears_of_joy::double_exclamation_mark::sparkler::trade_mark:\
 						  :beaming_face_with_smiling_eyes::beaming_face_with_smiling_eyes::face_with_tears_of_joy::face_with_tears_of_joy::trade_mark::trade_mark::trade_mark::flushed_face::flushed_face::flushed_face::face_with_rolling_eyes::face_with_rolling_eyes:\
						  :trade_mark::trade_mark::trade_mark::trade_mark::face_with_tears_of_joy::snowflake::snowflake::male_sign::backhand_index_pointing_right::trade_mark::face_with_tears_of_joy::trade_mark::hugging_face::registered:\
 						  :face_with_tears_of_joy::trade_mark::trade_mark::face_with_tears_of_joy::face_with_rolling_eyes::trade_mark::trade_mark::double_exclamation_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark:\
 						  :trade_mark::trade_mark::trade_mark::trade_mark::sleeping_face::trade_mark::grinning_face_with_sweat::trade_mark::trade_mark::trade_mark::thinking_face::see-no-evil_monkey::victory_hand::trade_mark::copyright::trade_mark::trade_mark:\
 						  :smiling_face_with_smiling_eyes::grinning_face::face_with_tears_of_joy::grinning_face_with_big_eyes::face_with_tears_of_joy::smiling_face_with_smiling_eyes:\
						  :face_with_tears_of_joy::grinning_face_with_big_eyes::grinning_face_with_big_eyes::grinning_face_with_big_eyes::face_with_tears_of_joy::grinning_face::grinning_face::grinning_face_with_big_eyes::grinning_face_with_big_eyes::face_with_tears_of_joy:\
						  :beaming_face_with_smiling_eyes::beaming_face_with_smiling_eyes::face_with_tears_of_joy::grinning_face_with_big_eyes::grinning_face_with_big_eyes::grinning_face_with_big_eyes::beaming_face_with_smiling_eyes::beaming_face_with_smiling_eyes::face_with_tears_of_joy:\
						  :grinning_face_with_big_eyes::grinning_face_with_big_eyes::face_with_tears_of_joy::grinning_face_with_big_eyes::grinning_face_with_big_eyes::honeybee::skull::United_States::trade_mark::trade_mark::trade_mark::grinning_face_with_smiling_eyes::trade_mark:',':trade_mark:\
 						  :trade_mark::trade_mark::trade_mark::trade_mark::female_sign::copyright::fire::fire::fire::trade_mark::flushed_face::trade_mark::film_frames::blossom::trade_mark::trade_mark::trade_mark::trade_mark::copyright::play_button:\
 						  :trade_mark::trade_mark::play_button::face_with_steam_from_nose::pile_of_poo::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::skier::thinking_face::face_with_tears_of_joy::sun::trade_mark::expressionless_face::sparkles::smiling_face:\
 						  :neutral_face::face_with_tears_of_joy::face_with_tears_of_joy::snowflake::snowflake::snowflake::trade_mark::weary_face::confused_face::loudly_crying_face::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::sun:\
 						  :trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::smiling_face_with_smiling_eyes::sun::smiling_face_with_heart-eyes::smiling_face_with_heart-eyes::tired_face::tired_face::tired_face::hundred_points::hundred_points::OK_hand_medium_skin_tone::fire::fire::weary_face::trade_mark::trade_mark:\
 						  :trade_mark::hand_with_fingers_splayed::mountain::trade_mark::trade_mark::snowflake::fire::fire::smirking_face::face_with_tears_of_joy::face_with_tears_of_joy::face_with_tears_of_joy::face_with_tears_of_joy::smiling_face_with_heart-eyes::face_with_tears_of_joy::face_with_tears_of_joy:\
 						  :rolling_on_the_floor_laughing::face_with_tears_of_joy::rolling_on_the_floor_laughing::face_with_tears_of_joy::rolling_on_the_floor_laughing::face_with_tears_of_joy::trade_mark::face_with_tears_of_joy::face_with_tears_of_joy::face_with_tears_of_joy:\
 						  :grinning_face_with_smiling_eyes::thinking_face::trade_mark::trade_mark::trade_mark::beaming_face_with_smiling_eyes::beaming_face_with_smiling_eyes::trade_mark::trade_mark::face_with_tears_of_joy::face_with_tears_of_joy::trade_mark::index_pointing_up::cyclone::face_with_tears_of_joy:\
 						  :trade_mark::trade_mark::red_heart:',':trade_mark::face_with_tears_of_joy::play_button::trade_mark::copyright::copyright::copyright::copyright:::trade_mark::trade_mark::globe_showing_Americas::collision::face_with_rolling_eyes::trade_mark::frog_face::hot_beverage:\
 						  :loudly_crying_face::loudly_crying_face::loudly_crying_face::loudly_crying_face::loudly_crying_face::hundred_points::raising_hands_light_skin_tone::trade_mark::trade_mark::trade_mark::slightly_frowning_face::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::person_shrugging_light_skin_tone::female_sign::trade_mark:\
 						  :trade_mark::trade_mark::ghost::face_with_tears_of_joy::umbrella_with_rain_drops::trade_mark::play_button::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::airplane::trade_mark::trade_mark::thinking_face::grinning_face::skull::face_with_tears_of_joy::eyes:\
 						  :tired_face::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::face_with_tears_of_joy::trade_mark::fire::trade_mark::victory_hand::face_with_tears_of_joy::fire::globe_showing_Americas::snowflake::sun::folded_hands_light_skin_tone::face_with_tears_of_joy::baby_angel::ring:\
 						  :red_heart::face_with_tears_of_joy::face_with_tears_of_joy::face_with_tears_of_joy::grinning_squinting_face::grinning_squinting_face::grinning_squinting_face::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::grinning_cat_face::sweat_droplets::face_with_tears_of_joy::trade_mark::trade_mark:\
						  :play_button::trade_mark::trade_mark::fire::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::face_with_rolling_eyes::peace_symbol::trade_mark::trade_mark::grinning_face_with_sweat::globe_showing_Asia-Australia::trade_mark::trade_mark:\
 						  :trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::angry_face::trade_mark::trade_mark::squinting_face_with_tongue::face_with_tears_of_joy::face_with_tears_of_joy::winking_face_with_tongue::trade_mark::trade_mark::trade_mark::red_heart::trade_mark::trade_mark:\
 						  :grinning_face_with_sweat::grinning_face_with_sweat::grinning_face_with_sweat::trade_mark::trade_mark::trade_mark::Cancer::trade_mark::face_with_tears_of_joy::trade_mark::trade_mark::trade_mark::copyright::copyright::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::telescope::microscope::magnifying_glass_tilted_left::flashlight:\
 						  :trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::face_with_tears_of_joy::face_with_tears_of_joy::loudly_crying_face::face_with_thermometer::face_with_tears_of_joy::trade_mark::sad_but_relieved_face::zipper-mouth_face::trade_mark::trade_mark::trade_mark::red_heart::trade_mark::trade_mark::trade_mark::trade_mark::female_sign:\
 						  :trade_mark::sun_with_face::sun_with_face::sun_with_face::sun_with_face::thinking_face::face_with_tears_of_joy::copyright::trade_mark::trade_mark::trade_mark::eyes::trade_mark::trade_mark::trade_mark::trade_mark::squinting_face_with_tongue::green_heart::trade_mark::trade_mark::trade_mark::tired_face::trade_mark::trade_mark:\
 						  :trade_mark::hot_beverage::face_without_mouth::face_without_mouth::face_with_tears_of_joy::trade_mark::trade_mark::trade_mark::trade_mark::trade_mark::face_with_tears_of_joy::registered::trade_mark::trade_mark::trade_mark::smirking_face::trade_mark::fast-forward_button::exclamation_question_mark::exclamation_question_mark:\
 						  :trade_mark::trade_mark::red_heart::trade_mark::face_with_medical_mask::snowflake::cloud_with_snow::face_with_tears_of_joy::winking_face_with_tongue::trade_mark::trade_mark::pensive_face::unamused_face::trade_mark::trade_mark::snowflake::trade_mark::trade_mark::trade_mark:']
			for i in emojis_list:
				list_emojis = emoji.emojize(i)	
				st.markdown(list_emojis)

	# Building out the "Information" page
	if selection == "Information":
		st.subheader("Climate change tweet classification")
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.subheader("Climate change tweet classification")
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		all_ml_models = ["Linear Support Vector","LogisticRegression","K-nearest neighbour"]
		model_choice = st.selectbox("Choose ML Model",all_ml_models)

		if st.button("Classify"):
			# We loading in multiple models to give the user a choice
			if model_choice == 'Linear Support Vector':
				model = LinearSVC(C=1, multi_class='ovr')
				model.fit(X_train_tfidf,y_train)
				text_classifier = Pipeline([
					('bow',CountVectorizer(lowercase=False)),  # strings to token integer counts
					('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
					('classifier',LinearSVC()),  # train on TF-IDF vectors w/ Linear Support Vector Classifier
				])
				text_classifier.fit(X_train, y_train)
				prediction = text_classifier.predict(pd.Series(tweet_text.split(",")))
				
			elif  model_choice == 'K-nearest neighbour':
				model = KNeighborsClassifier(n_neighbors = 3)
				model.fit(X_train_tfidf, y_train)
				text_classifier = Pipeline([
						('bow',CountVectorizer(lowercase=False)),  # strings to token integer counts
						('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
						('classifier',KNeighborsClassifier(n_neighbors = 3)),  # train on TF-IDF vectors w/ K-neighbors Classifier
				])
				text_classifier.fit(X_train, y_train)			
				prediction = text_classifier.predict(pd.Series(tweet_text.split(",")))

			elif model_choice == 'LogisticRegression':
				model =  LogisticRegression(C=1, max_iter=10)
				model.fit(X_train_tfidf,y_train)
				text_classifier = Pipeline([
						('bow',CountVectorizer(lowercase=False)),  # strings to token integer counts
						('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
						('classifier',LogisticRegression(C=1, max_iter=10)),  # train on TF-IDF vectors w/ LogisticRegression
				])
				text_classifier.fit(X_train, y_train)
				prediction = text_classifier.predict(pd.Series(tweet_text.split(",")))
			# When model has successfully run, will print prediction	
			st.success("Text Categorized as: {}".format(prediction))

	# Building out the EDA page	
	if selection == "EDA":
		st.title("Insights on people's peception on climate change ")

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


		st.subheader("Understanding Relationship of Hashtags and Sentiment of Tweets")
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

		# selecting top 10 most frequent hashtags     
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

			# selecting top 10 most frequent hashtags     
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

			# selecting top 10 most frequent hashtags     
			d = d.nlargest(columns="Count", n = 10) 
			plt.figure(figsize=(10,5))
			ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
			plt.setp(ax.get_xticklabels(),rotation=17, fontsize=10)
			plt.title('Top 10 Hashtags in Factual Tweets', fontsize=14)
			st.pyplot()	

	# Building out the About Machine Learning App page		
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
		st.markdown('<p>For more information about building data Apps Please go to :<a href="https://www.streamlit.io/">streamlit site</a></p>', unsafe_allow_html=True)	
		st.markdown('<p> </p>', unsafe_allow_html=True)	
	# Building out the Instruction of use page
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
		st.image('images/pages.png', width=500)
		st.markdown('The EDA page is for all the insights we found on the dataset we used.')
		st.image('images/eda.png', width=700)
		st.markdown('We have another page for t-sne plot of the words you can find on the dataset we used.')
		st.image('images/t-sne.png', width=700)
		st.markdown('Below the selection box we have a section for the algorithms and their hyperparameters and you can change the hyperparameters to see how the Accuracy,Precision, Recall , confusion matrix , then after you change the hyperparameters you can plot the confusion matrix and then press the classify button to see the recall , accuracy ,precision and confusion matrix.', unsafe_allow_html=True)		
		st.image('images/classify.png', width=600)
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
