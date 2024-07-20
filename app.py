import streamlit as st
import pickle
import string 
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from googletrans import Translator

ps = PorterStemmer()
translator = Translator()

def transform_text(text):
    #1.
    text = text.lower()
    #2.
    text = nltk.word_tokenize(text)
    #3.
    y=[]
    for i in text:
        if i.isalnum(): #if i is alpha numeric
            y.append(i)
            
    text = y[:]
    y.clear()
    #4.
    for i in text:
         if i not in stopwords.words('english') and i not in string.punctuation:
             y.append(i)
    
    text = y[:]
    y.clear()
    #5.
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)
    

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
model3 = pickle.load(open('model3.pkl', 'rb'))

st.title("SpamShield SMS")
st.sidebar.header("About SpamShield")
st.sidebar.write("Welcome to the SpamShield SMS! This project is designed to help identify spam messages using various machine learning models.")
st.sidebar.write("Sample Spam SMS: England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND.")
st.sidebar.write("Sample Non Spam SMS: Hey how are u doing")
input_text = st.text_area("Enter SMS (Hindi or English):")
col1, col2 = st.columns(2)
translated_text = None
with col2:
    if st.button('Translate to English'):
        translated_text = translator.translate(input_text, src='hi', dest='en').text
        st.write(f"Translated to English: {translated_text}")
with col1:
    if st.button('Predict'):
        if translated_text:
            transformed_sms = transform_text(translated_text)
        else:
            transformed_sms = transform_text(input_text)
        vector_input = tfidf.transform([transformed_sms]).toarray()

        result = model1.predict(vector_input)[0]

        vector_input = tfidf.transform([transformed_sms]).toarray()
        result_svc = model1.predict(vector_input)[0]
        accuracy_svc = 0.977756  
        precision_svc = 0.975207	
        f1_score_svc = 0.911197

        st.write("Support Vector Classifier Prediction:")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            st.write(f"Prediction: {'Spam' if result_svc == 1 else 'Not Spam'}")
        with col2:
            st.write(f"Accuracy: {accuracy_svc}")
        with col3:
            st.write(f"Precision: {precision_svc}")
        with col4:
            st.write(f"F1 Score: {f1_score_svc}")


model_options = ['Extra Trees','Naive Bayes']
selected_model = st.selectbox("Choose other model", model_options)

if st.button(f'Predict {selected_model}'):
    if translated_text:
        transformed_sms = transform_text(translated_text)
    else:
        transformed_sms = transform_text(input_text)
    
    vector_input = tfidf.transform([transformed_sms]).toarray()

    if selected_model == 'Extra Trees':
        result = model2.predict(vector_input)[0]
        accuracy = 0.974855 
        precision = 0.974576
        f1_score = 0.898438
    else:
        result = model3.predict(vector_input)[0]
        accuracy = 0.970019
        precision = 0.973451
        f1_score = 0.876494

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        st.write(f"Prediction: {'Spam' if result == 1 else 'Not Spam'}")
    with col2:
        st.write(f"Accuracy: {accuracy}")
    with col3:
        st.write(f"Precision: {precision}")
    with col4:
        st.write(f"F1 Score: {f1_score}")

