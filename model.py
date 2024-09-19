import streamlit as st
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv('spam.csv')


df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})


X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train).toarray()


classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)


db_path = r'C:\Users\ps713\OneDrive\Desktop\SpamEmailDetector\spam_detection.db'


try:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
except Exception as e:
    st.error(f'Error connecting to database: {e}')
    st.stop()


c.execute('''
    CREATE TABLE IF NOT EXISTS spam_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email_text TEXT,
        prediction TEXT
    )
''')
conn.commit()

st.title('Spam Email Detector')


email_input = st.text_area('Enter an email or message to classify:')

if st.button('Classify Email'):
    if email_input:
        
        email_vectorized = vectorizer.transform([email_input]).toarray()
        prediction = classifier.predict(email_vectorized)

        result = 'Spam' if prediction[0] == 1 else 'Not Spam'
        st.write(f'This email is classified as: **{result}**')

        try:
            c.execute("INSERT INTO spam_results (email_text, prediction) VALUES (?, ?)", (email_input, result))
            conn.commit()
            st.success('Result saved to the database!')
        except Exception as e:
            st.error(f'Error saving to database: {e}')


conn.close()
