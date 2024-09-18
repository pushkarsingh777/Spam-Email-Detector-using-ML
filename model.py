import streamlit as st
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load data
df = pd.read_csv('spam.csv')

# Preprocess the data
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train).toarray()

# Train a Naive Bayes model
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Define the new path to your database file
db_path = r'C:\Users\ps713\OneDrive\Desktop\SpamEmailDetector\spam_detection.db'

# Try connecting to SQLite database
try:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
except Exception as e:
    st.error(f'Error connecting to database: {e}')
    st.stop()

# Create a table if it doesn't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS spam_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email_text TEXT,
        prediction TEXT
    )
''')
conn.commit()

# Streamlit interface
st.title('Spam Email Detector')

# Input from user
email_input = st.text_area('Enter an email or message to classify:')

if st.button('Classify Email'):
    if email_input:
        # Vectorize and predict
        email_vectorized = vectorizer.transform([email_input]).toarray()
        prediction = classifier.predict(email_vectorized)

        # Display the prediction result
        result = 'Spam' if prediction[0] == 1 else 'Not Spam'
        st.write(f'This email is classified as: **{result}**')

        # Store the result in the SQLite database
        try:
            c.execute("INSERT INTO spam_results (email_text, prediction) VALUES (?, ?)", (email_input, result))
            conn.commit()
            st.success('Result saved to the database!')
        except Exception as e:
            st.error(f'Error saving to database: {e}')

# Option to view previous classifications from the database
if st.checkbox('Show previous classifications'):
    try:
        df_results = pd.read_sql('SELECT * FROM spam_results', conn)
        st.write(df_results)
    except Exception as e:
        st.error(f'Error fetching data from database: {e}')

# Close the database connection
conn.close()
