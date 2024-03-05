import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import load
import os

def main():
    # Set up Streamlit page
    hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            # header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    page_bg_img = '''
      <style>
      [data-testid = "stAppViewContainer"] {
      background-image: url("https://cdn.dribbble.com/users/675405/screenshots/16991196/media/99c347f1e0f663a5cc51815d2efcd543.png?resize=800x600&vertical=center");
      background-size: cover;
      }
      [data-testid = "stToolbar"] {
      right: 2rem;
      }
      </style>
      '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.title("Mental Illness Bot!!")

    st.subheader("This bot is trained on the dataset available on kaggle and hugging face the bot cannot think it can only predict")
    
    st.write("The answer generated is not guaranteed to be right")

    # repo_id = "./mental2.joblib"

    question = st.text_input("Write Something Here: ")
    st.button("Ask")
    if question:
        if st.button:
            clf = load("./mental4.joblib")
            c = clf.predict([question])[0]
            with st.chat_message("user"):
                st.write(c)
        # else:
        #     st.warning("Oops! Something went wrong. Please try again.")
if __name__ == "__main__":
    main()

