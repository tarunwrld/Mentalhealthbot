import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import load
import os
import time

def main():
    # Set up Streamlit page
    hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    page_bg_img = '''
      <style>
      [data-testid = "stAppViewContainer"] {
      background-image: url("https://e1.pxfuel.com/desktop-wallpaper/64/802/desktop-wallpaper-dark-night-mountains-minimalist-simple-backgrounds-black-oled.jpg");
      background-size: cover;
      }
      [data-testid = "stToolbar"] {
      right: 2rem;
      }
      </style>
      '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    # with st.container():
    st.title("Mental Illness Bot!!")

    st.text("This bot is trained on the Datasets available on Kaggle and Hugging-Face the bot cannot think it only can predict")
    
    st.caption("_The model is not optimized properly. Therefore, it cannot guarantee the accuracy of the generated answers._")

    st.divider()

    greet_text = """Hello!! How can I assist you?"""

    def greet():
        for word in greet_text.split():
            yield word + " "
            time.sleep(0.1)

    repo_id = "./mental5.joblib"
    clf = load(repo_id)

    def call(question):
        d = clf.predict([question])[0]
        return d
    
    def model(question):
        c = call(question)
        for word in c.split():
            yield word + " "
            time.sleep(0.1)
        

    question = st.chat_input("Write Something Here: ")
    if question:
        with st.status("In Progress..."):
            st.write("Loading Model")
            time.sleep(1)
            st.write("Searching for data...")
            time.sleep(1)
        if question.lower() in ["hi","hy", "yo", "hello", "how are you", "hola", "heya", "hey"]:
            with st.chat_message("user"):
                st.write_stream(('You' + question,))
            with st.chat_message("assistant"):
                st.write_stream(greet)
        else:
            with st.chat_message("user"):
                st.write_stream(('You' + question,))
            with st.chat_message("assistant"):
                st.write_stream(model(question))
            # c = clf.predict([question])[0]
            # st.write(c)
    else:
        # st.warning("Oops! Something went wrong. Please try again.")
        st.write("""Try asking What is Panic Attack , What is Stress , What is mental illness""")

if __name__ == "__main__":
    main()
