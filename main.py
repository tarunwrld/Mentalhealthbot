import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import load
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACETOKEN2")

headers = {"Authorization": f"Bearer {API_TOKEN}"}

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
      background-image: url("https://cdn.dribbble.com/userupload/12006881/file/original-27c0ab401cb6cc7abe6a4418264d08ee.jpg?crop=0x0-2000x1500&resize=400x300&vertical=center");
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

    repo_id = "chomu99/mentalhealthbot"

    question = st.text_input("Write Something Here: ")
    st.button("Ask")
    if question:
        if st.button:
            clf = load(repo_id)
            c = clf.predict([question])[0]
            # template = """Question: {question}
            #         Answer: Lets think step by step I'm a smart assistant My work is to provide efficient answer developed by Mr. Tarun"""

            # prompt = PromptTemplate(template=template, input_variables=["question"])
            # llm = HuggingFaceHub(
            #     repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 500}
            # )
            # llm_chain = LLMChain(prompt=prompt, llm=llm)

            # generated_text = llm_chain.run(question)
            with st.chat_message("user"):
                st.write(c)
        # else:
        #     st.warning("Oops! Something went wrong. Please try again.")
if __name__ == "__main__":
    main()

