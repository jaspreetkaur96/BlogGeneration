import streamlit as st
from langchain.prompts import PromptTemplate
# from langchain.llms import CTransformers
from langchain_community.llms import CTransformers
from datetime import datetime


#no need of oepnai key, as model in local
### Function to get repsonse from LLama2 modle
def getLlamaResponse(input_text, no_words, blog_style):
    ###llama2 model
    llm = CTransformers(model = "models/llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama",
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})
    
    ### prompt template
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words.
    """

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"], template=template)
    print(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print("start time: ", datetime.now())
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)
    print("end time: ", datetime.now())

    return response




st.set_page_config(page_title="Generate Blogs",
                    page_icon="📝",
                    layout="centered",
                    initial_sidebar_state="collapsed")
        
st.header("Generate Blogs 📝")
input_text = st.text_input("Enter the blog Topic")

#creating two more columns for additional two fields
col1, col2 = st.columns([5,5])

with col1:
    no_words = st.text_input("No. of words")

with col2:
    blog_style = st.selectbox("Writing th Blog for",
                            ("Researchers", "Data Scientists", "Common people"), index=0)
    
submit_button = st.button("Generate")

### Final response
if submit_button:
    st.write(getLlamaResponse(input_text, no_words, blog_style))