import streamlit as st
from rag import process_urls, generate_answer

st.title("📝 Real Estate Research Tool")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

placeholder = st.empty()
sidebar_button = st.sidebar.button("Process URLs")
if sidebar_button:
    urls = [url for url in (url1, url2, url3) if url != '']
    if len(urls) == 0:
        placeholder.text("You must provide one valid URL")
    else:
        for status in process_urls(urls):
            placeholder.text(status)

st.subheader("Ask a Question")
query = st.text_input("Enter your query here.")

if st.button("Get Answer"):
    try:
        with st.spinner("Thinking..."):
            result = generate_answer((query))

        if result:
            st.write("### Answer")
            st.write(result.content.strip())

    except RuntimeError as e:
        placeholder.text("You must process URLs first")