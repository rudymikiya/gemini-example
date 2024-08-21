import os

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

# Set proxy if it is needed...........
proxy = 'http://localhost:7897'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy


# read the api key from MyApiKey.txt
with open('MyApiKey.txt', 'r') as file:
    api_key = file.read().strip()
    print(f"API key: {api_key}")
    os.environ['GOOGLE_API_KEY'] = api_key

if __name__ == "__main__":
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
    all_splits = text_splitter.split_documents(data)
    vectorstore = FAISS.from_documents(documents=all_splits, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    question = "What are the approaches to Task Decomposition?"
    docs = vectorstore.similarity_search(question)

    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest")
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
    query_result = qa_chain({"query": question})
    print(query_result)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever()
    chat = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    result = chat({"question": "What are some of the main ideas in self-reflection?"})
    print(result['answer'])


