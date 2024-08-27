import os

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Set proxy if it is needed...........
proxy = 'http://localhost:7897'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy
# read the api key from MyApiKey.txt
with open('./MyApiKey.txt', 'r') as file:
    api_key = file.read().strip()
    print(f"API key: {api_key}")
os.environ["GOOGLE_API_KEY"] = api_key


def split_text(path):
    current = 0
    lineNum = 200
    text = ""
    all_splits = []
    with open(path, 'r', encoding="utf-8") as f:
        line = f.readline()
        while line:
            text += line
            current = current + 1
            if lineNum == current:
                all_splits.append(text)
                text = ""
                current = 0
            line = f.readline()
    return all_splits


if __name__ == "__main__":

    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest")

    all_splits = split_text("./a.wav.srt")

    # chunks = split_text(text, chunk_size)
    # loader = TextLoader("a.wav.srt", encoding="utf-8")
    # data = loader.load()
    # text_splitter = TextSplitter(chunk_size=1000)
    # all_splits = text_splitter.split_documents(data)

    template = """You are a helpful translator. You can translate the Japanese to Chinese. 
    Just replace the Japanese with Chinese and don't change other language.
    Please don't add any seperator or metadata in response. """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # template = """Please only return the content of 'page_content'. Below are the contents to translate:
    #             {context}
    #
    #             """
    prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    retrieval_chain = (
            prompt
            | llm
    )

    # template = """You are a helpful translator. You can translate the Japanese to Chinese."""
    # system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    # human_template = "{text}"
    # human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    # chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    # chain = (chat_prompt | ChatGoogleGenerativeAI(
    #     model="models/gemini-1.5-flash-latest"))
    with open("b.srt", 'w') as file:
        for split in all_splits:
            res = retrieval_chain.invoke({"text": split})
            print(res.content, sep="", end="")
            file.write(res.content)
    # llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest")
    # qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
    # query_result = qa_chain({"query": question})
    # print(query_result)
    #
    # myfile = genai.upload_file("a.wav.srt")
    # print(f"{myfile=}")
    #
    # model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
    # response = model.generate_content([myfile,
    #                                    ],
    #                                   stream=True)

    # for chunk in response:
    #     print(chunk.text, sep="", end="")
    #     file.write(chunk.text)
