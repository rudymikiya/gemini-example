from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI


class EmailSummarizer(object):

    def __init__(self, model_name="models/gemini-1.5-flash-latest"):
        self.model = ChatGoogleGenerativeAI(
            model=model_name)

    def summarize(self, important_criteria: str, processed_email_path: str, summary_output_file: str):
        with open(processed_email_path, 'r') as file:
            # Load the JSON data from the file
            json_string = file.read()
        template = """Below is a list of email summaries and format is a json. Please order the summaries in descending order of importance. The result should have summary and fileName:
                {context}
    
                Treat it as important: {important_criteria}
                You should response like a secretary and the content is not a json. The start should be like 'These are important the unread emails recently:' and use arabic numerals for list. Put fileName in a new line and separated with 2 new lines.
                """
        prompt = ChatPromptTemplate.from_template(template)

        retrieval_chain = (
                {"context": lambda x: json_string,
                 "important_criteria": RunnablePassthrough()}
                | prompt
                | self.model
        )

        res = retrieval_chain.invoke(important_criteria)
        print(res.content)
        self.__write_summary(res.content, summary_output_file)

    # write the email summaries to a json file
    def __write_summary(self, summary: str, output_file: str):
        with open(output_file, "w") as _file:
            _file.write(summary)
