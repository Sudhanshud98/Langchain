from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template = "Prepare a report on the {topic}.",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template = "Prepare the 5 point summary of {report}",
    input_variables = ["report"]
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': "Future of AI"})
print(result)

