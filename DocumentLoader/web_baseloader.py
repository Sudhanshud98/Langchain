from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template = 'Answer the following question \n {question} from the following text \n {text}',
    input_variables = ['question', 'text']
)

parser = StrOutputParser()

url = "https://www.flipkart.com/apple-macbook-air-m2-8-gb-256-gb-ssd-mac-os-monterey-mly33hn-a/p/itmdc5308fa78421?pid=COMGFB2GMCRXZG85&lid=LSTCOMGFB2GMCRXZG855GPGWQ&marketplace=FLIPKART&store=6bo%2Fb5g&srno=b_1_3&otracker=browse&fm=organic&iid=af522b1d-b087-4518-8d55-0acf54b8c5a9.COMGFB2GMCRXZG85.SEARCH&ppt=None&ppn=None&ssid=313z9p8swg0000001775380970500&ov_redirect=true"
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser
print(chain.invoke({'question': 'What type of product is this?', 'text': docs[0].page_content}))