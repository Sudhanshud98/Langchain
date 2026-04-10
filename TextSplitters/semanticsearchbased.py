from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text = ''

splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type = 'standard deviation',
    breakpoint_threshold_amount = 1,
)

docs = text_splitter.create_documents([text])
print(len(docs))
print(docs)