from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model = 'gpt-4', temperature = 0.3)

## 1a Indexing the data

video_id = 'Gfr50f6ZBvo'

try:
    api = YouTubeTranscriptApi()

    transcript_list = api.fetch(video_id, languages=['en'])

    transcripts = ' '.join(chunk.text for chunk in transcript_list)
    # print(transcripts)

except TranscriptsDisabled:
    print("No captions available for this video.")

#1b Splitting the data

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
chunks = splitter.create_documents([transcripts])

# print(len(chunks))

#1c and 1d Creating the vector store and embedding the data

embedding = OpenAIEmbeddings(model = 'text-embedding-3-small')
vector_store = FAISS.from_documents(chunks, embedding)

# print(vector_store.index_to_docstore_id)

## 2 Retrieval

retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {'k': 4})
# print(retriever.invoke('What is deep mind?'))

## 3 Augmentation

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables = ['context', 'question']
)

question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed about it?"
retrived_docs = retriever.invoke(question)

# print(retrived_docs)

context_text = "\n\n".join(doc.page_content for doc in retrived_docs)

final_prompt = prompt.invoke({'context': context_text, 'question': question})
# print(final_prompt)

#4 Generation

answer = model.invoke(final_prompt)
# print(answer)

def formated_docs(retrived_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrived_docs)
    return context_text

parallel_chain = RunnableParallel({
    'context' : retriever | RunnableLambda(formated_docs),
    'question' : RunnablePassthrough()
})

parallel_chain.invoke('Who is Demis?')

parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser
print(main_chain.invoke('Can you summerize the video?'))
