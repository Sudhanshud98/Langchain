from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = ''

splitter = RecursiveCharacterTextSplitter.from_Language(
    language = Language.PYTHON,
    chunk_size = 100,
    chunk_overlap = 0,
)

chunk = splitter.split_text(text)

print(len(chunk))
print(chunk)