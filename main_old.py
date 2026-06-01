from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter , RecursiveCharacterTextSplitter
load_dotenv()
# Text Loader
data_1 = TextLoader("documents_loaders/notes.txt")

docs_1 = data_1.load()
template_1 = ChatPromptTemplate.from_messages([
    ("system", "You are a AI summarizes assistant."),
    ("human", "{docs_1}")
])
model_1 = ChatMistralAI(model="mistral-small-2506")
prompt_1 = template_1.format_messages(docs_1= docs_1[0].page_content)
result_1 = model_1.invoke(prompt_1)
#print(result_1.content)

#PDF Loader
data = PyPDFLoader("documents_loaders/GRU.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks  = splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a AI summarizes assistant."),
    ("human", "{docs}")
])
model = ChatMistralAI(model="mistral-small-2506")
prompt = template.format_messages(docs= docs)
result = model.invoke(prompt)
print(result.content)