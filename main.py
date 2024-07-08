from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.context_callback import ContextCallbackHandler
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


import os


load_dotenv()


# Load the environment variable
openai_api_key = os.getenv("OPENAIAPIKEY")
# Use the environment variable
if openai_api_key:
    print(f"OpenAI API Key: {openai_api_key}")
else:
    print("OPENAIAPIKEY environment variable not set.")


try:
    loader = PyPDFDirectoryLoader("pdfs")
    pages = loader.load()
    print(f"Total Docs: {len(pages)}")
except Exception as e:
    print(f"Error loading PDF: {e}")


content = pages
corpus = " ".join([page.page_content.replace("\t", " ") for page in content])
print(f"length of Corpus: {len(corpus)}, \n\n\ncorpus[:100]: {corpus[:100]}")


import re


def clean_corpus(text):
    """Nothing to clean yet"""
    return text


cleaned_corpus = clean_corpus(corpus)


openAIclient = ChatOpenAI(
    api_key=openai_api_key,
    # model_name = "gpt-3.5-turbo-16k", #default4k
    # temperature=0.1,
    callbacks=[ContextCallbackHandler(token="C2nN2SuVyaKE92pGT3HtcSsY")],
)

openai_api_key


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # gpt-3-5-turbo 179) , the docs quote a 16k context window
    chunk_overlap=10,
    length_function=len,
)


splitted_corpus = text_splitter.split_text(cleaned_corpus)


splitted_corpus = text_splitter.split_text(cleaned_corpus)
len(splitted_corpus), splitted_corpus[0]


splitted_corpus_in_docs = text_splitter.create_documents(splitted_corpus)
splitted_corpus_in_docs


openaiEmbeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
openaiEmbeddings


vectordb = FAISS.from_documents(splitted_corpus_in_docs, openaiEmbeddings)
retriever = vectordb.as_retriever()
retriever


### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    openAIclient, retriever, contextualize_q_prompt
)


### Answer question ###
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(openAIclient, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

print("=====================")
response = conversational_rag_chain.invoke(
    {
        "input": "List the categories covered by the paper titled 'TextGrad: Automatic Differentiation viaText'."
    },
    config={"configurable": {"session_id": "abc123"}},
)["answer"]


print(response)
