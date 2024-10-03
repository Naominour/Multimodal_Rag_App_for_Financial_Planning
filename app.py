#Import dependencies
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chain import LLMChain
from langchain_community.prompts import PromptTemplate
from langchain_community.shema.document import Document
from langchain.schema_messages import HummanMessage, SystemMessage
from langchain.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
from PIL import Image
import os
import nltk
import base64
import uuid

