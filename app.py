#Import dependencies
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chain import LLMChain
from langchain_community.prompts import PromptTemplate
from langchain_community.shema.document import Document
from langchain.schema_messages import HummanMessage, SystemMessage
from langchain.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
from unstructured.partition.padf import partition_pdf
from airflow import DAG
from airflow.operators.python_operator import Pythonoperator
from datetime import datetime
from PIL import Image
import os
import nltk
import base64
import uuid



nltk.data.path.append('Multimodal_Rag_Finance/nltk_data')
os.environ['NLTK_DATA'] = 'Multimodal_Rag_Finance/nltk_data'
nltk.download('punkt')


openai_api_key = os.getenv('OPENAI_API_KEY')
#Path for input and output
pdf_path = 'E:/Journey 02/Personal Project/Multimodal_Rag_Finance/finacialplanning.pdf'
output_path = 'E:/Journey 02/Personal Project/Multimodal_Rag_Finance/Output'

def extract_and_summarize_pdf():
    if not os.path.exist(output_path):
        os.makedirs(output_path)

    
    # Load PDF and extract text
    raw_pdf_elements = partition_pdf(
        filename = pdf_path,
        extract_images_in_pdf = True,
        infer_table_structure = True,
        chunking_strategy = "by_title",
        max_characters = 4000,
        new_after_n_chars = 3800,
        combine_text_n_chars = 2000,
        extract_imabe_block_output_dir = output_path

    )

    text_summarize = []
    table_summarize = []
    text_elements = []
    table_elements = []

    summary_prompt = """
        Summarize the following {element_type}:
        {element}
        """
    
    summary_chain = LLMChain(
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key = openai_api_key, max_tokens = 1024),
        prompt = PromptTemplate.from_template(summary_prompt)
    )