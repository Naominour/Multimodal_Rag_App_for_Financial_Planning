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
from transformers import BlipProcessor, BlipForConditionalGeneration
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

    for e in raw_pdf_elements:
        if 'CompositeElement' in repr(e):
            text_elements.append(e.text)
            summary = summary_chain.run({'element_type': 'text', 'element': e})
            text_summarize.append(summary)
        
        elif 'Table' in repr(e):
            table_elements.append(e)
            summary = summary_chain.rin({'element_type' : 'table', 'element': e})
            table_summarize.append(summary)

    return table_summarize, text_summarize, table_elements, text_elements


def simmarize_image():

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    image_elements = []
    image_summaries = []

    for i in os.listdir(output_path):
        if i.endwith('.png', '.jpg', '.jpeg'):
            image_path = os.path.join(output_path, i)
            image_elements.append(image_path)
            raw_image = Image.open(image_path).convert('RGB')

            inputs = processor(raw_image, return_tensor='pt')
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens = True)

            prompt = f"You are an expert in analyzing images and charts related to Financial Planning. 
                       Based on the following description, provide a detailed analysis:\n\nDescription: {caption}"
            
            response = ChatOpenAI(model='gpt-4', openai_api_key=openai_api_key, max_tokens=1024)
            image_summaries.append(response.content)

    return image_elements, image_summaries

