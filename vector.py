from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os 
import pandas as pd 

'''IMPORTS'''

df = pd.read_csv("realistic_restaurant_reviews.csv")
''''LOAD DATA '''


#define the embedding file 
embeddging = OllamaEmbeddings(model="nomic-embed-text") 


#define location 
db_location = "./chroma_langchain_db"
#check location 
add_documents = not os.path.exists(db_location)
'''
# if os.path.exists(db_location):
#     add_documents = False   # DB already there, no need to add
# else:
#     add_documents = True    # First run, add the documents
'''

#create if not exists prepare all data 
if add_documents:
    documents = []
    ids=[]

    for i , row in df.iterrows():
        document = Document(
            page_content = row["Title"]+" "+row["Review"],
            metadata = {"rating":row["Rating"],"data":row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

#initialise the vector store
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory="db_location",
    embedding_function=embeddging
)            
# if dir doesn't exists into vector store 
if add_documents:
    vector_store.add_documents(documents=documents,ids=ids)

retriever = vector_store.as_retriever(
    search= {"k":6}
)