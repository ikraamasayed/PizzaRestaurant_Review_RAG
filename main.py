from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="gemma3:4b")
template = """  
You are an expert in answering questions and you know all the reviews s

here are some relevent reviews :{review}

here are some questions to answer : {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = prompt|model


# result = chain.invoke({
#     "review":[],
#     "question":"what is the best pizza in mumbai,India"})

#Provides Continuity for constant question & answer
while True:
    question = input("Ask Your Question (Q to Quit)")
    if question.lower() == "q":
        break
    reviews = retriever.invoke(question)
    result = chain.invoke({"review":reviews,"question":question})
    print(result)