from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline


model_name = "facebook/bart-large-mnli"
revision = "c626438"
zero_shot_classifier=pipeline("zero-shot-classification",model=model_name, revision=revision)

HUGGING_FACE_TOKEN="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"
_OPENAI_API_KEY="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"

embeddings_model = HuggingFaceEmbeddings()
llm = OpenAI(openai_api_key=_OPENAI_API_KEY)

app = Flask(__name__)
CORS(app)
chroma = Chroma()

# alice_in_wonderland-1 
# Frankenstein -2
# the_importance_of_being_earnest-3
# Adventures_of_tom_sawyer-4
# anne_of_green_gables-5
# Dracula-6
# great_expectations-7
# little_women -8
# oliver_twist-9
# Sherlock_holmes-10 
# the_great_gatsby-11

# all means topic value -0

@app.route('/query', methods=['POST'])
def process_query():
    try:
        if request.method == 'POST':
            data = request.json['uInput']
            #topic = int(data.get('topic'))

            #if topic in ["None", None, 0, "0"]:

            #query = data.get('query')
            query=data
        
            print(query)
           
            answer = get_answer(query)
            # print("answer:--------",answer)
            
            return jsonify({'ANSWER': answer.strip()})
    except Exception as e:
        print("e",e)
        return jsonify({'ANSWER': ""})




    
def get_answer(query):
    new_line = '\n'
    template = f"Use the following pieces of context to answer truthfully.{new_line}If the context does not provide the truthful answer, make the answer as truthful as possible.{new_line}Use 15 words maximum. Keep the response as concise as possible.{new_line}{{context}}{new_line}Question: {{question}}{new_line}Response: "
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

    # Run chain
    from langchain.chains import RetrievalQA

    question = query

    vector_db = Chroma(persist_directory="./chroma_db",embedding_function=embeddings_model)

    qa_chain = RetrievalQA.from_chain_type(llm,
                                          retriever=vector_db.as_retriever(),
                                          return_source_documents=True,
                                          chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


    result = qa_chain({"query": question})
    return result["result"]

@app.route('/query/classify', methods=['POST'])
def process_query2():
    try:
        if request.method == 'POST':
            #data = request.get_json()
            query = request.json['query']
            
            general_chat=["General Chat", "Weather", "Pets", "Food", "Restaurants","Places","Trips", "Countries","Chat with Bot",
                    "Bot Interaction", "Ask Bot", "Engage Bot", "General Questions", "Trivia"]
            novel=["Adventures of Tom Sawyer","Alice's Adventures in Wonderland",
            "Anne of Green Gables",
            "Dracula",
            "Frankenstein",
            "Great Expectations",
            "Little Women",
            "Oliver Twist",
            "The Adventures of Sherlock Holmes",
            "The Great Gatsby",
            "The Importance of Being Earnest"]

    
            results=zero_shot_classifier(
            sequences = query,
            candidate_labels=[
            "Adventures of Tom Sawyer",
            "Alice's Adventures in Wonderland",
            "Anne of Green Gables",
            "Dracula",
            "Frankenstein",
            "Great Expectations",
            "Little Women",
            "Oliver Twist",
            "The Adventures of Sherlock Holmes",
            "The Great Gatsby",
            "The Importance of Being Earnest",

            #general categories 
            "General Chat", 
            "Weather",
            "Pets",
            "Food", 
            "Restaurants",
            "Places",
            "Trips", 
            "Countries",
            "Chat with Bot",
            "Bot Interaction", 
            "Ask Bot", 
            "Engage Bot", 
            "General Questions", 
            "Trivia"

            ],
            multiclass=True
            )

            
            labels = results["labels"]
            scores = results["scores"]

            max_score_indices = [i for i, score in enumerate(scores) if score == max(scores)]
            category=[]
            if len(max_score_indices) == 1:
   
                max_score_index = max_score_indices[0]
                highest_score_label = labels[max_score_index]
                # print("1.",highest_score_label)
                highest_score = scores[max_score_index]
                many_category=0
                category.append(highest_score_label)
                result_data={"max_score":highest_score, "categories":category, "many_flag":many_category}

            else:
                # Handle tie-breaking and give both categories in response 
                max_score_index = max_score_indices[0]
                highest_score_labels = [labels[i] for i in max_score_indices]
                highest_score_values = [scores[i] for i in max_score_indices]
                # print("2.",highest_score_labels)
                many_category=1
                category.extend(highest_score_labels)
                result_data={"max_score":highest_score_values[0], "categories":category,"many_flag":many_category}

                # print("data",data)

            chat_query =0 
            novel_query =0
            for cat in category:
                if cat in general_chat:
                    chat_query+=1
                elif cat in novel:
                    novel_query+=1


            if novel_query > chat_query:
                query_flag=2
            elif chat_query >= novel_query:
                query_flag=1

            result_data["query_category"]=query_flag
            # 1-general query
            # 2-novel based query 
            
            return jsonify({'ANSWER':result_data})
    except Exception as e:
        print("e",e)
        return jsonify({'ANSWER': ""})

if __name__ == '__main__':
    app.run(debug=True)
