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
import nltk
from nltk.corpus import stopwords
import re
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


adventures_of_sherlock_holmes_elements = {
    "characters": [
  "Sherlock Holmes",
    "Dr. John Watson",
    "Professor James Moriarty",
    "Irene Adler",
    "Inspector Lestrade",
    "Mycroft Holmes",
    "Mrs. Hudson (landlady of Baker Street)",
    "Mary Morstan",
    "Colonel Sebastian Moran",
    "Violet Hunter",
    "Mrs. Irene Norton (née Adler)",
    "Sir Henry Baskerville",
    "Dr. James Mortimer",
    "Stapleton",
    "Mrs. Barrymore",
    "John Hector McFarlane",
    "Jonathan Small",
    "Jabez Wilson",
    "Wilson's assistant, Vincent Spaulding",
    "Inspector Gregson",
    "Inspector Baynes",
    "Inspector Bradstreet",
    "Mr. Melas",
    "The King of Bohemia (Wilhelm Gottsreich Sigismond von Ormstein)",
    "The Hound of the Baskervilles"
    ],
    "places": [
        "221B Baker Street (Sherlock Holmes's residence)",
        "The Diogenes Club",
        "The Strand (newspaper office)",
        "St. Monica's (boarding house in 'The Adventure of the Crooked Man')",
        "The Red-Headed League office",
        "Briony Lodge (in 'A Scandal in Bohemia')",
        "The Count's estate (in 'The Adventure of the Six Napoleons')"
    ],
    "specific_elements": [
        "Sherlock's Pipe",
        "Watson's Revolver",
        "The Stradivarius Violin",
        "The Persian Slipper",
        "The 7% solution (Cocaine)",
        "The Code in the Dancing Men",
        "The Sign of Four (The missing treasure)",
        "The Speckled Band (Snake)",
        "The Engineer's Thumb",
        "The Beryl Coronet (Precious jewelry)",
        "The Copper Beeches (Tree)",

    ],
    "chapter_titles": [
        "A Scandal in Bohemia",
        "The Red-Headed League",
        "A Case of Identity",
        "The Boscombe Valley Mystery",
        "The Five Orange Pips",
        "The Man with the Twisted Lip",
        "The Adventure of the Blue Carbuncle",
        "The Adventure of the Speckled Band",
        "The Adventure of the Engineer's Thumb",
        "The Adventure of the Noble Bachelor",
        "The Adventure of the Beryl Coronet",
        "The Adventure of the Copper Beeches"

    ]
}

anne_of_green_gables_elements = {
    "characters": [
        "Anne Shirley",
        "Marilla Cuthbert",
        "Matthew Cuthbert",
        "Diana Barry",
        "Gilbert Blythe",
        "Mrs. Rachel Lynde",
        "Josie Pye",
        "Ruby Gillis",
        "Miss Stacy",
        "Charlie Sloane",
        "Mr. Phillips",
        "Mr. and Mrs. Barry",
        "Jerry Baynard",
        "Prissy Andrews",
        "Moody Spurgeon",
        "Aunt Josephine Barry",
        "Mr. Bell",
    ],
    "places": [
        "Green Gables",
        "Avonlea",
        "White Way of Delight",
        "Lake of Shining Waters",
        "Carmody",
        "Bolingbroke",
        "Bright River",
        "The Haunted Wood",
        "The Dryad's Bubble",
        "The Sloping Main",
    ],
    "specific_elements": [
        "Anne's Red Hair",
        "Puffed Sleeve Dress",
        "Cuthberts' Farm",
        "Anne's Imagination",
        "Avenue of White Birches",
        "Anne's Bosom Friend",
        "The Lady of Shalott",
        "The Haunted Wood",
        "Diana's Amethyst Brooch",
        "Lake of Shining Waters"
    ],
    "chapter_titles": [
        "Chapter 1: Mrs. Rachel Lynde is Surprised",
        "Chapter 2: Matthew Cuthbert is Surprised",
        "Chapter 3: Marilla Cuthbert is Surprised",
        "Chapter 4: Morning at Green Gables",
        "Chapter 5: Anne's History",
        "Chapter 6: Marilla Makes Up Her Mind",
        "Chapter 7: Anne Says Her Prayers",
        "Chapter 8: Anne's Bringing-Up is Begun",
        "Chapter 9: Mrs. Rachel Lynde is Properly Horrified",
        "Chapter 10: Anne's Apology",
        "Chapter 11: Anne's Impressions of Sunday School",
        "Chapter 12: A Solemn Vow and Promise"
    ]
}

dracula_elements = {
    "characters": [
        "Count Dracula",
        "Jonathan Harker",
        "Mina Murray",
        "Lucy Westenra",
        "Abraham Van Helsing",
        "Arthur Holmwood",
        "Quincey Morris",
        "Dr. John Seward",
        "Renfield",
        "Mrs. Westenra",
        "Mr. Hawkins",
        "Mr. Swales",
    ],
    "places": [
        "Transylvania",
        "Castle Dracula",
        "Whitby",
        "Carfax",
        "Seward's Asylum",
        "Piccadilly",
        "Exeter",
        "Bistritz",
    ],
    "specific_elements": [
        "Vampire",
        "Renfield's obsession with consuming life",
        "Jonathan's journal",
        "Blood transfusions",
        "The Demeter",
        "Crucifix",
        "St. Mary's Churchyard",
        "Garlic",
        "The three female vampires"
    ],
    "chapter_titles": [
        "Jonathan Harker's Journal",
        "The Whitby Adventure",
        "Mina's Journal",
        "The Terror of Blue Eyes",
        "The Blood Transfusion",
        "The Demeter",
        "The Piccadilly Horror",
        "Dr. Seward's Diary",
        "Mina Harker's Journal"
    ]
}

great_gatsby_elements = {
    "characters": [
        "Jay Gatsby",
        "Nick Carraway",
        "Daisy Buchanan",
        "Tom Buchanan",
        "Jordan Baker",
        "Myrtle Wilson",
        "George Wilson",
        "Meyer Wolfsheim",
        "Owl Eyes",
        "Dan Cody",
        "Klipspringer"
    ],
    "places": [
        "West Egg",
        "East Egg",
        "The Valley of Ashes",
        "New York City",
        "Gatsby's Mansion",
        "Tom and Daisy's Mansion",
        "Wilson's Garage",
        "The Plaza Hotel",
        "The Buchanan Estate"
    ],
    "specific_elements": [
        "The Green Light",
        "The Eyes of Dr. T.J. Eckleburg",
        "Gatsby's Parties",
        "The Valley of Ashes Symbolism",
        "The Gold Headed Cane",
        "The Yellow Car",
        "Gatsby's Library",
        "The Tattered Copy of 'Hopalong Cassidy'",
        "The Clock Stopping at the Accident",
        "The Unused Pool"
    ],
    "chapter_titles": [
        "Chapter 1: Nick Carraway's Introduction",
        "Chapter 2: Tom Buchanan's Affair",
        "Chapter 3: Gatsby's Extravagant Parties",
        "Chapter 4: Gatsby's Background",
        "Chapter 5: The Reunion of Gatsby and Daisy",
        "Chapter 6: Gatsby's Past Revealed",
        "Chapter 7: The Plaza Hotel Incident",
        "Chapter 8: Tragedy Strikes",
        "Chapter 9: Gatsby's Funeral"
    ]
}

importance_of_being_earnest_elements = {
    "characters": [
        "Jack Worthing",
        "Algernon Moncrieff",
        "Gwendolen Fairfax",
        "Cecily Cardew",
        "Lady Bracknell",
        "Miss Prism",
        "Dr. Chasuble",
        "Lane",
        "Merriman"
    ],
    "places": [
        "Jack's Country Estate",
        "Algernon's Flat",
        "The Garden"
    ],
    "specific_elements": [
        "The Importance of Being Earnest (The name)",
        "The Handbag",
        "The Diary",
        "The Cigarette Case",
        "The Engagement Ring",
        "The Train Schedule",
        "The Sermon on Marriage",
        "The Miss Prism Case",
        "The Bunburying Concept",
        "The Pianola"
    ],
    "act_titles": [
        "Act I: Algernon's Flat in Half Moon Street",
        "Act II: The Garden at the Manor House",
        "Act III: Morning-room at the Manor House"

    ]
}
#6 th book 
# "Adventures of Tom Sawyer"

tom_sawyer_elements = {
    "characters": [
        "Tom Sawyer",
        "Huckleberry Finn",
        "Aunt Polly",
        "Becky Thatcher",
        "Injun Joe",
        "Muff Potter",
        "Sid Sawyer",
        "Joe Harper",
        "Judge Thatcher",
        "The Widow Douglas"
    ],
    "places": [
        "St. Petersburg",
        "Mississippi River",
        "Jackson's Island",
        "McDougal's Cave"
    ],
    "specific_elements": [
        "The Whitewashed Fence",
        "The Treasure Map",
        "The Catfish Trade",
        "The Schoolhouse Event",
        "Tom's Relationship with Becky",
        "The Mysterious Murder"

    ],
    "chapter_titles": [
        "Chapter 1: Aunt Polly Decides Upon her Duty",
        "Chapter 2: Tom Plays, Fights, and Hides",
        "Chapter 3: Sunday School and the Buzzing Afterwards",
        "Chapter 4: Huck and Tom's Superstitions",
    ]
}
# #7 th book 
# "Alice's Adventures in Wonderland"

alice_in_wonderland_elements = {
    "characters": [
        "Alice",
        "The White Rabbit",
        "The Mad Hatter",
        "The Queen of Hearts",
        "The Cheshire Cat",
        "The Caterpillar",
        "The March Hare",
        "The Mock Turtle",
        "The King of Hearts",
        "The Gryphon"
    ],
    "places": [
        "Wonderland",
        "The Rabbit Hole",
        "The Queen's Croquet Ground",
        "The Mad Tea-Party"
    ],
    "specific_elements": [
        "Drink Me Bottle",
        "Eat Me Cake",
        "The Caterpillar's Mushroom",
        "The Cheshire Cat's Smile",
        "The Queen's Tarts",
        "The Caucus Race"
    ],
    "chapter_titles": [
        "Chapter 1: Down the Rabbit-Hole",
        "Chapter 2: The Pool of Tears",
        "Chapter 3: A Caucus-Race and a Long Tale",
        "Chapter 4: The Rabbit Sends in a Little Bill"
    ]
}
# # 8 th book 
# "Frankenstein"
frankenstein_elements = {
    "characters": [
        "Victor Frankenstein",
        "The Creature (Frankenstein's Monster)",
        "Elizabeth Lavenza",
        "Henry Clerval",
        "Robert Walton",
        "Alphonse Frankenstein",
        "Justine Moritz",
        "William Frankenstein",
        "Professor Waldman",
        "The De Lacey Family"

    ],
    "places": [

        "The Swiss Alps",
        "The Orkney Islands"
    ],
    "specific_elements": [
        "The Monster's Creation",
        "The Monster's Education",
        "The Monster's Quest for Companionship",
        "The Monster's Revenge",
        "Victor's Guilt and Remorse",
        "The Iceberg Incident"
    
    ],
    "chapter_titles": [
        "Letter 1: To Mrs. Saville, England",
        "Letter 2: To Mrs. Saville, England",
        "Chapter 1: Victor Frankenstein's Background",
        "Chapter 2: Victor's Education",
        "Chapter 3: Victor's Experiments and the Creation of the Monster",
        "Chapter 4: The Monster's First Experiences",
        "Chapter 5: The Monster's Education",
        "Chapter 6: The Monster's Quest for Companionship"
    
    ]
}

# # 9th book 
#  "Great Expectations"
great_expectations_elements = {
    "characters": [
        "Pip (Philip Pirrip)",
        "Miss Havisham",
        "Estella",
        "Abel Magwitch",
        "Joe Gargery",
        "Mrs. Joe",
        "Herbert Pocket",
        "Bentley Drummle",
        "Jaggers",
        "Wemmick"
       
    ],
    "places": [
        "Kent",
        "Satis House",
        "The Forge",
        "The Three Jolly Bargemen",
        
    ],
    "specific_elements": [
        "Pip's Great Expectations",
        "Miss Havisham's Manipulation",
        "Estella's Coldness",
        "Abel Magwitch's Secret",
        "The Mysterious Benefactor",
        "The Reveal of True Identities"
        
    ],
    "chapter_titles": [
        "Chapter 1: The Convict and the Boy",
        "Chapter 2: Miss Havisham and Estella",
        "Chapter 3: Pip's Education",
        "Chapter 4: The Mysterious Benefactor",
        "Chapter 5: The Reveal of True Identities"
    ]
}

# # 10 book 
little_women_elements = {
    "characters": [
        "Jo March",
        "Meg March",
        "Beth March",
        "Amy March",
        "Marmee",
        "Laurie Laurence",
        "Mr. March",
        "Aunt March",
        "Friedrich Bhaer",
        "Mr. Laurence"
       
    ],
    "places": [
        "March Family Home",
        "The Laurence Estate",
    ],
    "specific_elements": [
        "The March Sisters' Bond",
        "Jo's Writing Aspirations",
        "Beth's Piano",
        "Amy's Artistic Pursuits",
        "Meg's Marriage",
        "The Loss and Grief",
        "Friedrich Bhaer's Influence"
       
    ],
    "chapter_titles": [
        "Chapter 1: Playing Pilgrims",
        "Chapter 2: A Merry Christmas",
        "Chapter 3: The Laurence Boy",
        "Chapter 4: Burdens",
        "Chapter 5: Being Neighborly"
       
    ]
}

oliver_twist_elements = {
    
                         "characters":[
    "Oliver Twist",
    "Mr. Bumble (the beadle)",
    "Mr. Brownlow",
    "Fagin (the Jew)",
    "Nancy",
    "Bill Sikes",
    "The Artful Dodger (Jack Dawkins)",
    "Charley Bates",
    "Mr. Grimwig",
    "Monks (Edward Leeford)",
    "Rose Maylie",
    "Mrs. Maylie",
    "Mr. Losberne",
    "Toby Crackit",
    "Noah Claypole",
    "Bet (Nancy's friend)",
    "Mr. Sowerberry",
    "Mrs. Sowerberry",
    "Charlotte (maid in the Sowerberry household)",
    "Mr. Gamfield (a chimney sweep)",
    "The Old Woman (who takes Oliver in)",
    "Mrs. Corney (the matron of the workhouse)",
    "Mr. Fang (the magistrate)",
    "Mr. Giles (servant of Mrs. Maylie)",
    "Mr. Brittles (servant of Mrs. Maylie)",
    "Mr. and Mrs. Mann (overseers of the workhouse)",
    "The Bow Street Runners (police officers)",
    "The inmates of Fagin's den (including Noah Claypole, Charlotte, Barney, etc.)",
    "The prisoners at Newgate Prison",
    ]
                         ,
    "places": [
        "The Workhouse",
        "Fagin's Den",
        "Saffron Hill",
        "The Three Cripples"
    ],
    "specific_elements": [
        "Oliver's Orphan Status",
        "The Pickpocketing Gang",
        "Fagin's Influence on the Boys",
        "Nancy's Loyalty Conflict",
        "Bill Sikes' Ruthlessness",
        "Monks' Secret",
        "The Mystery of Oliver's Birth",
        "Rose Maylie's Role"
        
    ],
    "chapter_titles": [
        "Chapter 1: Treats of the Place Where Oliver Twist Was Born",
        "Chapter 2: Treats of Oliver Twist's Growth, Education, and Board",
        "Chapter 3: Relates How Oliver Twist Was Very Near Getting a Place, Which Would Not Have Been a Sinecure",
        "Chapter 4: Oliver, Being Offered Another Place, Makes His First Entry Into Public Life",
        "Chapter 5: Oliver Mingles with New Associates. Going to a Funeral for the First Time, He Forms an Unfavourable Notion of His Master’s Business"
       
    ]
}

def preprocess_query(query):
    query_cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
    query_lower = query_cleaned.lower().split()
    query_words_without_stopwords = [word for word in query_lower if word not in stop_words]
    return query_words_without_stopwords

def classify_query(query):
    query_lower = preprocess_query(query)

    query_words_without_stopwords = [word for word in query_lower if word not in stop_words]
    # print(query_words_without_stopwords)
    
    book_title_to_elements = {
        "Adventures of Tom Sawyer": tom_sawyer_elements,
        "Alice's Adventures in Wonderland": alice_in_wonderland_elements,
        "Anne of Green Gables": anne_of_green_gables_elements,
        "Dracula": dracula_elements,
        "Frankenstein": frankenstein_elements,
        "Great Expectations": great_expectations_elements,
        "Little Women": little_women_elements,
        "Oliver Twist": oliver_twist_elements,
        "The Adventures of Sherlock Holmes": adventures_of_sherlock_holmes_elements ,
        "The Great Gatsby": great_gatsby_elements,
        "The Importance of Being Earnest": importance_of_being_earnest_elements,
    }

    for book_title, elements in book_title_to_elements.items():
        for key, values in elements.items():
            for value in values:
                if any(word in value.lower() for word in query_words_without_stopwords):
                    return book_title, 2  # Returning book title and numeric value

    return "General Chat",1

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
            result_data["categories"] ,result_data["query_category"]=classify_query(query)
            return jsonify({'ANSWER':result_data})
    except Exception as e:
        print("e",e)
        return jsonify({'ANSWER': ""})

if __name__ == '__main__':
    app.run(debug=True)