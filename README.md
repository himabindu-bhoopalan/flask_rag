# RAG (Retrieval Augmented Generation) Flask Application for Query Classification and Answer Retrieval

This project is a Flask-based web application that processes and classifies queries using Natural Language Processing (NLP) techniques. It supports text classification and retrieval of answers based on a predefined set of literary elements.

## Features

- **Query Classification**: Classifies user queries into predefined categories such as books or general topics.
- **Answer Retrieval**: Retrieves concise answers from a vector database using a language model.
- **Text Preprocessing**: Cleans and prepares user queries for better classification and retrieval.
- **Zero-Shot Classification**: Utilizes a zero-shot classification model to categorize queries without specific training data.

## Technologies Used

- **Flask**: A lightweight WSGI web application framework in Python.
- **LangChain**: For managing language model chains and embeddings.
- **Transformers**: For zero-shot classification using Hugging Face's `facebook/bart-large-mnli`.
- **NLTK**: For natural language processing tasks, such as removing stop words.
- **Chroma**: Vector database for storing and retrieving embeddings.
- **OpenAI**: For language modeling and question-answering tasks, used GPT4 APIs to paraphrase the text retrieved.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
