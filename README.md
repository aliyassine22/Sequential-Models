# Sequential Models — RNN, GRU, LSTM (Mental-Health Text Classification)

A small, end-to-end project that compares **RNN**, **GRU**, and **LSTM** models on a mental-health text dataset, exposes a **FastAPI** endpoint for inference, and ships a minimal **React + Vite** UI to try predictions.



## What this project does
- Preprocesses raw text (de-contractions, cleanups, stopword removal, lemmatization).
- Trains **Word2Vec** embeddings and builds sequence tensors.
- Implements and evaluates **RNN**, **GRU**, and **LSTM** models for **4-class** classification:
  - *Normal, Depression, Suicidal, Other*.
- Serves a `/predict` API that validates inputs (Pydantic) and returns the predicted label.
- Provides a simple React frontend to submit text and visualize results.



## Why sequential models?
- **RNN:** baseline sequence learner; struggles with long dependencies (vanishing/exploding gradients).
- **GRU:** lighter than LSTM (update/reset gates); faster to train, good general performance.
- **LSTM:** adds an explicit cell state + gates to handle long-range context; often best on longer texts.



## Project structure
- Notebook/ training & experimentation (model notebook)
- Presentation/ slide deck summarizing concepts 
- api/ FastAPI app, routers, schemas (Pydantic), model I/O
- src/ react + Vite UI



## Tech stack
- **Modeling:** PyTorch (sequential models), Gensim (Word2Vec)  
- **API:** FastAPI + Pydantic (request/response schemas, validation)  
- **UI:** React + Vite



## Data & labels
- Clinical/mental-health text dataset mapped to four labels (*Normal, Depression, Suicidal, Other*).
- Tokenized, cleaned, lemmatized → embedded with **Word2Vec** → padded sequences for training.

## Maintanable Application Structure 
In our deployment, it is important to note that we tried adopting the best backend practices.
You may find a controller directory where our input's preprocessing and model deployment functions are found.
The routes directory contains the routes.py file where we are defining the HTTP endpoint and calling our controllers
The models directory contains our Pydantic/Swagger schemas.  
The  main.py file is the one responsible for running our backend.

## API (FastAPI)
- **Endpoint:** `POST /predict`
- **Input schema:** text field (validated with Pydantic).
- **Output schema:** predicted `class` (aliased field), plus optional metadata (e.g., probabilities).
- **Docs:** interactive Swagger UI at `/docs` (and ReDoc at `/redoc`) when the server is running.



## Frontend (React, by Chatgpt)
- Minimal form to input a text sample and display the model’s predicted class from the API.

- **cd ../**
- **npm install**
- **npm run dev**

## Quick start (indicative)
> Adjust commands to your environment; paths and script names may vary.

**Backend**

- **cd api**
- **create venv**
- **(install deps) pip install -r requirements.txt  # or install FastAPI, Uvicorn, Torch, Gensim**
- **after accessing the virtual environment, type: uvicorn main:app --reload**

Note that we did use Swagger UI to test our endpoints, it may be accessed on **http://localhost:8000/docs**. Note that the schema (only) was developed with the help of ChatGPT and our learning notes are found there.

## Results (high-level)

We compare accuracy across RNN, GRU, and LSTM on the same dataset/splits to highlight trade-offs:

- **RNN** (baseline)
- **GRU** (speed/efficiency)
- **LSTM** (long-dependency handling)
See notebooks and the slide deck for the exact scores and plots.


