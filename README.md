Project Title: Text-to-SQL Query System with Multi-Source Data
Table of Contents
Introduction
Model Used
Contact
Introduction
This project is a Text-to-SQL Query System with Multi-Source Data. It converts natural language queries into SQL queries that can retrieve data from both MySQL databases and CSV data sources. The system leverages machine learning models (specifically, the T5-small model) to facilitate intuitive query generation.

I explored several text generative models, including Olama, BERT, T5-base, Seq2Seq, and T5-small. Due to my laptop's specifications, I had to choose a model that was compatible without consuming too much time or GPU resources. I ultimately selected T5-small.

However, I encountered an issue where the model generated correct SQL queries but was unable to relate them properly to the database. To address this, I fine-tuned the model on the WikiSQL dataset. Initially, I achieved 92% accuracy after the first training session. Unfortunately, I could not save the model, necessitating a second round of training. Due to limitations on Google Colab, the training process was interrupted midway, taking around five hours to train for just half the epochs. Each time, the session was disconnected, which forced me to save the partially trained model, but it still resulted in errors.

While I have the ML model ready, its performance is limited by the available GPU resources. I then explored other pre-trained models on Hugging Face, but their accuracy matched that of my half-trained model, making them unhelpful. I also discovered the pandas-nql library, which is promising but requires a ChatGPT key. I believe that with access to a proper GPU, I can complete the remaining work to achieve better accuracy.

Model Used
For the backend, I utilized Flask to create a REST API, connecting it to my SQL database where my files are stored. I used Postman to test the functionality of my API. My approach effectively searches through the database, and the 92% accuracy machine learning model is included in this repository under the name fine_fine_tuned_model.ipynb, along with the Flask app in app.py.
