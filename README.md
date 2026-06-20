# Health Assistant Chatbot 🩺🤖

An intelligent, interactive Health Assistant Chatbot that uses Natural Language Processing (NLP) and Machine Learning to predict potential diseases based on user-reported symptoms. 

The application features a responsive chat interface, dynamic follow-up questioning to improve prediction confidence, and an integrated Text-to-Speech (TTS) engine with visual voice animations for enhanced accessibility.

## ✨ Features

* **Symptom Extraction via NLP:** Uses `CountVectorizer` with n-gram support to intelligently extract symptoms from natural language input.
* **Machine Learning Predictions:** Employs a `RandomForestClassifier` trained on a comprehensive symptom-disease dataset to calculate the probabilities of potential conditions.
* **Dynamic Follow-up Questions:** If the initial input lacks sufficient symptoms for a confident prediction, the backend dynamically generates targeted "yes/no" follow-up questions to narrow down the possibilities.
* **Treatment Recommendations:** Provides standard treatment information for the highest-probability disease.
* **Text-to-Speech (TTS):** Includes a customizable Web Speech API integration that reads bot responses aloud, paired with a dynamic CSS "speaking ball" animation.
* **Responsive UI:** A clean, mobile-friendly chat interface built with pure HTML/CSS/JS.

## 🛠️ Tech Stack

### Backend (Machine Learning & API)
* **Python:** Core programming language.
* **Flask & Flask-CORS:** Lightweight WSGI web application framework to serve the API and frontend.
* **Scikit-Learn:** Used for feature extraction (`CountVectorizer`) and predictive modeling (`RandomForestClassifier`).
* **Pandas & NumPy:** For data manipulation and probability calculations.

### Frontend (User Interface)
* **HTML5 / CSS3:** Styling with custom animations for the TTS UI.
* **Vanilla JavaScript:** Handles DOM manipulation, state management (follow-up queues), API requests, and Web Speech API integration.

## 📁 Project Structure

```text
├── server.py                 # Main Flask application and ML logic
├── Diseases_Symptoms.csv     # Dataset containing diseases, symptoms, and treatments
└── index.html       # Frontend UI (served by Flask)
