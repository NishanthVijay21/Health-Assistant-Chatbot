from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# Load and preprocess data
df = pd.read_csv("Diseases_Symptoms.csv")

def preprocess_symptoms(symptoms):
    symptoms = symptoms.lower().strip()
    symptoms = re.sub(r"[^\w\s,]", "", symptoms)  # Remove punctuation
    return symptoms

df["Symptoms"] = df["Symptoms"].apply(lambda x: preprocess_symptoms(x))
df["Symptoms"] = df["Symptoms"].apply(lambda x: x.split(", "))  # Convert to list
df["Symptoms_Joined"] = df["Symptoms"].apply(lambda x: " ".join(x))

# Calculate the number of symptoms for each disease for later comparison
df["Symptom_Count"] = df["Symptoms"].apply(len)

# Initialize vectorizer and model
vectorizer = CountVectorizer(binary=True, 
                           token_pattern=r"\b\w+\b",
                           stop_words="english", 
                           ngram_range=(1, 2))
symptom_matrix = vectorizer.fit_transform(df["Symptoms_Joined"])

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(symptom_matrix, df["Name"])

def extract_symptoms(user_input):
    """Extract symptoms from user input"""
    user_input = preprocess_symptoms(user_input)
    tokens = user_input.split()
    bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    
    extracted_symptoms = []
    for bigram in bigrams:
        if bigram in vectorizer.get_feature_names_out():
            extracted_symptoms.append(bigram)
            words_in_bigram = bigram.split()
            tokens = [t for t in tokens if t not in words_in_bigram]
    
    for token in tokens:
        if token in vectorizer.get_feature_names_out():
            extracted_symptoms.append(token)
    
    return extracted_symptoms

def check_disease_confidence(user_symptoms, disease_name, threshold=0.7):
    """Check if we have enough symptoms to confidently predict a disease"""
    # Get all symptoms for this disease
    disease_symptoms = set(df[df["Name"] == disease_name]["Symptoms"].values[0])
    
    # Convert user symptoms to a set for comparison
    # First normalize by replacing spaces with underscores
    user_symptoms_set = set(symptom.replace(" ", "_") for symptom in user_symptoms)
    
    # Calculate how many disease symptoms the user has confirmed
    matching_symptoms = sum(1 for symptom in disease_symptoms if symptom.replace(" ", "_") in user_symptoms_set)
    
    # Calculate percentage of disease symptoms confirmed
    if len(disease_symptoms) > 0:
        confidence = matching_symptoms / len(disease_symptoms)
        return confidence >= threshold
    
    return False

def generate_follow_ups(top_diseases, user_symptoms):
    # Normalize user symptoms (replace spaces with underscores)
    normalized_user_symptoms = set(symptom.replace(" ", "_") for symptom in user_symptoms)
    
    # First check if we have enough confidence in the top disease
    top_disease = top_diseases[0]
    if check_disease_confidence(user_symptoms, top_disease):
        # If we're confident, return empty list to stop follow-ups
        return []
    
    # Otherwise, continue with follow-up questions
    follow_up_symptoms = set()
    
    # Focus on symptoms from the most likely disease first
    disease_symptoms = df[df["Name"] == top_disease]["Symptoms"].values[0]
    for symptom in disease_symptoms:
        symptom_underscore = symptom.replace(" ", "_")
        if symptom_underscore not in normalized_user_symptoms:
            follow_up_symptoms.add(symptom)
    
    # If we don't have enough from the top disease, add some from other candidates
    if len(follow_up_symptoms) < 3:
        for disease in top_diseases[1:3]:
            disease_symptoms = df[df["Name"] == disease]["Symptoms"].values[0]
            for symptom in disease_symptoms:
                symptom_underscore = symptom.replace(" ", "_")
                if symptom_underscore not in normalized_user_symptoms:
                    follow_up_symptoms.add(symptom)
                    if len(follow_up_symptoms) >= 5:  # Limit to 5 follow-ups max
                        break
            if len(follow_up_symptoms) >= 5:
                break
    
    # Sort follow-ups by importance (for now, just use the top disease symptoms first)
    sorted_follow_ups = []
    
    # First add symptoms from top disease
    top_disease_symptoms = df[df["Name"] == top_disease]["Symptoms"].values[0]
    for symptom in top_disease_symptoms:
        if symptom in follow_up_symptoms:
            sorted_follow_ups.append(symptom)
            follow_up_symptoms.remove(symptom)
    
    # Then add remaining follow-ups
    sorted_follow_ups.extend(list(follow_up_symptoms))
    
    # Limit to at most 5 follow-up questions
    return sorted_follow_ups[:5]

# Serve frontend files
@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_input = data.get('symptoms', '')
        current_symptoms = data.get('current_symptoms', '')  # For follow-ups
        
        # EXACT extraction logic
        extracted = extract_symptoms(user_input)
        all_symptoms = current_symptoms.split() + extracted if current_symptoms else extracted
        symptom_text = " ".join(all_symptoms)
        
        # Vectorize EXACTLY like backend
        user_vector = vectorizer.transform([symptom_text])
        
        # Get probabilities EXACTLY like backend
        probabilities = model.predict_proba(user_vector)[0]
        top_indices = np.argsort(probabilities)[::-1][:5]  # Top 5 predictions
        
        # Generate follow-ups using same logic
        top_diseases = model.classes_[top_indices]
        follow_ups = generate_follow_ups(top_diseases, all_symptoms)
        
        # Calculate confidence for the top disease
        is_confident = check_disease_confidence(all_symptoms, top_diseases[0])
        
        # Add the top disease probability to the response
        top_probability = probabilities[top_indices[0]]
        
        # Get the treatment for the top predicted disease by finding its index in the dataframe
        top_disease_name = top_diseases[0]
        top_disease_index = df[df["Name"] == top_disease_name].index[0]  # Get the actual index in df
        treatment = df.iloc[top_disease_index]["Treatments"] if not pd.isna(df.iloc[top_disease_index]["Treatments"]) else "No treatment info"
        
        return jsonify({
            "diseases": top_diseases.tolist(),
            "probabilities": probabilities[top_indices].tolist(),
            "matched_symptoms": extracted,
            "current_symptoms": symptom_text,  # For follow-ups
            "follow_up_questions": [s.replace("_", " ") for s in follow_ups],
            "treatment": treatment,
            "confidence_reached": is_confident,
            "top_probability": float(top_probability)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)