
import requests
import json
import pandas as pd
import time
import sys
import subprocess

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/")
        if response.status_code == 200:
            print("✅ Ollama is running!")
            return True
    except requests.exceptions.ConnectionError:
        print("❌ Ollama is NOT running. Please start it manually via Start Menu.")
        return False
    return False

def pull_model():
    print(f"⬇️ Pulling {MODEL_NAME} model (this may take a while)...")
    # Using subprocess because the API for pull is strictly streaming and simpler via CLI
    try:
        subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
        print(f"✅ Model {MODEL_NAME} pulled successfully.")
    except Exception as e:
        print(f"❌ Failed to pull model: {e}")

def classify_tweet(tweet_text):
    prompt = f"""
    Classify the following tweet into one of these intents: 
    [casual, infrastructure_stress, migration_signal, rumor_mill, mobilization, escalation, satire_mockery, opinion]
    
    Tweet: "{tweet_text}"
    
    Return ONLY the intent label. Do not explain.
    Intent:
    """
    
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=data)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return "ERROR"
    except Exception as e:
        return f"ERROR: {e}"

def main():
    if not check_ollama_status():
        return
        
    pull_model()
    
    print("\n🧪 Testing Inference on Synthetic Data...")
    try:
        df = pd.read_csv("data/synthetic_kenya/tweets.csv")
    except FileNotFoundError:
        print("❌ usage: need data/synthetic_kenya/tweets.csv")
        return

    # Sample 5 random tweets
    sample = df.sample(5)
    
    results = []
    print(f"{'Tweet':<50} | {'True Intent':<20} | {'Llama 3 Pred':<20} | {'Match?'}")
    print("-" * 105)
    
    for _, row in sample.iterrows():
        text = row['text']
        true_intent = row['intent']
        pred = classify_tweet(text)
        match = "✅" if pred.lower() in true_intent.lower() or true_intent.lower() in pred.lower() else "❌"
        
        print(f"{text[:47]:<50}... | {true_intent:<20} | {pred:<20} | {match}")
        
    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    main()
