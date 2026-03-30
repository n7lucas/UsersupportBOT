
from src.planner.planner import Planner
import joblib
import numpy as np
from src.data.preprocessing import clean_instruction_text
from src.rag.embedding_model import EmbeddingModel

def getIntentModel():
   loaded_model = joblib.load("./pretrain_models/intent_classifier_model.pkl")
   return loaded_model   

def getSentimentModel():
    loaded_model = joblib.load("./pretrain_models/sentiment_classifier_model.pkl")
    return loaded_model

def getItentPrediction(input):
  
    #Clean user input
    clean_input = clean_instruction_text(input)

    #Instantiate embedding model wich implements SentenceTransformer w/all-MiniLM-L6-v2
    embedder = EmbeddingModel()
    emb_input = embedder.encode(clean_input)
    
    model = getIntentModel()
    prediction = model.predict(emb_input.reshape(1,-1))
    print(prediction)


def getSentimentPrediction(input):

    #clean_input = clean_instruction_text(input)

    embbeded = EmbeddingModel()
    emb_input = embbeded.encode(input)

    model = getSentimentModel()

    prediction =  model.predict_proba(emb_input.reshape(1,-1))
    print(prediction)


def test():
    
    embedder = EmbeddingModel()
    test_data = [
        # 1. Frustrated
        "I've been waiting for my refund for over a month! Every time I call, I get put on hold for an hour and no one actually helps me. This is completely unacceptable.",
        
        # 2. Urgent
        "Please help! I just submitted my order but I accidentally used the wrong shipping address. It says it ships in 20 minutes, can someone please intercept this right now?!",
        
        # 3. Polite
        "Hello there, I hope you're having a wonderful day. Could you please let me know how I can go about returning an item that didn't quite fit? Thank you so much in advance!",
        
        # 4. Neutral
        "I need to update the billing credit card on file for my monthly subscription. Please provide the instructions to do this."
    ]

    test_embeddings = embedder.encode(test_data)
    X_test_custom = np.array(test_embeddings, dtype=np.float32)

    # Get the PROBABILITIES, not just the hard prediction

    model = getSentimentModel()

    probabilities = model.predict_proba(X_test_custom)
    classes = model.classes_ # Gets ['Frustrated', 'Neutral', 'Polite', 'Urgent']

    for i, text in enumerate(test_data):
        print(f"\nText: {text[:50]}...")
        
        # Print the probability for every single class
        for j, class_name in enumerate(classes):
            print(f"  {class_name}: {probabilities[i][j]*100:.2f}%")

def main():
    #mainPlanner = Planner()
    #print(mainPlanner.hello("Lucas"))
    #getItentPrediction("Can i get the invoice from my last buy please ?")
    #getSentimentPrediction("I need to update the billing credit card on file for my monthly subscription. Please provide the instructions to do this.")
    test()
if __name__ == "__main__":
    main()
