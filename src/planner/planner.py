
from src.agents.agent import Agent
from src.rag.embedding_model import EmbeddingModel
from src.pipelines.intent_training_pipeline import IntentClassifier
from src.pipelines.sentiment_training_pipeline import SentimentClassifier
from src.data.preprocessing import clean_instruction_text
import joblib
import numpy as np
from pathlib import Path

class Planner(Agent):
    name = "Planner agent"
    color = Agent.GREEN

    MODEL = "gpt-4o-mini"

    def __init__(self, sentiment_threshold=0.55):
        """
        Create instances of the Agents this planner coordinates
        """
        self.log("Planner agent is initializing")

        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent

        intent_path = project_root / "pretrain_models" / "intent_classifier_model.pkl"
        sentiment_path = project_root / "pretrain_models" / "sentiment_classifier_model.pkl"
       
        # Import pre-trained Classifiers
        self.embedding = EmbeddingModel()

        self.intent_model = joblib.load(intent_path)
                                    
        self.sentiment_model = joblib.load(sentiment_path)
                                
        self.threshold = sentiment_threshold
        
        self.log("Planner ready. Systems online")

    def route_message(self, input: str) -> dict:

        # 1 - Itent prediction phase

        #clean and embedding dor intent classifier
        clean_input = clean_instruction_text(input)
        emb_cleaninput = self.embedding.encode([clean_input])
        X_intent = np.array(emb_cleaninput, dtype=np.float32)

        #get itent prediction
        intent_pred = self.intent_model.predict(X_intent)[0] #The default result would be ["prediciton"], but since we want o populate a json we want the string so we stract the string to get "prediction"

        # 2 - Sentiment prediction phase

        emb_rawinput = self.embedding.encode([input])
        X_sentiment = np.array(emb_rawinput, dtype=np.float32)
        #get probability predictions
        sentiment_probs = self.sentiment_model.predict_proba(X_sentiment)[0]
        #get the index with max %
        max_sentiment_prob_index = np.argmax(sentiment_probs)

        sentiment_confidence = sentiment_probs[max_sentiment_prob_index]

        raw_sentiment_prediction = self.sentiment_model.classes_[max_sentiment_prob_index]


        if sentiment_confidence >= self.threshold :
            sentiment = raw_sentiment_prediction
            flag = "High Confidence"
        else:
            sentiment = "Neutral"
            flag = f"Low Confidende ({sentiment_confidence*100:.1f}%) - Default to Neutral"
       
       
        plan = {
            "user_message": input,
            "target_intent": intent_pred,
            "detected_sentiment": sentiment,
            "sentiment_debug_flag": flag
        }

        return plan

            

    def hello (self, message):

        return f"Hello there {message}"


if __name__ == "__main__":
    planner = Planner(sentiment_threshold=55)

    test_msg = "I've been waiting for my refund for over a month! Every time I call, I get put on hold for an hour."

    print("\nProcessing Incoming Message...")
    action_plan = planner.route_message(test_msg)

    print("\n--- PLANNER ROUTING DECISION ---")
    for key, value in action_plan.items():
        print(f"{key.upper()}: {value}")
