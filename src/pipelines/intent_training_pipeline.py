# download and preproccess
# create embedding column
# Training intent classifier using embedding cllumn to predict intent column

#Step 2 Sentiment step
# use LLM to create a new column sentiment based on 2000 user instructions column
    #Stratfy dataset based on intent first
# classify 2000 samples


from src.data import dataset_loader
from src.data import preprocessing
from src.rag.embedding_model import EmbeddingModel
from src.ml_models.intent_classifier import IntentClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from src.pipelines.sentiment_training_pipeline import run_sentiment_training_pipeline
import numpy as np
import joblib


def run_intent_training_pipeline():

    print("Loading dataset...")
    df = dataset_loader.load_customer_support_dataset()

    print("Preprocessing text...")
    df = preprocessing.preprocessing_dataset(df)

    # Merge the overlapping invoice intents
    df['intent'] = df['intent'].replace({'check_invoice': 'get_invoice'})

    print("Generating Embeddings...")
    embbeder = EmbeddingModel()

    embedding = embbeder.encode_dataframe(df)
    
    #Saving the Model
    print("Saving embedding vector as np arr...")
    embbeder.save_embeddings("instruction_embedding.npy", embedding )

    df["embeddings"] = list(embedding)

    X = np.array(df["embeddings"].to_list(), dtype=np.float32)
    y = np.array(df["intent"].to_list())

    print("Train test Split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = IntentClassifier()

    print("Training Model")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy*100:.2f}%\n")

    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'intent_classifier_model.pkl')
    print("Saved Intent Model -> intent_classifier_model.pkl")

if __name__ == "__main__":
    run_intent_training_pipeline()
    run_sentiment_training_pipeline()