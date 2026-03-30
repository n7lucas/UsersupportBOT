

from src.data.dataset_loader import load_customer_support_dataset
from src.agents.labelling_agent import get_sentiment_label
from src.rag.embedding_model import EmbeddingModel
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from src.ml_models.sentiment_classifier import SentimentClassifier

from sklearn.metrics import accuracy_score, classification_report
import joblib
def  run_sentiment_training_pipeline():

    SAMPLES_PER_INTENT = 80
    df = load_customer_support_dataset()

    #This will reduce our training data into 80 samples per intent inr order to minimize the cost and computation for the LLM
    df_sample = df.groupby("intent").apply(lambda x: x.sample(SAMPLES_PER_INTENT, random_state=42)).reset_index(drop=True)


    print(f"Create a sample of {len(df_sample)} rows for LLM labeling")

    #Create the embedding vectors for the 2000 instructions
    embbeder = EmbeddingModel()

    embeddings = embbeder.encode_dataframe(df_sample) #encode_dataferame is set to encode the instruction column

    #Saving the embedding
    print("Saving embedding vector as numpy arr...")
    EmbeddingModel.save_embeddings("sampled_embedding.npy", embeddings)
    
    df_sample["embeddings"] = list(embeddings )


    print("Starting LLM labeling proccess...")
    sentiment_labels = []

    for i in tqdm(df_sample["instruction"]):
        sentiment_labels.append(get_sentiment_label(i))

    df_sample["sentiment"] = sentiment_labels

    print("Saving the gold dataset...")
    df_sample.to_csv('llm_labeled_sentiment_data.csv', index=False)

    print("\nSuccess! Dataset saved as llm_labeled_sentiment_data.csv")

    #Converting attributes and classes to numpy arr
    X = np.array(embeddings, dtype=np.float32)

    y = np.array(df_sample["sentiment"].to_list())

    #Spliting data into training/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = SentimentClassifier(class_weight='balanced')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy score: {accuracy*100:.2f}%\n")

    print(classification_report(y_test, y_pred))
  
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'sentiment_classifier_model.pkl')
    print("Saved Sentiment Model -> sentiment_classifier_model.pkl")

