from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

def load_customer_support_dataset(split: str = "train") -> pd.DataFrame:
    """ Download the dataset and convert into Pandas"""
    dataset = load_dataset(DATASET_NAME, split=split)
    df = dataset.to_pandas()
    
    return df