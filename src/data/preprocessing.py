import pandas as pd
import numpy
import re
import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


def clean_instruction_text(text: str) -> str:
    """
    Cleans and preprocesses instructional text for natural language processing.
    
    This function performs several standard NLP cleaning steps:
    1. Validates the input.
    2. Strips HTML tags.
    3. Removes punctuation and converts text to lowercase.
    4. Tokenizes, removes stopwords, and lemmatizes the remaining words using spaCy.

    Args:
        text (str): The raw instruction text to be cleaned.

    Returns:
        str: A single string containing the cleaned, lemmatized tokens separated 
             by spaces. Returns the original input if it is not a valid, non-empty string.
             
    Dependencies:
        - The `re` module for regular expressions.
        - A global or in-scope spaCy language object named `nlp`.
    """

    # If text is NOT a string or if text is Empty/blank spaces
    if not isinstance(text, str) or not text.strip(): # if text is NOT  a string or if text is Empty/blank spaces
        return text

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove punctuation and convert to lowercase
    text = re.sub(r"[^\w\s]", "", text).lower()
    # Process text with spaCy
    doc = nlp(text)
    # Extract lemmas, ignoring stopwords and empty tokens
    tokens = [
        token.lemma_ for token in doc if not token.is_stop and token.lemma_.strip()

    ]
    # Join tokens back into a single string
    return " ".join(tokens)


def preprocessing_dataset(df):

    df["instruction"] = df["instruction"].apply(clean_instruction_text)

    return df


def preprocessing_input(input):

    return 