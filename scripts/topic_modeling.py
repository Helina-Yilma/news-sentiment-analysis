import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
import nltk
import os

# --- Configuration ---
CLEANED_DATA_PATH = 'data/df_cleaned.pkl'
NMF_RESULT_PATH = 'data/df_with_topics.pkl'
SAMPLE_SIZE = 50000 
N_TOPICS = 10 
RANDOM_STATE = 42
NO_TOP_WORDS = 8
# ---------------------

def display_topics(model, feature_names, no_top_words):
    """Prints the top keywords for each NMF topic."""
    topic_keywords = {}
    print("\n--- Discovered Topics (Top Keywords) ---")
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-no_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_keywords[topic_idx] = top_words
        print(f"Topic {topic_idx}: {', '.join(top_words)}")
    return topic_keywords

def run_topic_modeling():
    """
    Main function to run the NMF topic modeling pipeline.
    """
    print(f"Loading cleaned data from {CLEANED_DATA_PATH}...")
    try:
        df = pd.read_pickle(CLEANED_DATA_PATH)
    except FileNotFoundError:
        print("Error: Cleaned data not found. Run 01_data_cleaning.py first.")
        return

    # Download standard NLTK resources
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass
        
    print(f"Total data size for topic assignment: {df.shape[0]}")

    # 1. Data Sampling for Training (Efficiency)
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    docs_sample = df_sample['headline'].apply(str).tolist()
    print(f"NMF model will be trained on a sample of {len(docs_sample)} headlines.")

    # 2. Vectorization (DTM Creation)
    vectorizer = CountVectorizer(
        stop_words='english',
        max_df=0.95, 
        min_df=5,    
        max_features=10000 # Critical for memory control
    )
    dtm_sample = vectorizer.fit_transform(docs_sample)
    feature_names = vectorizer.get_feature_names_out()

    # 3. NMF Model Training
    print(f"Starting NMF Topic Model Training with {N_TOPICS} topics...")
    nmf_model = NMF(
        n_components=N_TOPICS,
        random_state=RANDOM_STATE,
        max_iter=200, 
        init='nndsvda',
        solver='mu',
        beta_loss='frobenius'
    )
    nmf_model.fit(dtm_sample)
    print("NMF Model Training Complete.")
    
    # Print the topics for interpretation
    display_topics(nmf_model, feature_names, NO_TOP_WORDS)

    # 4. Assign Topics to the FULL Dataset
    print("\nStarting Topic Assignment for FULL DataFrame...")
    full_docs = df['headline'].apply(str).tolist()
    full_dtm = vectorizer.transform(full_docs)
    
    # Get the Topic-Document matrix
    full_doc_topic_matrix = nmf_model.transform(full_dtm)
    
    # Assign the dominant topic ID back to the original DataFrame
    df['nmf_topic_id'] = full_doc_topic_matrix.argmax(axis=1)
    
    # 5. Save the result
    df.to_pickle(NMF_RESULT_PATH)
    print(f"DataFrame with 'nmf_topic_id' saved to {NMF_RESULT_PATH}.")

if __name__ == '__main__':
    run_topic_modeling()