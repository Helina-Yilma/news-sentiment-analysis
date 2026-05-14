import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
DATA_PATH = 'data/df_with_topics.pkl'
TOP_N_PUBLISHERS = 5
TOPICS_TO_PLOT = 5 # Focus on the first 5 topics for the plot clarity
# ---------------------

def run_publisher_specialization():
    """
    Main function to analyze and visualize publisher topic specialization.
    """
    print(f"Loading topic-assigned data from {DATA_PATH}...")
    try:
        df = pd.read_pickle(DATA_PATH)
    except FileNotFoundError:
        print("Error: Topic-assigned data not found. Run 02_topic_modeling.py first.")
        return

    os.makedirs('plots', exist_ok=True)

    # 1. Pivot the table to show Topic Counts per Publisher Domain
    print("Calculating topic distribution by publisher...")
    topic_distribution_by_publisher = df.groupby('publisher_domain')['nmf_topic_id'].value_counts().unstack(fill_value=0)

    # 2. Add a Total Count column for sorting
    topic_distribution_by_publisher['Total_Count'] = topic_distribution_by_publisher.sum(axis=1)

    # 3. Sort and select the top N publishers
    top_publishers_by_topic = topic_distribution_by_publisher.nlargest(TOP_N_PUBLISHERS, columns='Total_Count')
    
    # 4. Select the topics to plot (e.g., Topics 0-4)
    topics_to_include = [i for i in range(TOPICS_TO_PLOT) if i in top_publishers_by_topic.columns]
    plot_df = top_publishers_by_topic[topics_to_include]
    
    # --- 5. Generate the Stacked Bar Chart ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    
    # Plotting the stacked bars directly on the current axes
    plot_df.plot(kind='bar', stacked=True, colormap='Spectral', ax=plt.gca())

    plt.title(f'Topic Specialization of Top {TOP_N_PUBLISHERS} News Publishers', fontsize=16)
    plt.xlabel('News Publisher (Source)', fontsize=12)
    plt.ylabel(f'Total Article Count (Topics {min(topics_to_include)}-{max(topics_to_include)})', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='NMF Topic ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

  

if __name__ == '__main__':
    run_publisher_specialization()