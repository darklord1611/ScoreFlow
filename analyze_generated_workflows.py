#!/usr/bin/env python3

"""
Analyzes a workflow graph dataset from a pickle file.

This script performs the following steps:
1.  Loads the dataset (e.g., dataset-GSM8K-0.pkl).
2.  Performs an exact-match analysis to count specific "bad" templates.
3.  Performs a similarity-based analysis to find:
    - Malformed graphs (parse errors).
    - "Unchanged" graphs (too similar to a default template).
4.  Analyzes and provides metrics for:
    - Distinct questions that *ever* caused an error.
    - Distinct questions that *always* parsed successfully.
5.  Takes *both* query types, embeds them, and runs PCA and t-SNE.
6.  Saves a cluster visualization that clearly distinguishes
    error-inducing queries from successful queries.
"""

import pickle
import re
import datetime
from difflib import SequenceMatcher
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

# --- 1. Constants and Configuration ---

# Path to your pickle file
PKL_PATH = "/teamspace/studios/this_studio/modal_remote/ScoreFlow/scoreflow_workspace/output_workflow/dataset-GSM8K-0.pkl"

# Default/unwanted graph templates for analysis
TEMP_AVOID = """class Workflow:
    def __init__(
        self,
        config,
        problem
    ) -> None:
        self.problem = problem
        self.config = create(config)
        self.custom = operator.Custom(self.config, self.problem)
        self.sc_ensemble = operator.ScEnsemble(self.config, self.problem)
        self.programmer = operator.Programmer(self.config, self.problem)
        self.review = operator.Review(self.config, self.problem)

    async def run_workflow(self):
        \"\"\"
        This is a workflow graph.
        \"\"\"
        solution = await self.custom(instruction="Can you solve this problem by breaking it down into detailed steps and explaining the reasoning behind each step?")

        return solution"""

TEMP_AVOID_V2 = """class Workflow:
    def __init__(
        self,
        config,
        problem
    ) -> None:
        self.problem = problem
        self.config = create(config)
        self.custom = operator.Custom(self.config, self.problem)
        self.sc_ensemble = operator.ScEnsemble(self.config, self.problem)
        self.programmer = operator.Programmer(self.config, self.problem)
        self.review = operator.Review(self.config, self.problem)

    async def run_workflow(self):
        solution = await self.custom(instruction="Can you solve this problem by breaking it down into detailed steps and explaining the reasoning behind each step?")

        return solution"""

TEMP_GRAPH = "<graph>\nclass Workflow:\n    async def run_workflow(self):\n        return ''\n</graph>"

# Similarity threshold for flagging "unchanged" graphs
SIM_THRESHOLD = 0.8

# Dimensionality reduction configuration
PCA_COMPONENTS = 74


# --- 2. Helper Function ---

def similarity_ratio(str1, str2):
    """Calculates the SequenceMatcher similarity ratio between two strings."""
    return SequenceMatcher(None, str1, str2).ratio()

def extract_workflow_body(class_script: str) -> str:
    """Uses regex to extract the body of the run_workflow method."""
    # re.DOTALL makes '.' match newline characters
    match = re.search(
        r"async def run_workflow\(self\)(.*?)return", class_script, re.DOTALL
    )
    if not match:
        raise ValueError("Could not find 'async def run_workflow' body.")
    return match.group(1).strip()


# --- 3. Main Analysis Script ---

def main():
    """Runs the full analysis pipeline."""
    
    # --- Section 1: Load Data (from Cell 1) ---
    print("--- 1. Loading Data ---")
    try:
        with open(PKL_PATH, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: Pickle file not found at {PKL_PATH}")
        return
    except Exception as e:
        print(f"❌ ERROR: Failed to load pickle file. {e}")
        return

    print(f"✅ Loaded pickle file with {len(data)} entries.")
    if data:
        print("🔍 Example entry [\"graph\"]:")
        print(data[0]["graph"])
    else:
        print("⚠️ Data is empty. Exiting.")
        return

    print("-" * 40)

    # --- Section 2: Exact Match Analysis (from Cell 3) ---
    print("--- 2. Analyzing Exact Graph Matches ---")
    all_graphs = [entry["graph"] for entry in data]
    unique_graphs = set(all_graphs)
    
    print(f"Total Entries: {len(all_graphs)}")
    print(f"Unique Graph Structures: {len(unique_graphs)}")

    # Count exact matches
    unwanted_counts = {
        "TEMP_GRAPH": 0,
        "TEMP_AVOID": 0,
        "TEMP_AVOID_V2": 0
    }
    for graph in all_graphs:
        if graph == TEMP_GRAPH:
            unwanted_counts["TEMP_GRAPH"] += 1
        elif graph == TEMP_AVOID:
            unwanted_counts["TEMP_AVOID"] += 1
        elif graph == TEMP_AVOID_V2:
            unwanted_counts["TEMP_AVOID_V2"] += 1
            
    print("\nCounts for specific unwanted graphs (exact match):")
    print(f"  TEMP_GRAPH:    {unwanted_counts['TEMP_GRAPH']}")
    print(f"  TEMP_AVOID:    {unwanted_counts['TEMP_AVOID']}")
    print(f"  TEMP_AVOID_V2: {unwanted_counts['TEMP_AVOID_V2']}")
    
    print(f"\n(Original notebook output: {list(unwanted_counts.values())}, {len(unique_graphs)})")
    print("-" * 40)

    # --- Section 3: Similarity & Error Analysis (from Cell 4) ---
    print("--- 3. Analyzing Similarity and Parse Errors ---")
    
    error_graphs_questions = []
    unchanged_graphs_questions = []
    # This list will hold questions for graphs that parsed
    # AND were not "unchanged" (i.e., truly "good" graphs)
    successful_graphs_questions = []
    ok_graphs = []
    
    # Extract the "bad" template's body once, outside the loop
    try:
        template_body = extract_workflow_body(TEMP_AVOID)
        print("Successfully extracted body from TEMP_AVOID template for comparison.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not parse TEMP_AVOID. {e}")
        return

    for i, entry in enumerate(data):
        try:
            # 1. Extract content inside <graph> tags
            graph_content_match = re.search(r"<graph>(.*?)</graph>", entry["graph"], re.DOTALL)
            if not graph_content_match:
                raise ValueError("Could not find <graph> tags.")
            
            class_script = graph_content_match.group(1).strip()
            
            # 2. Extract body of run_workflow
            workflow_body = extract_workflow_body(class_script)
            
            # 3. Compare similarity
            similar_score = similarity_ratio(workflow_body, template_body)
            
            if similar_score >= SIM_THRESHOLD:
                unchanged_graphs_questions.append(entry["question"])
            else:
                # Parsed successfully and is not default
                successful_graphs_questions.append(entry["question"])
                ok_graphs.append(entry["graph"])
                
        except Exception as e:
            # Any regex failure or error lands here
            error_graphs_questions.append(entry["question"])

    print("\nAnalysis Complete:")
    print(f"  Total workflows processed: {len(data)}")
    print(f"  workflows with parse errors (malformed): {len(error_graphs_questions)}")
    print(f"  workflows too similar to default (>= {SIM_THRESHOLD*100}%): {len(unchanged_graphs_questions)}")
    print(f"  workflows parsed successfully: {len(successful_graphs_questions)}")

    print(f" Distinct successfully parsed workflows generated: {len(set(ok_graphs))}")
    
    print(f"\n(Original notebook output for reference: {len(data)}, {len(error_graphs_questions)}, {len(unchanged_graphs_questions)})")
    print("-" * 40)

    # --- Section 4: Unique Question Analysis (from Cell 5) ---
    print("--- 4. Analyzing Unique Problematic Questions ---")
    
    # Get all unique questions from the dataset
    set_all_queries = set(entry["question"] for entry in data)
    
    # Get unique questions that *ever* caused an error
    set_error_queries = set(error_graphs_questions)

    set_unchanged_queries = set(unchanged_graphs_questions)
    
    set_successful_queries = set(successful_graphs_questions)

    # Convert to lists for processing
    distinct_error_queries = list(set_error_queries)
    distinct_unchanged_queries = list(set_unchanged_queries)
    distinct_successful_queries = list(set_successful_queries)
    
    if len(set_all_queries) == 0:
        print("No queries found in dataset. Exiting.")
        return
        
    percent_error = (len(distinct_error_queries) / len(set_all_queries)) * 100
    percent_success = (len(distinct_successful_queries) / len(set_all_queries)) * 100
    percent_unchanged = (len(distinct_unchanged_queries) / len(set_all_queries)) * 100

    # Sanity check
    total_accounted = len(distinct_error_queries) + len(distinct_unchanged_queries) + len(distinct_successful_queries)
    print(f"  (Total queries categorized: {total_accounted})/{len(set_all_queries)}")

    print("--- Query Type Metrics ---")
    print(f"Total distinct questions in dataset: {len(set_all_queries)}")
    print(f"  - Distinct questions that *always* parsed: {len(distinct_successful_queries)} ({percent_success:.2f}%)")
    print(f"  - Distinct questions that *ever* caused an error: {len(distinct_error_queries)} ({percent_error:.2f}%)")
    print(f"  - Distinct questions that create workflows similar to default: {len(distinct_unchanged_queries)} ({percent_unchanged:.2f}%)")
    
    print("-" * 40)

    # --- Section 5: Embedding & Clustering (from Cells 6 & 7) ---
    print("--- 5. Visualizing Query Types (Successful vs. Error) ---")
    
    if not distinct_error_queries and not distinct_successful_queries:
        print("⚠️ No queries found to cluster. Skipping.")
        print("-" * 40)
        return

    # 1. Combine lists and create labels
    # 0 = Successful, 1 = Error
    # <-- MODIFIED SECTION: Combine all three lists -->
    # 0 = Successful, 1 = Unchanged, 2 = Error
    all_queries_to_plot = (
        distinct_successful_queries + 
        distinct_unchanged_queries + 
        distinct_error_queries
    )
    labels = (
        [0] * len(distinct_successful_queries) + 
        [1] * len(distinct_unchanged_queries) + 
        [2] * len(distinct_error_queries)
    )

    # 2. Load model
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Calculate embeddings
    print(f"Calculating embeddings for {len(all_queries_to_plot)} total queries...")
    embeddings = model.encode(all_queries_to_plot)
    print(f"Embeddings shape: {embeddings.shape}")

    # 4. PCA
    # Adjust PCA components if we have fewer samples than components
    n_components = min(PCA_COMPONENTS, len(all_queries_to_plot))
    if n_components < 2:
        print("⚠️ Not enough data points to run PCA/t-SNE. Exiting.")
        return
        
    print(f"Performing PCA (reducing to {n_components} components)...")
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    print(f"PCA-reduced embeddings shape: {embeddings_pca.shape}")

    # 5. t-SNE
    print("Performing t-SNE for visualization (reducing to 2 components)...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, len(embeddings_pca) - 1),  # Perplexity must be < n_samples
        n_iter=500,
        init='pca',
        learning_rate='auto'
    )
    embeddings_tsne = tsne.fit_transform(embeddings_pca)
    print("t-SNE complete.")

    # 6. Visualization
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = PKL_PATH.replace(PKL_PATH.split("/")[-1], "") + f'embeddings_tsne_clustering_{timestamp}.png'
    print(f"Generating and saving plot to '{output_filename}'...")

    # <-- MODIFIED SECTION: Separate all three results for plotting -->
    labels_array = np.array(labels)
    successful_tsne = embeddings_tsne[labels_array == 0]
    unchanged_tsne = embeddings_tsne[labels_array == 1]
    error_tsne = embeddings_tsne[labels_array == 2]
    
    plt.figure(figsize=(12, 9))
    
    # Plot successful queries (Blue)
    if len(successful_tsne) > 0:
        plt.scatter(
            successful_tsne[:, 0],
            successful_tsne[:, 1],
            c='blue',
            label=f'Successful Queries (n={len(successful_tsne)})',
            s=50,
            alpha=0.6
        )
    
    # Plot unchanged queries (Orange)
    if len(unchanged_tsne) > 0:
        plt.scatter(
            unchanged_tsne[:, 0],
            unchanged_tsne[:, 1],
            c='orange',
            label=f'Unchanged Queries (n={len(unchanged_tsne)})',
            s=50,
            alpha=0.6
        )
    
    # Plot error queries (Red)
    if len(error_tsne) > 0:
        plt.scatter(
            error_tsne[:, 0],
            error_tsne[:, 1],
            c='red',
            label=f'Error Queries (n={len(error_tsne)})',
            s=50,
            alpha=0.6
        )
    # <-- END MODIFIED SECTION -->
    
    plt.title('t-SNE Visualization of Query Types')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    try:
        plt.savefig(output_filename)
        plt.close()
        print(f"✅ Plot saved successfully.")
    except Exception as e:
        print(f"❌ ERROR: Could not save plot. {e}")
    
    print("-" * 40)
    print("🎉 Analysis finished.")


if __name__ == "__main__":
    main()