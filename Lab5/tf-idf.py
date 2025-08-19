import numpy as np
from tabulate import tabulate
import math

def calculate_tf_idf_weights(tf_table, idf_values):
    """Calculates TF-IDF weights for each term and document."""
    tf_idf_weights = {term: {} for term in tf_table}
    for term, docs in tf_table.items():
        for doc, tf in docs.items():
            if term in idf_values:
                tf_idf_weights[term][doc] = tf * idf_values[term]
    return tf_idf_weights

def calculate_query_score(query, weights, documents):
    """Calculates the score for a given query against each document."""
    scores = {doc: 0 for doc in documents}
    for doc in scores:
        for term in query:
            if term in weights and doc in weights[term]:
                scores[doc] += weights[term][doc]
    return scores

def euclidean_normalize_tf(tf_table, documents, terms):
    """Performs Euclidean normalization on the TF values."""
    tf_vectors = {
        doc: np.array([tf_table[term][doc] for term in terms]) for doc in documents
    }
    
    euclidean_norms = {doc: np.linalg.norm(vec) for doc, vec in tf_vectors.items()}
    
    normalized_tf = {term: {} for term in terms}
    for i, term in enumerate(terms):
        for doc in documents:
            if euclidean_norms[doc] != 0:
                normalized_tf[term][doc] = tf_vectors[doc][i] / euclidean_norms[doc]
            else:
                normalized_tf[term][doc] = 0.0
            
    return normalized_tf

def print_tables(data_set, tf_idf_weights, scores1, scores2, normalized_tf):
    """Formats and prints all calculated data in a tabular format."""
    
    documents = list(list(tf_idf_weights.values())[0].keys())
    terms = list(tf_idf_weights.keys())

    print(f"### TF-IDF Weights for {data_set}\n")
    headers_weights = ["Term"] + documents
    table_data_weights = []
    for term, weights in tf_idf_weights.items():
        row = [term] + [f"{weights.get(doc, 0.0):.2f}" for doc in documents]
        table_data_weights.append(row)
    print(tabulate(table_data_weights, headers=headers_weights, tablefmt="github"))
    print("\n")

    print(f"### Query Scores for {data_set}\n")
    print(f"#### Query 1: '{scores1['query_string']}'\n")
    headers_scores = ["Document", "Score"]
    table_data_scores1 = [[doc, f"{scores1['scores'][doc]:.2f}"] for doc in documents]
    print(tabulate(table_data_scores1, headers=headers_scores, tablefmt="github"))
    print("\n")
    
    print(f"#### Query 2: '{scores2['query_string']}'\n")
    table_data_scores2 = [[doc, f"{scores2['scores'][doc]:.2f}"] for doc in documents]
    print(tabulate(table_data_scores2, headers=headers_scores, tablefmt="github"))
    print("\n")
    
    print(f"### Euclidean Normalized TF for {data_set}\n")
    headers_norm = ["Term"] + documents
    table_data_norm = []
    for term in terms:
        row = [term] + [f"{normalized_tf.get(term, {}).get(doc, 0.0):.2f}" for doc in documents]
        table_data_norm.append(row)
    print(tabulate(table_data_norm, headers=headers_norm, tablefmt="github"))
    print("\n")

def get_user_input_and_process():
    """Prompts user for input and processes the data."""
    print("--- User Input Section ---\n")
    print("Please provide your own documents and data.\n")

    try:
        doc_names_input = input("Enter document names, separated by commas (e.g., DocA, DocB, DocC): ")
        documents = [doc.strip() for doc in doc_names_input.split(',')]
        
        doc_content = {}
        all_words = []
        for doc in documents:
            content = input(f"Enter the full text for '{doc}': ")
            words = content.lower().split()
            doc_content[doc] = words
            all_words.extend(words)

        # --- AUTOMATICALLY GENERATE TERMS ---
        terms = sorted(list(set(all_words)))
        print(f"\nAutomatically generated terms from documents: {', '.join(terms)}\n")
        
        tf_table_user = {term: {} for term in terms}
        for term in terms:
            for doc in documents:
                tf_table_user[term][doc] = doc_content[doc].count(term)
        
        idf_values_user = {}
        total_docs = len(documents)
        for term in terms:
            document_frequency = sum(1 for doc_words in doc_content.values() if term in doc_words)
            if document_frequency > 0:
                idf_values_user[term] = math.log10(total_docs / document_frequency)
            else:
                idf_values_user[term] = 0.0

        print("--- Enter Queries for Scoring ---")
        query1_input = input("Enter the first query terms, separated by commas (e.g., apple, banana): ")
        query1_user = [q.strip().lower() for q in query1_input.split(',')]

        query2_input = input("Enter the second query terms, separated by commas (e.g., orange, apple): ")
        query2_user = [q.strip().lower() for q in query2_input.split(',')]

        tf_idf_weights_user = calculate_tf_idf_weights(tf_table_user, idf_values_user)
        scores1_user = {'scores': calculate_query_score(query1_user, tf_idf_weights_user, documents), 'query_string': ', '.join(query1_user)}
        scores2_user = {'scores': calculate_query_score(query2_user, tf_idf_weights_user, documents), 'query_string': ', '.join(query2_user)}
        normalized_tf_user = euclidean_normalize_tf(tf_table_user, documents, terms)
        
        print("\n--- Results for User-Provided Data ---\n")
        print_tables("User Data", tf_idf_weights_user, scores1_user, scores2_user, normalized_tf_user)

    except (ValueError, IndexError) as e:
        print(f"\nInvalid input: {e}. Please ensure data is correctly formatted.")

if __name__ == "__main__":
    # --- Case given in the question ---
    print("--- Results for the Case Data ---\n")
    tf_table_case = {
        'car': {'Doc1': 27, 'Doc2': 4, 'Doc3': 24},
        'auto': {'Doc1': 3, 'Doc2': 33, 'Doc3': 0},
        'insurance': {'Doc1': 0, 'Doc2': 33, 'Doc3': 29},
        'best': {'Doc1': 14, 'Doc2': 0, 'Doc3': 17}
    }
    idf_values_case = {
        'car': 1.65, 'auto': 2.08, 'insurance': 1.62, 'best': 1.5
    }
    tf_idf_weights_case = calculate_tf_idf_weights(tf_table_case, idf_values_case)

    scores1_case = {'scores': calculate_query_score(['car', 'insurance'], tf_idf_weights_case, ['Doc1', 'Doc2', 'Doc3']), 'query_string': 'car, insurance'}
    scores2_case = {'scores': calculate_query_score(['best', 'car'], tf_idf_weights_case, ['Doc1', 'Doc2', 'Doc3']), 'query_string': 'best, car'}

    normalized_tf_case = euclidean_normalize_tf(tf_table_case, ['Doc1', 'Doc2', 'Doc3'], list(tf_table_case.keys()))
    print_tables("Case Data", tf_idf_weights_case, scores1_case, scores2_case, normalized_tf_case)

    get_user_input_and_process()