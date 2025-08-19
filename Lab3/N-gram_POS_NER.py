import nltk
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter, defaultdict
from nltk import word_tokenize, pos_tag, bigrams
from nltk.probability import FreqDist
from spacy import displacy
import math

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp_spacy = spacy.load("en_core_web_sm")

def pos_tagger():
    text = input("Enter a sentence for POS tagging: ")
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    print("\nPOS Tags:")
    for word, tag in pos_tags:
        print(f"{word}: {tag}")
    visualize_pos_tags(pos_tags)

def visualize_pos_tags(tags):
    tag_counts = Counter([tag for word, tag in tags])
    plt.figure(figsize=(10, 5))
    plt.bar(tag_counts.keys(), tag_counts.values(), color='skyblue')
    plt.title("POS Tag Frequency")
    plt.xlabel("POS Tag")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def ngram_model():
    text = input("Enter text for training N-gram model: ").lower()
    tokens = word_tokenize(text)

    # Unigram
    fdist = FreqDist(tokens)
    total = len(tokens)
    print("\nUnigram Probabilities:")
    for word in fdist:
        print(f"P({word}) = {fdist[word]/total:.4f}")

    # Bigram
    bigram_list = list(bigrams(tokens))
    bigram_freq = FreqDist(bigram_list)
    word_freq = FreqDist(tokens)
    bigram_probs = {bg: freq / word_freq[bg[0]] for bg, freq in bigram_freq.items()}

    print("\nBigram Probabilities:")
    for bg in bigram_probs:
        print(f"P({bg[1]}|{bg[0]}) = {bigram_probs[bg]:.4f}")

    # Specific probability checks
    print("\nSpecific Bigram Probability:")
    print("P(Sam|am) =", bigram_probs.get(('am', 'sam'), 0.0))
    print("P(green|like) =", bigram_probs.get(('like', 'green'), 0.0))

    visualize_bigram_probs(bigram_probs)
    
    # Perplexity
    test_text = input("Enter a test sentence to compute perplexity: ").lower()
    test_tokens = word_tokenize(test_text)
    test_bigrams = list(bigrams(test_tokens))
    prob_product = 1.0
    N = len(test_bigrams)
    for bg in test_bigrams:
        prob = bigram_probs.get(bg, 1e-6)  # small smoothing factor
        prob_product *= prob
    perplexity = math.pow(1 / prob_product, 1 / N) if N > 0 else float('inf')
    print(f"Perplexity of test sentence: {perplexity:.4f}")

def visualize_bigram_probs(bigram_probs):
    data = []
    for (w1, w2), prob in bigram_probs.items():
        data.append([w1, w2, prob])
    df = pd.DataFrame(data, columns=['w1', 'w2', 'prob'])
    pivot = df.pivot(index='w1', columns='w2', values='prob').fillna(0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Bigram Transition Probabilities")
    plt.show()

def named_entity_recognition():
    text = input("Enter a sentence for Named Entity Recognition: ")
    doc = nlp_spacy(text)
    print("\nEntities:")
    for ent in doc.ents:
        print(f"{ent.text}: {ent.label_}")
    print("\nTagged Output:")
    for token in doc:
        if token.ent_iob_ != 'O':
            print(f"{token.text}: {token.ent_iob_}-{token.ent_type_}")
        else:
            print(f"{token.text}: O")
    displacy.render(doc, style="ent", jupyter=False)

# Main Menu
while True:
    print("\nSelect NLP Task:")
    print("1. POS Tagging")
    print("2. N-Gram Language Modeling")
    print("3. Named Entity Recognition")
    print("4. Exit")
    choice = input("Enter your choice (1/2/3/4): ")

    if choice == '1':
        pos_tagger()
    elif choice == '2':
        ngram_model()
    elif choice == '3':
        named_entity_recognition()
    elif choice == '4':
        print("Exiting...")
        break
    else:
        print("Invalid choice. Try again.")
