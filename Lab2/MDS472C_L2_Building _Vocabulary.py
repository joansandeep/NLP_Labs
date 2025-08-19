import re
from collections import Counter
import matplotlib.pyplot as plt

def preprocess_text(text):
    # Convert to lowercase
    text_lower = text.lower()
    # Tokenize words using regex
    words = re.findall(r'\w+', text_lower)
    return words

def option_1_vocab(words):
    vocab = set(words)
    print("\nVocabulary set (unique words):")
    print(vocab)
    print("Vocabulary size:", len(vocab))

def option_2_vocab_counts(words):
    counts = Counter(words)
    print("\nWord counts:")
    print(counts)
    print("Number of distinct words:", len(counts))
    
    # Plot bar chart for the top 5 words
    top_words = counts.most_common(5)
    labels, values = zip(*top_words)
    
    plt.bar(range(len(labels)), values, color='skyblue')
    plt.xticks(range(len(labels)), labels)
    plt.title("Top 5 Word Counts")
    plt.show()

if __name__ == "__main__":
    text = input("Enter your text corpus:\n").strip()
    words = preprocess_text(text)
    print("\nAfter preprocessing:")
    print(f"Tokenized words: {words}")
    print(f"Total words: {len(words)}")
    
    while True:
        print("\nChoose an option:")
        print("1. Create vocabulary (unique words)")
        print("2. Create vocabulary with word counts")
        print("3. Exit")
        
        choice = input("Enter 1, 2 or 3: ").strip()
        
        if choice == '1':
            option_1_vocab(words)
        elif choice == '2':
            option_2_vocab_counts(words)
        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid choice, please enter 1, 2 or 3.")
