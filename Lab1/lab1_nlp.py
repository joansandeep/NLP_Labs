import re
import nltk
import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.book import *

# ---------------- Function for Q1: Interactive Regular Expressions ----------------
def run_regular_expressions():
    text = input("Enter a text for Regular Expression examples: ")
    pattern = input("Enter a regex pattern to search: ")
    matches = re.findall(pattern, text)
    print(f"Matches found for '{pattern}':", matches)


# ---------------- Function for Q2: NLTK Concordance / Similar / Dispersion ----------------
def run_nltk_book():
    nltk.download('punkt_tab')
    word = input("Enter a word for concordance search (e.g., monstrous): ")
    print(text1.concordance(word))
    similar_word = input("Enter a word to find similar words: ")
    print(text2.similar(similar_word))
    words_to_plot = input("Enter words to plot (comma separated): ").split(",")
    text4.dispersion_plot(words_to_plot)
    plt.show()


# ---------------- Function for Q3: Word Frequency from File ----------------
def run_practice_q1():
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize
    from nltk import FreqDist

    file_name = input("Enter the file name (e.g., chapter.txt): ")
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print("File not found.")
        return

    tokens = word_tokenize(text.lower())
    freq_dist = FreqDist(tokens)
    n = int(input("How many top frequent words to display? "))
    print(freq_dist.most_common(n))
    freq_dist.plot(n, cumulative=False)
    plt.show()


# ---------------- Function for Q4: Filtering Words ----------------
def run_practice_q2():
    nltk.download('webtext')
    from nltk.corpus import webtext
    from nltk.book import text1

    corpus_file = input("Enter corpus file name from webtext (e.g., firefox.txt): ")
    try:
        text6 = webtext.words(corpus_file)
    except:
        print("Invalid file.")
        return

    ending = input("Find words ending with: ")
    print([w for w in text6 if w.endswith(ending)][:10])

    contains = input("Find words containing: ")
    print([w for w in text6 if contains in w][:10])

    prefix = input("Find words starting with: ")
    print([w for w in text6 if w.startswith(prefix)][:10])

    avg_len = sum(len(w) for w in text1) / len(text1)
    print(f"Average word length in text1: {avg_len:.2f}")


# ---------------- Function for Q5: Simple Text Processing ----------------
def run_text_processing():
    paragraph = input("Enter your paragraph: ")
    cleaned_text = paragraph.lower().translate(str.maketrans('', '', string.punctuation))
    words = cleaned_text.split()

    total_words = len(words)
    unique_words = len(set(words))
    word_freq = Counter(words)
    most_frequent = word_freq.most_common(1)[0]
    least_frequent = min(word_freq.items(), key=lambda x: x[1])
    longest_word = max(words, key=len)

    print("Total words:", total_words)
    print("Unique words:", unique_words)
    print("Most Frequent:", most_frequent)
    print("Least Frequent:", least_frequent)
    print("Longest Word:", longest_word)


# ---------------- Menu Driver ----------------
while True:
    print("\n--- MENU ---")
    print("1. Regular Expressions (interactive)")
    print("2. NLTK Concordances / Similar / Dispersion")
    print("3. Word Frequency from File")
    print("4. Filtering Words from Corpus")
    print("5. Simple Text Processing on user input")
    print("0. Exit")

    choice = input("Enter your choice: ")

    if choice == '1':
        run_regular_expressions()
    elif choice == '2':
        run_nltk_book()
    elif choice == '3':
        run_practice_q1()
    elif choice == '4':
        run_practice_q2()
    elif choice == '5':
        run_text_processing()
    elif choice == '0':
        print("Exiting...")
        break
    else:
        print("Invalid choice! Try again.")
