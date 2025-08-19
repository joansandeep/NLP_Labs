# Import required modules
import string
from collections import Counter

# 1) Define a string containing a paragraph as the value
paragraph = """
Natural Language Processing (NLP) is a subfield of artificial intelligence that deals with the interaction 
between computers and humans through natural language. It enables machines to understand, interpret, and 
generate human language in a valuable way.
"""

# Preprocessing: convert to lowercase and remove punctuation
cleaned_text = paragraph.lower().translate(str.maketrans('', '', string.punctuation))

# Tokenize the paragraph into words
words = cleaned_text.split()

# 2) Count total words and total unique words
total_words = len(words)
unique_words = len(set(words))

# 3) Calculate word frequency
word_freq = Counter(words)

# Get the most frequent word
most_frequent = word_freq.most_common(1)[0]

# Get the least frequent word
least_frequent = min(word_freq.items(), key=lambda item: item[1])

# 4) Find the longest word
longest_word = max(words, key=len)

# Print Results
print("Total number of words:", total_words)
print("Total number of unique words:", unique_words)
print("\nWord Frequencies:")
for word, freq in word_freq.items():
    print(f"{word}: {freq}")

print("\nMost Frequent Word:", most_frequent)
print("Least Frequent Word:", least_frequent)
print("Longest Word:", longest_word)