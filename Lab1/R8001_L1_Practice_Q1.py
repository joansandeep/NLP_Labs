import nltk
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import matplotlib.pyplot as plt

# Download required NLTK data (run once)
nltk.download('punkt')

# Read text file
with open('chapter.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Normalize: lowercase and tokenize
tokens = word_tokenize(text.lower())

# Frequency distribution using nltk
freq_dist = FreqDist(tokens)

# Print top 10 frequent words
print(freq_dist.most_common(10))

# Plot frequency distribution
freq_dist.plot(30, cumulative=False)
plt.show()

