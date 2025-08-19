import nltk
nltk.download('webtext')
from nltk.corpus import webtext
text6 = webtext.words('firefox.txt')  # Example file from webtext corpus

#24
# a. Words ending in 'ize'
words_ending_ize = [w for w in text6 if w.endswith('ize')]
print(words_ending_ize[:10])

# b. Words containing 'z'
words_contain_z = [w for w in text6 if 'z' in w]
print(words_contain_z[:10])

# c. Words containing 'pt'
words_contain_pt = [w for w in text6 if 'pt' in w]
print(words_contain_pt[:10])

# d. Words with all lowercase except initial capital (titlecase)
titlecase_words = [w for w in text6 if w.istitle()]
print(titlecase_words[:10])

#25
sent = ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']

# a. Words beginning with 'sh'
print([w for w in sent if w.startswith('sh')])

# b. Words longer than 4 characters
print([w for w in sent if len(w) > 4])

#26
import nltk
from nltk.book import text1  # 'Moby Dick' text

total_chars = sum(len(w) for w in text1)
total_words = len(text1)
average_word_length = total_chars / total_words

print(f"Average word length: {average_word_length:.2f}")

#27
def vocab_size(text):
    return len(set(text))


print("Vocabulary size of text1:", vocab_size(text1))

#28
def percent(word, text):
    count = text.count(word)
    total = len(text)
    return 100 * count / total if total > 0 else 0


print(f"Percentage of 'whale' in text1: {percent('whale', text1):.2f}%")

