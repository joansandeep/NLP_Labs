import nltk
nltk.download('punkt_tab')
from nltk.book import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 1.Computing with Language: Texts and Words
# Concordances
print("\n--- text1: concordance for 'monstrous' ---")
text1.concordance("monstrous")

print("\n--- text2: concordance for 'sensibilty' (misspelled) ---")
text2.concordance("sensibilty") 

print("\n--- text2: concordance for 'sensibility' ---")
text2.concordance("sensibility")

print("\n--- text3: concordance for 'lived' ---")
text3.concordance("lived")

print("\n--- text4: concordance for 'nation' ---")
text4.concordance("nation")

print("\n--- text4: concordance for 'terror' ---")
text4.concordance("terror")

print("\n--- text4: concordance for 'god' ---")
text4.concordance("god")

print("\n--- text5: concordance for 'im' ---")
text5.concordance("im")

print("\n--- text5: concordance for 'ur' ---")
text5.concordance("ur")

print("\n--- text5: concordance for 'lol' ---")
text5.concordance("lol")

# Similar words
print("\n--- text1: similar to 'monstrous' ---")
text1.similar("monstrous")

print("\n--- text2: similar to 'monstrous' ---")
text2.similar("monstrous")

print("\n--- text4: similar to 'god' ---")
text4.similar("god")

# Common contexts
print("\n--- text2: common contexts for 'monstrous' and 'very' ---")
text2.common_contexts(["monstrous", "very"])

print("\n--- text4: common contexts for 'god' and 'war' ---")
text4.common_contexts(["god", "war"])

# Dispersion plot
print("\n--- text4: dispersion plot for political words ---")

text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])
plt.show()

print("\n--- Generating random text from text3 ---")
text3.generate()

print("\n--- number of characters in text3: ---\n", len(text3))

print("\n--- number of vocabulary items of text3: ---\n",sorted(set(text3)),"\n\n --- and length of it is ---\n",len(sorted(set(text3))))

print("\n --- Lexical Richness of Text3 --- \n",len(set(text3)) / len(text3))

print("\n --- how many time these words come in their respective texts ---\n","the word 'smote' comes in text3 :",text3.count("smote"),
      "times","\n","with the percentage of",100 * text3.count('smote') / len(text3),"\n\n",
      "the word 'a' comes in text4 :",text4.count("a"),"times","\n","with the percentage of",100 * text4.count('a') / len(text4),"\n\n",
      "the word 'lol comes in text5 :",text5.count("lol"),"times","\n","with the percentage of",100 * text5.count('lol') / len(text5))

def lexical_diversity(text):
    return len(set(text)) / len(text)

def percentage(count,total):
    return (count / total) * 100

print("\n --- lexical diversity of text2 ---\n",lexical_diversity(text2))
print("\n --- lexical diversity of text6 ---\n",lexical_diversity(text6))

print("\n percntage example :",percentage(3,27))
print(" percentage of 'money' in text3 :",percentage(text3.count('money'),len(text3)))

# 2) A Closer Look at Python: Texts as Lists of Words
#Lists
sent1 = ['Call', 'me', 'Ishmael', '.']
print("\n sent1",sent1)
print("length of sent1",len(sent1))
print("lexical diversity of sent1",lexical_diversity(sent1))

persons = ['albin', 'shawn', 'bennison', 'antony','albin']
print("\n sent1",persons)
print("length of sent1",len(persons))
print("lexical diversity of sent1",lexical_diversity(persons))

print("\naddition in lists",sent1 + persons)
print("\naddition in lists",['Monty', 'Python'] + ['and', 'the', 'Holy', 'Grail'])

persons.append("joan")
print("\nappend in list of persons",persons)

#Indexing List
print("\n173rd word in text4: ",text4[173])
print("\n100th word in text5: ",text5[100])

print("\n the word Egyptian's index number in text3 is",text3.index('Egyptian'))

sent = ['word1', 'word2', 'word3', 'word4', 'word5','word6', 'word7', 'word8', 'word9', 'word10']
print(sent)
print("indexing will always start from 0 so sent[0] will be ",sent[0])

print("\n the words between 50-55 index numbers in text2",text2[50:56])

sent[0] = 'First'
sent[9] = 'Last'
print("\n length of sent",len(sent),"\n sent:",sent)
sent[1:9] = ['Second', 'Third']
print("\n updated sent",sent)

#variables
my_sent = ['Bravely', 'bold', 'Sir', 'Robin', ',',
            'rode','forth', 'from', 'Camelot', '.']
noun_phrase = my_sent[:4]
print("\n noun phrase",noun_phrase)
print("\n sorted noun phrase",sorted(noun_phrase))

vocab = set(text1)
vocab_size = len(vocab)
print("vocabulary size of text1",vocab_size)

#strings
name = "joker"
print("\n name",name)
print("\n name[0] is",name[0])
print("\n name[:4] is",name[:4])

print("\n after adding ! :\n",name + '!')
print("\n Multiplicating itself : \n",name * 3)

my_name=['joan','sandeep','larson']
full_name=' '.join(my_name)
print("\n full name:\n",full_name)

splitted_name=full_name.split()
print("\n splitted name:\n",splitted_name)