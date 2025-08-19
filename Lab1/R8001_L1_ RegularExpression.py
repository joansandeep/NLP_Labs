# Basic Regulsr Expressionns
import re

text="The simplest kind of regular expression is a sequence of Simple characters, then  1 by 1 we will learn"
match=re.findall(r"simple",text)
print("\n")
print(match)

match=re.findall("[Ss]imple",text)
print("\n")
print(match)
print("\n")

match=re.findall("[A-Z]",text)
print(match)
print("\n")

match=re.findall("[0-9]",text)
print(match)
print("\n")

match=re.findall("[^a-z]",text)
print(match)
print("\n")

match=re.findall("[Tt]hen?",text)
print(match)
print("\n")

match=re.findall("a*",text)
print(match)
print("\n")

match=re.findall("then+",text)
print(match)
print("\n")

match=re.findall("aa*",text)
print(match)
print("\n")

match=re.findall("w.",text)
print(match)
print("\n")

match=re.findall("^n",text)
print(match)
print("\n")

match=re.findall("[Tt]$",text)
print(match)
print("\n")

#Disjunction, Grouping, and Precedence
text= "in this we will see the use of disjunction, grouping and precedence"
match=re.findall("in|this",text)
print(match)
print("\n")

match=re.findall("th(is|e)",text)
print(match)
print("\n")

#A Simple Example
text="In The jungle a quick brown fox jumps into the bush"
match=re.findall("in",text)
print(match)
print("\n")

match=re.findall("[Ii]n",text)
print(match)
print("\n")

match=re.findall("\b[Ii]n\b",text)
print(match)
print("\n")

match=re.findall(r"\b[Ii]n\b",text)
print(match)
print("\n")

match=re.findall("(?:^|[^a-zA-Z])([Ii]n)(?=[^a-zA-Z]|$)",text)
print(match)
print("\n")

#A More Complex Example
text="any PC with more than 500 MHz and 32 Gb of disk space for less than $1000.5. soso sososo"
match=re.findall(r"\$[0-9]+",text)
print(match)
print("\n")

match=re.findall(r"\$[0-9]+(?:\.[0-9]+)?",text)
print(match)
print("\n")

match=re.findall(r"\b[0-9]+ *(MHz|[Mm]egahertz|GHz|[Gg]igahertz)\b",text)
print(match)
print("\n")

match=re.findall("\W",text)
print(match)
print("\n")

match=re.findall("(?:so){3}",text)
print(match)
print("\n")

text1 = "The bigger they were, the bigger they will be"
text2 = "The bigger they were, the faster they will be"

pattern = r"the (.*)er they were, the \1er they will be"

# Using re.IGNORECASE so "The" matches "the"
match1 = re.search(pattern, text1, re.IGNORECASE)
match2 = re.search(pattern, text2, re.IGNORECASE)

print("Match 1:", bool(match1))
print("Match 2:", bool(match2))
