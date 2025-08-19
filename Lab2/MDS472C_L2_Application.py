import re

def get_splits(word):
    return [(word[:i], word[i:]) for i in range(len(word) + 1)]

def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits     = get_splits(word)
    deletes    = [L + R[1:]           for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]       for L, R in splits if R for c in letters]
    inserts    = [L + c + R           for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def known(words, vocab):
    return set(w for w in words if w in vocab)

def correct(word, vocab):
    # correction logic:
    candidates = (known([word], vocab) or
                  known(edits1(word), vocab) or
                  [word])
    return candidates

def build_vocab(text):
    # quick vocab builder from user text
    words = re.findall(r'\w+', text.lower())
    return set(words)

if __name__ == "__main__":
    # build vocabulary from user input text
    text = input("Enter your reference text to build the vocabulary:\n").strip()
    vocabulary = build_vocab(text)
    print(f"\nVocabulary created ({len(vocabulary)} words): {vocabulary}")
    
    while True:
        word = input("\nEnter a word to spell-check (or type 'exit' to quit): ").strip().lower()
        if word == "exit":
            print("Goodbye!")
            break
        suggestions = correct(word, vocabulary)
        print(f"Suggested corrections: {suggestions}")
