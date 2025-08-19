def get_splits(word):
    return [(word[:i], word[i:]) for i in range(len(word) + 1)]

def deletes(splits):
    return [L + R[1:] for L, R in splits if R]

def filter_candidates(edits, vocab):
    return set(vocab).intersection(edits)

if __name__ == "__main__":
    word = input("Enter a word to edit: ").strip()
    
    # Step 1: Get splits
    splits = get_splits(word)
    print("\nSplits:")
    for s in splits:
        print(s)
    
    # Step 2: Generate deletes
    deletes_list = deletes(splits)
    print("\nDeletes (candidate strings after one deletion):")
    for d in deletes_list:
        print(d)
    
    # Step 3: Input or predefined vocabulary
    vocab_input = input("\nEnter vocabulary words separated by space (or leave blank for default): ").strip()
    if vocab_input:
        vocab = vocab_input.split()
    else:
        vocab = ['dean', 'deer', 'dear', 'fries', 'and', 'coke']
    
    print("\nVocabulary:")
    print(vocab)
    
    # Step 4: Filter candidates by vocab
    candidates = filter_candidates(deletes_list, vocab)
    print("\nCandidate words (valid deletions found in vocabulary):")
    print(candidates)
