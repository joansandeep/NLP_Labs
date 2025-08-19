from Bio import pairwise2

def run_sequence_alignment():
    while True:
        a = input("Enter sequence A: ").strip()
        b = input("Enter sequence B: ").strip()
        alignments = pairwise2.align.globalxx(a, b)
        print("Top alignment:")
        print(pairwise2.format_alignment(*alignments[0]))

        cont = input("Do you want to continue (yes/no)? ").strip().lower()
        if cont != "yes":
            break

if __name__ == "__main__":
    run_sequence_alignment()
