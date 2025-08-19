#2.6 Implement a minimum edit distance algorithm and use your hand-computed results to check your code.
def levenshtein_distance(s, t):
    m = len(s)
    n = len(t)
    dp = [[0]*(n+1) for _ in range(m+1)]
    
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
        
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s[i-1] == t[j-1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    
    return dp[m][n]

if __name__ == "__main__":
    while True:
        str1 = input("Enter string 1: ").strip()
        str2 = input("Enter string 2: ").strip()
        distance = levenshtein_distance(str1, str2)
        print(f"The edit distance between '{str1}' and '{str2}' is {distance} \n")

        choice = input("Do you want to continue? (yes/no): ").strip().lower()
        if choice != "yes":
            print("Exiting program. Goodbye!")
            break