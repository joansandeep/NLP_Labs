import subprocess

while True:
    print("\n=== MAIN MENU ===")
    print("1. Run Minimum Edit Distance ")
    print("2. Run Sequence Alignment ")
    print("3. Exit")
    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == "1":
        subprocess.run(["python", "Lab2\MDS472C_L2_Edit_distance.py"])
    elif choice == "2":
        subprocess.run(["python", "Lab2\MDS472C_L2_ Sequence _lignment.py"])
    elif choice == "3":
        print("Goodbye!")
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
