# ==============================================================================
# Program    : Dice Rolling Simulator
# Objective  : Interactive CLI tool to roll single or double 6-sided dice.
# Concept    : Probabilistic Simulation & Menu Control Loop (random.randint)
# Why Used   : random.randint(1, 6) models independent uniform die roll outcomes.
# ==============================================================================

# What is used : import random
import random

def roll_single_dice():
    # What is used : random.randint(1, 6)
    # Why it is used: Simulates rolling 1 single 6-sided die
    # How it works : Returns random integer between 1 and 6 inclusive
    roll = random.randint(1, 6)
    print(f"\nResult: [Dice {roll}]")

def roll_two_dice():
    # What is used : Two independent random.randint(1, 6) calls
    # Why it is used: Simulates rolling 2 independent dice simultaneously
    # How it works : Generates roll1 and roll2 independently and computes sum
    roll1 = random.randint(1, 6)
    roll2 = random.randint(1, 6)
    total = roll1 + roll2
    print(f"\nResult: [Dice {roll1}]  [Dice {roll2}]  (Total: {total})")

def main():
    # What is used : while True infinite menu loop
    # Why it is used: Keeps program interactive until user explicitly selects exit option (3)
    while True:
        print("\n=== DICE ROLLING SIMULATOR ===")
        print("1. Roll Single Dice")
        print("2. Roll Two Dice")
        print("3. Exit")

        choice = input("Select option (1-3): ").strip()

        # What is used : Conditional branching (if-elif-else)
        if choice == "1":
            roll_single_dice()
        elif choice == "2":
            roll_two_dice()
        elif choice == "3":
            print("Exiting Dice Rolling Simulator. Goodbye!")
            break
        else:
            print("Invalid selection! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
