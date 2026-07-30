# ==============================================================================
# Program    : Dice Rolling Simulator
# Objective  : Interactive CLI tool to roll single or double 6-sided dice.
# Concept    : Simulation & Menu Loop (random.randint)
# Why Used   : random.randint(1, 6) models independent probabilistic die rolls.
# ==============================================================================

# What is used : import random
import random

def roll_single_dice():
    # What is used : random.randint(1, 6)
    roll = random.randint(1, 6)
    print(f"\nResult: [Dice {roll}]")

def roll_two_dice():
    roll1 = random.randint(1, 6)
    roll2 = random.randint(1, 6)
    total = roll1 + roll2
    print(f"\nResult: [Dice {roll1}]  [Dice {roll2}]  (Total: {total})")

def main():
    while True:
        print("\n=== DICE ROLLING SIMULATOR ===")
        print("1. Roll Single Dice")
        print("2. Roll Two Dice")
        print("3. Exit")

        choice = input("Select option (1-3): ").strip()

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
