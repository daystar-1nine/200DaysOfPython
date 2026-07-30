# ==============================================================================
# Program    : Coin Toss Simulator
# Objective  : Simulate tossing a 50/50 fair coin.
# Concept    : Random Choice Selection (random.choice)
# Why Used   : random.choice() picks a random element from a sequence.
# ==============================================================================

# What is used : import random
import random

def toss_coin():
    # What is used : List of coin outcomes
    outcomes = ["Heads", "Tails"]

    # What is used : random.choice(outcomes)
    # Why it is used: Selects one random element from the list with equal 50% probability
    # How it works : Index = randbelow(len(outcomes)); returns outcomes[Index]
    result = random.choice(outcomes)
    return result

print("Tossing a coin...")
result = toss_coin()
print(f"Result: [Coin Toss -> {result}]")
