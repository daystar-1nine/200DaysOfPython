# ==============================================================================
# Program    : Coin Toss Simulator
# Objective  : Simulate tossing a 50/50 fair coin.
# Concept    : Random Choice Selection (random.choice)
# Why Used   : random.choice() selects an item randomly from any non-empty sequence with equal probability.
# ==============================================================================

# What is used : import random
import random

# What is used : Function definition 'def toss_coin()'
# Why it is used: Encapsulates coin toss simulation logic
def toss_coin():
    # What is used : Python List containing string options
    # Why it is used: Represents discrete 2-sided coin outcomes
    outcomes = ["Heads", "Tails"]

    # What is used : random.choice(outcomes)
    # Why it is used: Selects one element from list with equal 50% probability
    # How it works : Generates random index 0 or 1 and returns outcomes[index]
    result = random.choice(outcomes)
    return result

print("Tossing a coin...")
# What is used : Function invocation
result = toss_coin()
print(f"Result: [Coin Toss -> {result}]")
