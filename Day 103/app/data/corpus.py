"""Corpus provider for Word2Vec training."""
from typing import List

# Controlled, rich corpus with distinct semantic domains:
# 1. Pets / Domestic Animals (cat, kitten, dog, puppy, drinks, milk, water, likes, fish, meat)
# 2. Wild / Farm Animals (lion, tiger, cow, sheep, eats, grass, meat)
# 3. Technology / Programming (python, java, programmer, developer, writes, code, software, builds)
DEFAULT_CORPUS = [
    "the cat drinks milk",
    "the dog drinks milk",
    "the cat likes fish",
    "the dog likes meat",
    "the kitten drinks milk",
    "the puppy drinks milk",
    "the kitten likes fish",
    "the puppy likes meat",
    "animals drink water",
    "animals eat food",
    "the lion eats meat",
    "the tiger eats meat",
    "the cow eats grass",
    "the sheep eats grass",
    "programmer writes python code",
    "developer writes java code",
    "programmer builds software",
    "developer builds software",
    "python is great for machine learning",
    "java is used for enterprise software",
    "the cat sleeps on the mat",
    "the dog sleeps on the mat",
    "the kitten plays with a ball",
    "the puppy plays with a ball"
]

def get_corpus() -> List[str]:
    """Returns the training corpus."""
    return DEFAULT_CORPUS
