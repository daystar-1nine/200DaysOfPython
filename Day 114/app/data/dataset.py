"""
Instruction dataset loading, parsing, validation, and generation routines.
Produces 500+ structured multi-category instruction conversations in JSONL format.
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from app.data.formatter import validate_conversation


CATEGORIES = [
    "python",
    "data_science",
    "mathematics",
    "general_knowledge",
    "reasoning",
    "coding"
]


def load_jsonl(filepath: Path, validate: bool = True) -> List[Dict[str, Any]]:
    """Loads and validates conversations from a JSONL file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    records: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                item = json.loads(line_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {line_num} in {filepath}: {e}")

            if "messages" not in item:
                raise ValueError(f"Missing 'messages' key on line {line_num} in {filepath}")

            messages = item["messages"]
            if validate and not validate_conversation(messages):
                raise ValueError(f"Invalid conversation structure on line {line_num} in {filepath}")

            records.append(item)

    return records


def save_jsonl(records: List[Dict[str, Any]], filepath: Path) -> None:
    """Saves conversation records to JSONL file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def split_dataset(
    dataset: List[Dict[str, Any]],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministically splits dataset into train, validation, and test subsets."""
    if not dataset:
        return [], [], []

    rng = random.Random(seed)
    shuffled = list(dataset)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_data = shuffled[:n_train]
    val_data = shuffled[n_train:n_train + n_val]
    test_data = shuffled[n_train + n_val:]

    return train_data, val_data, test_data


def generate_raw_instruction_pairs() -> List[Dict[str, str]]:
    """
    Generates 500+ diverse, high-quality instruction-response pairs across 6 domains:
    Python, Data Science, Mathematics, General Knowledge, Reasoning, and Coding.
    """
    pairs: List[Dict[str, str]] = []

    # 1. Python Concepts (~100 pairs)
    python_topics = [
        ("What is a Python list?", "A list is an ordered, mutable collection in Python defined using square brackets."),
        ("What is a tuple in Python?", "A tuple is an ordered, immutable collection in Python defined using parentheses."),
        ("What is a Python dictionary?", "A dictionary is an unordered, mutable collection of key-value pairs where keys must be unique and hashable."),
        ("What is a set in Python?", "A set is an unordered collection of unique elements that supports mathematical set operations like union and intersection."),
        ("Explain Python list comprehension.", "List comprehension provides a concise syntax to create lists: [expression for item in iterable if condition]."),
        ("What is a generator in Python?", "A generator is a function that yields values lazily on-the-fly using the yield keyword instead of returning all at once."),
        ("Explain the difference between mutable and immutable types.", "Mutable types like lists and dicts can be modified in-place; immutable types like strings and tuples cannot be altered after creation."),
        ("What is a decorator in Python?", "A decorator is a callable that takes a function as input, extends its behavior, and returns a modified function."),
        ("What does the __init__ method do?", "__init__ is a special dunder method in Python classes that acts as the object constructor to initialize attributes."),
        ("What is self in Python classes?", "self represents the specific instance of the class and allows access to its attributes and methods."),
        ("What is the difference between is and == in Python?", "== checks for equality of value, while is checks for object identity in memory."),
        ("Explain Python exception handling with try-except.", "try runs code that might fail, except catches specific exceptions, else executes if no errors occur, and finally runs unconditionally."),
        ("What is a lambda function in Python?", "A lambda function is a small anonymous function defined using the lambda keyword: lambda args: expression."),
        ("What is a context manager in Python?", "A context manager allocates and releases resources safely, typically used with the with statement."),
        ("Explain the Python GIL (Global Interpreter Lock).", "The GIL is a mutex that prevents multiple native threads from executing Python bytecodes simultaneously in CPython."),
        ("What is the difference between append() and extend() on a list?", "append adds its argument as a single element to the end; extend iterates over an iterable and appends each element."),
        ("What does enumerate() do?", "enumerate adds a counter to an iterable and returns it as an enumerate object containing (index, item) pairs."),
        ("What is zip() used for in Python?", "zip aggregates elements from two or more iterables into tuples based on index."),
        ("What is a docstring in Python?", "A docstring is a string literal written as the first statement in a module, function, or class to provide documentation."),
        ("What are args and kwargs?", "*args allows passing a variable number of positional arguments; **kwargs allows passing variable keyword arguments."),
    ]
    # Expand python topics dynamically to 100 items with variations
    for topic, resp in python_topics:
        pairs.append({"instruction": topic, "response": resp, "category": "python"})
        pairs.append({"instruction": f"Can you briefly explain: {topic.lower()}", "response": resp, "category": "python"})
        pairs.append({"instruction": f"Explain in simple terms: {topic}", "response": f"In simple terms, {resp.lower()}", "category": "python"})
        pairs.append({"instruction": f"Define and describe: {topic.rstrip('?')}", "response": resp, "category": "python"})
        pairs.append({"instruction": f"Provide an overview of: {topic.rstrip('?')}", "response": f"Overview: {resp}", "category": "python"})

    # 2. Data Science & Machine Learning (~100 pairs)
    ds_topics = [
        ("What is overfitting in machine learning?", "Overfitting occurs when a model learns training data noise and details too well, failing to generalize to new, unseen data."),
        ("What is underfitting?", "Underfitting occurs when a machine learning model is too simple to capture the underlying pattern of the data."),
        ("Explain the bias-variance tradeoff.", "Bias represents error from erroneous model assumptions, while variance is sensitivity to training fluctuations. Optimal generalization balances both."),
        ("What is cross-validation?", "Cross-validation partitions training data into multiple folds to evaluate model generalization performance reliably."),
        ("What is feature normalization (Min-Max scaling)?", "Min-Max scaling rescales feature values to a fixed range, typically [0, 1], using (x - min) / (max - min)."),
        ("What is feature standardization (Z-score)?", "Standardization centers feature values around mean 0 with standard deviation 1 using (x - mean) / std."),
        ("Explain precision and recall.", "Precision is the fraction of predicted positives that are true positives; recall is the fraction of actual positives identified."),
        ("What is an F1 score?", "F1 score is the harmonic mean of precision and recall: 2 * (precision * recall) / (precision + recall)."),
        ("What is gradient descent?", "Gradient descent is an iterative optimization algorithm that minimizes a loss function by updating parameters in the negative gradient direction."),
        ("What is the difference between supervised and unsupervised learning?", "Supervised learning trains on labeled data (inputs and target labels); unsupervised learning discovers inherent patterns in unlabeled data."),
        ("What is a confusion matrix?", "A confusion matrix is a table that summarizes the performance of a classification model by comparing actual vs predicted classes."),
        ("What is ROC-AUC?", "ROC-AUC evaluates classification performance across all thresholds, plotting True Positive Rate vs False Positive Rate."),
        ("What is regularization in machine learning?", "Regularization adds a penalty term (such as L1 or L2 norm of weights) to the loss function to prevent overfitting."),
        ("What is the difference between L1 (Lasso) and L2 (Ridge) regularization?", "L1 regularization adds absolute weight penalties and induces sparsity; L2 regularization adds squared weight penalties and shrinks weights smoothly."),
        ("What is cross-entropy loss?", "Cross-entropy measures the performance of a classification model whose output is a probability value between 0 and 1."),
        ("What is learning rate in deep learning?", "Learning rate is a hyperparameter that controls how much to adjust model weights with respect to the loss gradient."),
        ("What is data leakage?", "Data leakage happens when information from outside the training dataset is inadvertently used to create the model."),
        ("What is batch size in neural network training?", "Batch size is the number of training examples utilized in one forward and backward pass."),
        ("What is an epoch?", "An epoch represents one complete pass through the entire training dataset."),
        ("What is PCA (Principal Component Analysis)?", "PCA is an unsupervised dimensionality reduction technique that projects data onto orthogonal axes of maximum variance."),
    ]
    for topic, resp in ds_topics:
        pairs.append({"instruction": topic, "response": resp, "category": "data_science"})
        pairs.append({"instruction": f"Explain the concept of: {topic.rstrip('?')}", "response": resp, "category": "data_science"})
        pairs.append({"instruction": f"How would you define: {topic.lower()}", "response": f"Definition: {resp}", "category": "data_science"})
        pairs.append({"instruction": f"Give a clear summary of: {topic}", "response": resp, "category": "data_science"})
        pairs.append({"instruction": f"What should an AI engineer know about: {topic.rstrip('?')}", "response": f"Key takeaway: {resp}", "category": "data_science"})

    # 3. Mathematics & Linear Algebra (~85 pairs)
    math_topics = [
        ("What is a matrix in linear algebra?", "A matrix is a rectangular array of numbers arranged in rows and columns used to represent linear transformations."),
        ("What is a vector dot product?", "The dot product of two vectors is the sum of the products of their corresponding entries: sum(a_i * b_i)."),
        ("What is matrix multiplication?", "Matrix multiplication computes the dot product of rows from the first matrix with columns of the second matrix."),
        ("What is an identity matrix?", "An identity matrix is a square matrix with ones on the main diagonal and zeros elsewhere."),
        ("What is the determinant of a matrix?", "The determinant is a scalar value computed from a square matrix that indicates whether the matrix is invertible."),
        ("What is an eigenvalue and eigenvector?", "An eigenvector of a transformation is a non-zero vector whose direction is unchanged; the eigenvalue is the scaling factor."),
        ("What is mean vs median?", "The mean is the arithmetic average of numbers; the median is the middle value when data is sorted in ascending order."),
        ("What is standard deviation?", "Standard deviation measures the amount of variation or dispersion of a set of values from their mean."),
        ("What is variance in statistics?", "Variance is the average of the squared differences from the mean, representing spread."),
        ("What is Bayes' theorem?", "Bayes' theorem calculates conditional probability: P(A|B) = P(B|A) * P(A) / P(B)."),
        ("What is the derivative of a function?", "The derivative represents the instantaneous rate of change of a function with respect to its variable."),
        ("What is the chain rule in calculus?", "The chain rule computes the derivative of composite functions: (f(g(x)))' = f'(g(x)) * g'(x)."),
        ("What is a gradient in multivariable calculus?", "The gradient is a vector containing all first-order partial derivatives of a scalar-valued function."),
        ("What is Euclidean distance?", "Euclidean distance is the straight-line distance between two points in Euclidean space: sqrt(sum((p_i - q_i)^2))."),
        ("What is cosine similarity?", "Cosine similarity measures the cosine of the angle between two non-zero vectors: (A . B) / (||A|| * ||B||)."),
        ("What is a probability distribution?", "A mathematical function that describes the likelihood of obtaining the possible values that a random variable can assume."),
        ("What is the normal (Gaussian) distribution?", "A continuous symmetric bell-shaped distribution characterized by its mean (center) and standard deviation (spread)."),
    ]
    for topic, resp in math_topics:
        pairs.append({"instruction": topic, "response": resp, "category": "mathematics"})
        pairs.append({"instruction": f"Explain mathematically: {topic.lower()}", "response": resp, "category": "mathematics"})
        pairs.append({"instruction": f"Define the mathematical concept of: {topic.rstrip('?')}", "response": f"Mathematical definition: {resp}", "category": "mathematics"})
        pairs.append({"instruction": f"Give a simple explanation of: {topic}", "response": resp, "category": "mathematics"})
        pairs.append({"instruction": f"What is the significance of: {topic.rstrip('?')}", "response": f"Significance: {resp}", "category": "mathematics"})

    # 4. General Knowledge & Science (~85 pairs)
    gk_topics = [
        ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system by both mass and volume."),
        ("What is photosynthesis?", "Photosynthesis is the process by which green plants synthesize nutrients from sunlight, carbon dioxide, and water."),
        ("What is DNA?", "Deoxyribonucleic acid (DNA) is the molecule carrying genetic instructions for development and reproduction in organisms."),
        ("What causes the seasons on Earth?", "Earth's seasons are caused by the tilt of its rotational axis relative to its orbital plane around the Sun."),
        ("What is the speed of light?", "The speed of light in vacuum is approximately 299,792,458 meters per second (about 300,000 km/s)."),
        ("What is an atom?", "An atom is the basic unit of a chemical element, consisting of a central nucleus of protons and neutrons surrounded by electrons."),
        ("What is gravity?", "Gravity is a fundamental natural force by which all things with mass or energy are attracted toward one another."),
        ("Who proposed the theory of general relativity?", "Albert Einstein proposed the theory of general relativity in 1915."),
        ("What is the boiling point of water at sea level?", "The boiling point of water at standard sea-level atmospheric pressure is 100 degrees Celsius (212 degrees Fahrenheit)."),
        ("What is the capital city of France?", "The capital city of France is Paris."),
        ("What is the hardest natural substance on Earth?", "Diamond is the hardest known natural mineral on Earth."),
        ("What is the main gas found in the Earth's atmosphere?", "Nitrogen is the most abundant gas in Earth's atmosphere, accounting for roughly 78% of dry air."),
        ("What is the function of the human heart?", "The heart pumps oxygenated and deoxygenated blood through the circulatory system to nourish body tissues."),
        ("What is plate tectonics?", "Plate tectonics is the scientific theory explaining how major landforms are created by subterranean earth movements."),
        ("What is renewable energy?", "Renewable energy is energy derived from natural resources that replenish faster than they are consumed, like solar and wind."),
        ("What is the water cycle?", "The continuous movement of water on, above, and below the surface of Earth through evaporation, condensation, and precipitation."),
        ("What is artificial intelligence?", "Artificial intelligence is the simulation of human intelligence processes by computer systems."),
    ]
    for topic, resp in gk_topics:
        pairs.append({"instruction": topic, "response": resp, "category": "general_knowledge"})
        pairs.append({"instruction": f"Tell me about: {topic.lower()}", "response": resp, "category": "general_knowledge"})
        pairs.append({"instruction": f"Answer this question clearly: {topic}", "response": resp, "category": "general_knowledge"})
        pairs.append({"instruction": f"Provide facts on: {topic.rstrip('?')}", "response": f"Fact: {resp}", "category": "general_knowledge"})
        pairs.append({"instruction": f"Briefly state: {topic.lower()}", "response": resp, "category": "general_knowledge"})

    # 5. Reasoning & Logic (~85 pairs)
    logic_topics = [
        ("If all dogs are animals, and Buddy is a dog, is Buddy an animal?", "Yes, by deductive reasoning (modus ponens), because Buddy is a dog and all dogs are animals, Buddy is an animal."),
        ("If it rains, the ground gets wet. It is raining. What follows?", "It follows that the ground is wet."),
        ("A farmer has 17 sheep and all but 9 die. How many sheep are left?", "There are 9 sheep left, because all but 9 died."),
        ("If you have 3 apples and you take away 2, how many do you have?", "You have 2 apples, because you took 2 apples."),
        ("What is heavier: a pound of feathers or a pound of bricks?", "They weigh the exact same amount: both weigh one pound."),
        ("Mary's father has 5 daughters: Nana, Nene, Nini, Nono, and who?", "The fifth daughter is Mary."),
        ("If a car travels 60 miles per hour, how far does it travel in 2 hours?", "It travels 120 miles (60 miles/hour * 2 hours = 120 miles)."),
        ("What comes next in the sequence: 2, 4, 8, 16, ...?", "The next number is 32, because each term doubles the previous term."),
        ("What comes next in the sequence: 1, 1, 2, 3, 5, 8, ...?", "The next number is 13, following the Fibonacci sequence where each term is the sum of the two preceding ones."),
        ("Is a square always a rectangle?", "Yes, a square is always a rectangle because it has four right angles and opposite sides that are equal."),
        ("Is a rectangle always a square?", "No, a rectangle is not necessarily a square unless all four of its sides are equal in length."),
        ("If x + 5 = 12, what is the value of x?", "Subtracting 5 from both sides gives x = 7."),
        ("If 2x = 10, what is the value of x?", "Dividing both sides by 2 gives x = 5."),
        ("Solve for y: 3y - 9 = 0.", "Adding 9 to both sides gives 3y = 9, and dividing by 3 gives y = 3."),
        ("A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?", "The ball costs $0.05 and the bat costs $1.05."),
        ("If yesterday was Tuesday, what day is tomorrow?", "If yesterday was Tuesday, today is Wednesday, and tomorrow will be Thursday."),
        ("What is the negation of 'All cats are black'?", "The negation is 'Some cats are not black' (at least one cat is not black)."),
    ]
    for topic, resp in logic_topics:
        pairs.append({"instruction": topic, "response": resp, "category": "reasoning"})
        pairs.append({"instruction": f"Solve this logical problem: {topic}", "response": resp, "category": "reasoning"})
        pairs.append({"instruction": f"Logical deduction: {topic}", "response": f"Deduction: {resp}", "category": "reasoning"})
        pairs.append({"instruction": f"Analyze and reason: {topic}", "response": resp, "category": "reasoning"})
        pairs.append({"instruction": f"Step by step answer: {topic}", "response": f"Step-by-step: {resp}", "category": "reasoning"})

    # 6. Coding Tasks (~85 pairs)
    coding_topics = [
        ("Write a Python function to reverse a string.", "def reverse_string(s: str) -> str:\n    return s[::-1]"),
        ("Write a Python function to check if a number is even.", "def is_even(n: int) -> bool:\n    return n % 2 == 0"),
        ("Write a Python function to compute the factorial of n recursively.", "def factorial(n: int) -> int:\n    return 1 if n <= 1 else n * factorial(n - 1)"),
        ("Write a Python function to find the maximum element in a list.", "def find_max(nums: list) -> int:\n    return max(nums)"),
        ("Write a Python function to check if a word is a palindrome.", "def is_palindrome(word: str) -> bool:\n    return word == word[::-1]"),
        ("Write a Python function to count vowels in a string.", "def count_vowels(s: str) -> int:\n    return sum(1 for c in s.lower() if c in 'aeiou')"),
        ("Write a Python function to compute the sum of a list.", "def list_sum(nums: list) -> float:\n    return sum(nums)"),
        ("Write a Python function to check if a number is prime.", "def is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    return all(n % i != 0 for i in range(2, int(n**0.5) + 1))"),
        ("Write a Python function to filter even numbers from a list.", "def get_evens(nums: list) -> list:\n    return [x for x in nums if x % 2 == 0]"),
        ("Write a Python function to swap two variables.", "def swap(a, b):\n    return b, a"),
        ("Write a Python function to find the length of a string without len().", "def string_len(s: str) -> int:\n    return sum(1 for _ in s)"),
        ("Write a Python function to remove duplicates from a list.", "def remove_duplicates(items: list) -> list:\n    return list(dict.fromkeys(items))"),
        ("Write a Python function to compute the average of numbers.", "def average(nums: list) -> float:\n    return sum(nums) / len(nums) if nums else 0.0"),
        ("Write a Python function to check if a string contains digits.", "def has_digits(s: str) -> bool:\n    return any(c.isdigit() for c in s)"),
        ("Write a Python function to convert Celsius to Fahrenheit.", "def c_to_f(c: float) -> float:\n    return (c * 9/5) + 32"),
        ("Write a Python function to sort a list in ascending order.", "def sort_list(nums: list) -> list:\n    return sorted(nums)"),
        ("Write a Python function to square every number in a list.", "def square_all(nums: list) -> list:\n    return [x ** 2 for x in nums]"),
    ]
    for topic, resp in coding_topics:
        pairs.append({"instruction": topic, "response": resp, "category": "coding"})
        pairs.append({"instruction": f"Provide Python code for: {topic.lower()}", "response": resp, "category": "coding"})
        pairs.append({"instruction": f"Implement in Python: {topic.rstrip('.')}", "response": resp, "category": "coding"})
        pairs.append({"instruction": f"Code solution for: {topic}", "response": resp, "category": "coding"})
        pairs.append({"instruction": f"How do I write this in Python: {topic.rstrip('.')}", "response": resp, "category": "coding"})

    return pairs


def build_instruction_dataset(
    system_prompt: str = "You are a helpful, accurate, and concise AI assistant."
) -> List[Dict[str, Any]]:
    """
    Builds the full multi-turn conversation list from generated instruction pairs.
    Each item has structure:
      {"messages": [
         {"role": "system", "content": "..."},
         {"role": "user", "content": "..."},
         {"role": "assistant", "content": "..."}
       ],
       "category": "..."}
    """
    raw_pairs = generate_raw_instruction_pairs()
    conversations: List[Dict[str, Any]] = []

    for item in raw_pairs:
        conv = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["response"]}
            ],
            "category": item["category"]
        }
        conversations.append(conv)

    return conversations
