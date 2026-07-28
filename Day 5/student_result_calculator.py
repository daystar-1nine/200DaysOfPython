# ==============================================================================
# Program    : Student Result Calculator
# Objective  : Practice and master student result calculator logic.
# Concept    : Splitting logic into multiple focused functions — get, calculate, and display
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================

# ─────────────────────────────────────────────
# Function 1: Get student name and marks
# ─────────────────────────────────────────────

# What is used : Function definition 'def get_marks'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def get_marks():
    """Collects student name and marks for 5 subjects from the user"""
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
    name = input("Enter student name: ")
    print("\nEnter marks for 5 subjects (out of 100 each):")
    marks = []
    subjects = ["Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5"]

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
    for subject in subjects:
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        mark = float(input(f"  {subject}: "))
        marks.append(mark)
    return name, marks


# ─────────────────────────────────────────────
# Function 2: Calculate total marks
# ─────────────────────────────────────────────

# What is used : Function definition 'def calculate_total'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def calculate_total(marks):
    """Returns the sum of all subject marks"""
    return sum(marks)


# ─────────────────────────────────────────────
# Function 3: Calculate percentage
# ─────────────────────────────────────────────

# What is used : Function definition 'def calculate_percentage'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def calculate_percentage(total, max_marks=500):
    """Returns the percentage based on total marks out of max_marks"""
    return (total / max_marks) * 100


# ─────────────────────────────────────────────
# Function 4: Display the final result
# ─────────────────────────────────────────────

# What is used : Function definition 'def display_result'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def display_result(name, total, percentage):
    """Prints the formatted result card with grade and message"""

    # Determine grade based on percentage
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if percentage >= 90:
        grade = "A+"
        message = "Outstanding! Keep it up!"
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
    elif percentage >= 80:
        grade = "A"
        message = "Congratulations!"
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
    elif percentage >= 70:
        grade = "B"
        message = "Well done! Keep improving!"
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
    elif percentage >= 60:
        grade = "C"
        message = "Good effort! Study harder!"
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
    elif percentage >= 40:
        grade = "D"
        message = "You passed. Work on weak subjects."
    else:
        grade = "F"
        message = "Failed. Please try again."

    # Print result card
    print("\n------ RESULT ------")
    print(f"Name       : {name}")
    print(f"Total      : {int(total)}")
    print(f"Percentage : {percentage:.0f}%")
    print(f"Grade      : {grade}")
    print(f"\n{message}")
    print("--------------------")


# ─────────────────────────────────────────────
# Main Program — calling all functions in order
# ─────────────────────────────────────────────
name, marks = get_marks()
total = calculate_total(marks)
percentage = calculate_percentage(total)
display_result(name, total, percentage)
