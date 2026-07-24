# Program: Student Result Calculator
# Concept: Splitting logic into multiple focused functions — get, calculate, and display

# ─────────────────────────────────────────────
# Function 1: Get student name and marks
# ─────────────────────────────────────────────
def get_marks():
    """Collects student name and marks for 5 subjects from the user"""
    name = input("Enter student name: ")
    print("\nEnter marks for 5 subjects (out of 100 each):")
    marks = []
    subjects = ["Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5"]
    for subject in subjects:
        mark = float(input(f"  {subject}: "))
        marks.append(mark)
    return name, marks


# ─────────────────────────────────────────────
# Function 2: Calculate total marks
# ─────────────────────────────────────────────
def calculate_total(marks):
    """Returns the sum of all subject marks"""
    return sum(marks)


# ─────────────────────────────────────────────
# Function 3: Calculate percentage
# ─────────────────────────────────────────────
def calculate_percentage(total, max_marks=500):
    """Returns the percentage based on total marks out of max_marks"""
    return (total / max_marks) * 100


# ─────────────────────────────────────────────
# Function 4: Display the final result
# ─────────────────────────────────────────────
def display_result(name, total, percentage):
    """Prints the formatted result card with grade and message"""

    # Determine grade based on percentage
    if percentage >= 90:
        grade = "A+"
        message = "Outstanding! Keep it up!"
    elif percentage >= 80:
        grade = "A"
        message = "Congratulations!"
    elif percentage >= 70:
        grade = "B"
        message = "Well done! Keep improving!"
    elif percentage >= 60:
        grade = "C"
        message = "Good effort! Study harder!"
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
