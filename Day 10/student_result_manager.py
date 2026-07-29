# ==============================================================================
# Program    : Student Result Manager
# Objective  : Validate student marks (0-100), handle invalid inputs, compute average & grade.
# Concept    : Full try-except-else-finally Lifecycle
# Why Used   : Demonstrates try, except, else (happy path), and finally (cleanup) in a real project.
# ==============================================================================

class OutOfBoundsMarkError(Exception):
    pass

def evaluate_student():
    marks = []
    subject_count = 3
    print("--- Student Result Manager ---")

    # What is used : try-except-else-finally complete block architecture
    try:
        name = input("Enter Student Name: ").strip()
        if not name:
            raise ValueError("Student name cannot be empty!")

        for i in range(1, subject_count + 1):
            mark_input = input(f"Enter marks for Subject {i} (0-100): ").strip()
            score = float(mark_input)
            if score < 0 or score > 100:
                raise OutOfBoundsMarkError(f"Mark Error: Score '{score}' is out of bounds! Must be between 0 and 100.")
            marks.append(score)

    except ValueError as ve:
        # What is used : Handling type parsing or empty input errors
        print(f"[Error Handler] Invalid Input: {ve}")

    except OutOfBoundsMarkError as oe:
        # What is used : Handling custom mark boundary validation exception
        print(f"[Error Handler] {oe}")

    else:
        # What is used : else block
        # Why it is used: Executes ONLY if no exceptions were raised in the try block
        # How it works : Computes total, average, and letter grade on valid inputs
        total = sum(marks)
        average = total / subject_count

        if average >= 90:
            grade = "A+"
        elif average >= 80:
            grade = "A"
        elif average >= 70:
            grade = "B"
        elif average >= 60:
            grade = "C"
        elif average >= 40:
            grade = "D"
        else:
            grade = "F (Fail)"

        print("\n====================================")
        print("         STUDENT RESULT CARD        ")
        print("====================================")
        print(f"Student Name    : {name}")
        print(f"Total Marks     : {total:.2f} / 300")
        print(f"Average Score   : {average:.2f}%")
        print(f"Final Grade     : {grade}")
        print("====================================")

    finally:
        # What is used : finally block
        # Why it is used: Always executes to finalize process logging regardless of pass/fail
        print("\n[Log] Student Evaluation Process Completed.")

if __name__ == "__main__":
    evaluate_student()
