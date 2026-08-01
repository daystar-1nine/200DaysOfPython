# ==============================================================================
# Program    : Student Marks Analyzer (Mini Project)
# Objective  : Process student marks dataset functionally using filter, map, and reduce.
# Concept    : Functional Data Pipelines (filter, map, reduce, max)
# Why Used   : Filters passing students, adds bonus marks, calculates average, max, and sum.
# ==============================================================================

from functools import reduce

marks = [78, 92, 65, 88, 95, 71, 55, 48]

def main():
    print("=== STUDENT MARKS ANALYZER ===")
    print("Raw Marks Dataset:", marks)

    # 1. Filter passed students (marks >= 70)
    # What is used : filter() with lambda
    passed_students = list(filter(lambda score: score >= 70, marks))
    print(f"\n1. Passed Students (Score >= 70) : {passed_students}")

    # 2. Add 5 bonus marks to passed students (capped at 100)
    # What is used : map() with lambda and min() capping
    bonus_marks = list(map(lambda score: min(100, score + 5), passed_students))
    print(f"2. Passed Marks after 5 Bonus Points: {bonus_marks}")

    # 3. Calculate total marks of all students
    # What is used : reduce() accumulator
    total_marks = reduce(lambda acc, score: acc + score, marks)
    print(f"3. Total Marks Sum (reduce)       : {total_marks}")

    # 4. Calculate Average & Highest Marks
    average_marks = total_marks / len(marks)
    highest_marks = max(marks)

    print(f"4. Class Average Marks            : {average_marks:.2f}")
    print(f"5. Highest Mark Achieved           : {highest_marks}")

if __name__ == "__main__":
    main()
