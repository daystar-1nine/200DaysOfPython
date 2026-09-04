"""
Day 60 - NumPy Practical Test: Student Marks Analytics Matrix
Performs all 10 matrix analytics operations on 2D marks array using NumPy.
"""

# What is used: Import sys and numpy libraries.
# Why it is used: Cross-platform stdout encoding and high-performance numerical array operations.
# How it works: Brings sys and numpy namespaces into execution scope.
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_numpy_practical_test() -> dict:
    """
    Execute all 10 NumPy practical test requirements.

    Returns:
        dict: Results dictionary containing all 10 analytical outputs.
    """
    # What is used: 2D NumPy array creation.
    # Why it is used: Represents 5 students across 4 subjects.
    # How it works: Constructs 5x4 ndarray of integers.
    marks = np.array([
        [85, 90, 78, 92],
        [70, 75, 80, 68],
        [95, 88, 91, 94],
        [60, 65, 58, 70],
        [82, 79, 85, 88]
    ])

    # 1. Student averages (axis=1)
    student_averages = np.mean(marks, axis=1).round(2)

    # 2. Subject averages (axis=0)
    subject_averages = np.mean(marks, axis=0).round(2)

    # 3. Highest mark
    highest_mark = int(np.max(marks))

    # 4. Lowest mark
    lowest_mark = int(np.min(marks))

    # 5. Highest-scoring student (index and student ID)
    top_student_idx = int(np.argmax(student_averages))

    # 6. Students with average > 80
    high_achievers_mask = student_averages > 80
    high_achievers_indices = np.where(high_achievers_mask)[0]

    # 7. Number of individual marks >= 90
    marks_ge_90_count = int(np.sum(marks >= 90))

    # 8. Standard deviation across all marks
    std_deviation = round(float(np.std(marks)), 2)

    # 9. Rank students by average descending (1-indexed rank)
    sorted_order = np.argsort(-student_averages)
    ranks = np.empty_like(sorted_order)
    ranks[sorted_order] = np.arange(1, len(student_averages) + 1)

    # 10. Grade assignment using np.select
    condlist = [
        student_averages >= 85,
        student_averages >= 70,
        student_averages >= 50
    ]
    choicelist = ["A", "B", "C"]
    grades = np.select(condlist, choicelist, default="F")

    return {
        "student_averages": student_averages.tolist(),
        "subject_averages": subject_averages.tolist(),
        "highest_mark": highest_mark,
        "lowest_mark": lowest_mark,
        "top_student_idx": top_student_idx,
        "students_above_80_idx": high_achievers_indices.tolist(),
        "marks_ge_90_count": marks_ge_90_count,
        "std_deviation": std_deviation,
        "student_ranks": ranks.tolist(),
        "assigned_grades": grades.tolist()
    }


def main() -> None:
    res = run_numpy_practical_test()

    print("==================================================")
    print("             NUMPY PRACTICAL EXAM RESULTS         ")
    print("==================================================")
    print(f"1. Student Averages (axis=1)     : {res['student_averages']}")
    print(f"2. Subject Averages (axis=0)     : {res['subject_averages']}")
    print(f"3. Highest Mark Overall          : {res['highest_mark']}")
    print(f"4. Lowest Mark Overall           : {res['lowest_mark']}")
    print(f"5. Highest-Scoring Student Index : Student {res['top_student_idx'] + 1} (Score: {res['student_averages'][res['top_student_idx']]})")
    print(f"6. Students with Average > 80    : Indices {res['students_above_80_idx']}")
    print(f"7. Total Marks >= 90 Count       : {res['marks_ge_90_count']}")
    print(f"8. Overall Standard Deviation    : {res['std_deviation']}")
    print(f"9. Student Ranks (Descending)    : {res['student_ranks']}")
    print(f"10. Assigned Grades (A/B/C/F)    : {res['assigned_grades']}")


if __name__ == "__main__":
    main()
