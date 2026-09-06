"""
Task 7 — Conditional Probability
Calculate P(SQL | Python) from a cohort of 100 students.
"""

# Cohort specification
total_students = 100
python_students = 60
sql_students = 40
both_students = 25

# Conditional Probability: P(SQL | Python) = P(SQL and Python) / P(Python)
# or using counts: n(SQL and Python) / n(Python)
p_python = python_students / total_students
p_both = both_students / total_students
p_sql_given_python = p_both / p_python

print("=== TASK 7: CONDITIONAL PROBABILITY ===")
print(f"Total Students:                   {total_students}")
print(f"Know Python:                      {python_students} (P(Python) = {p_python:.2f})")
print(f"Know SQL:                         {sql_students} (P(SQL) = {sql_students/total_students:.2f})")
print(f"Know Both (Python AND SQL):       {both_students} (P(Python AND SQL) = {p_both:.2f})")
print("-" * 50)
print(f"P(SQL | Python) = P(SQL and Python) / P(Python)")
print(f"               = {both_students} / {python_students}")
print(f"               = {p_sql_given_python:.4f} ({p_sql_given_python * 100:.2f}%)")
print("-" * 50)
print(f"Notice: P(SQL | Python) = 41.67% > P(SQL) = 40.0%, indicating slight positive dependence.")
