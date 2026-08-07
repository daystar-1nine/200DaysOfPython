# ==============================================================================
# Program    : Write CSV Records Using csv.writer and csv.DictWriter
# Objective  : Export structured tabular records to a CSV file on disk.
# Concept    : CSV File Writing (csv.writer, newline="")
# Why Used   : newline="" prevents blank line gaps on Windows systems when writing rows.
# ==============================================================================

import csv
import os

csv_file = "output_employees.csv"

# What is used : Header list and row data
fieldnames = ["EmpID", "Name", "Department", "Salary"]
employees = [
    {"EmpID": "E101", "Name": "Suraj Sawant", "Department": "Engineering", "Salary": 85000},
    {"EmpID": "E102", "Name": "Rahul Sharma", "Department": "Marketing", "Salary": 65000},
    {"EmpID": "E103", "Name": "Priya Patel", "Department": "Finance", "Salary": 75000}
]

# What is used : csv.DictWriter with writeheader() and writerows()
# Why it is used: Writes column names automatically and maps dict keys to CSV columns
with open(csv_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(employees)

print(f"Successfully wrote {len(employees)} records to '{csv_file}'!")

# Verification
with open(csv_file, "r", encoding="utf-8") as file:
    print("\nCSV Content Preview:\n", file.read())

# Cleanup
if os.path.exists(csv_file):
    os.remove(csv_file)
