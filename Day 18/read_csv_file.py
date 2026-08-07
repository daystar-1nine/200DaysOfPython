# ==============================================================================
# Program    : Read CSV Records Using csv.reader and csv.DictReader
# Objective  : Parse comma-separated tabular data from CSV files.
# Concept    : CSV File Reading (csv.reader & csv.DictReader)
# Why Used   : csv.DictReader maps row values to column headers as dictionary key-value pairs.
# ==============================================================================

import csv
import os

csv_file = "temp_students.csv"

# Helper: Create sample CSV file
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age", "City"])
    writer.writerow(["Suraj", "20", "Mumbai"])
    writer.writerow(["Rahul", "21", "Pune"])
    writer.writerow(["Priya", "22", "Delhi"])

print(f"Created sample CSV '{csv_file}'.\n")

# What is used : csv.DictReader(file)
# Why it is used: Parses rows as dictionary objects using first row headers as keys
print("--- Reading CSV via csv.DictReader ---")
with open(csv_file, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(f"Student: {row['Name']:<8} | Age: {row['Age']:<3} | City: {row['City']}")

# Cleanup
if os.path.exists(csv_file):
    os.remove(csv_file)
