# ==============================================================================
# Program    : Student Grade and Result Calculator
# Objective  : Calculate total marks, percentage, and average for 5 subjects.
# Why Used   : Demonstrates accumulation, arithmetic operations, and report card formatting.
# ==============================================================================

print("----- Student Grade Calculator -----")

# Step 1: Input student identification details
name = input("Enter Student Name: ")
roll_number = input("Enter Roll Number: ")

# Step 2: Input marks for 5 subjects (float allows decimal marks)
subject1 = float(input("Enter marks for Subject 1: "))
subject2 = float(input("Enter marks for Subject 2: "))
subject3 = float(input("Enter marks for Subject 3: "))
subject4 = float(input("Enter marks for Subject 4: "))
subject5 = float(input("Enter marks for Subject 5: "))

# Step 3: Calculate total marks obtained (out of 500)
total = subject1 + subject2 + subject3 + subject4 + subject5

# Step 4: Calculate percentage and average
percentage = (total / 500) * 100
average = total / 5

# Step 5: Display Student Report Card
print("\n====================================")
print("          STUDENT REPORT CARD        ")
print("====================================")
print("Name:", name)
print("Roll Number:", roll_number)
print("Total Marks Obtained:", total, "/ 500")
print("Percentage:", round(percentage, 2), "%")
print("Average Marks:", round(average, 2))
print("====================================")
