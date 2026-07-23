# Program: Student Grade and Result Calculator
# Concept: String and Float Inputs, Accumulation, Percentage Calculation, and Output Formatting

print("----- Student Grade Calculator -----")

# Step 1: Input student details
name = input("Enter Student Name: ")
roll_number = input("Enter Roll Number: ")

# Step 2: Input marks for 5 subjects (out of 100 each)
subject1 = float(input("Enter marks for Subject 1: "))
subject2 = float(input("Enter marks for Subject 2: "))
subject3 = float(input("Enter marks for Subject 3: "))
subject4 = float(input("Enter marks for Subject 4: "))
subject5 = float(input("Enter marks for Subject 5: "))

# Step 3: Calculate Total Marks (Maximum 500)
total = subject1 + subject2 + subject3 + subject4 + subject5

# Step 4: Calculate Percentage and Average
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