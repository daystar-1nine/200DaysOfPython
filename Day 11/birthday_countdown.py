# ==============================================================================
# Program    : Birthday Countdown & Age Calculator
# Objective  : Calculate days until next birthday, age in years, and total days lived.
# Concept    : Date Parsing, Date Arithmetic & Timedelta Math (datetime module)
# Why Used   : datetime.strptime parses input date strings; date subtraction yields timedelta object.
# ==============================================================================

# What is used : from datetime import datetime, date
# Why it is used: Provides date parsing (strptime) and date arithmetic capabilities
from datetime import datetime, date

def calculate_birthday_stats(birth_date_str):
    # What is used : datetime.strptime(date_str, format).date()
    # Why it is used: Parses input text string "DD-MM-YYYY" into a date object
    # How it works : Matches %d (day), %m (month), %Y (4-digit year) directives
    birth_date = datetime.strptime(birth_date_str, "%d-%m-%Y").date()

    # What is used : date.today()
    # Why it is used: Fetches current system calendar date
    today = date.today()

    # What is used : Comparison operator (>) for future date validation
    if birth_date > today:
        raise ValueError("Birth date cannot be in the future!")

    # What is used : Date subtraction (today - birth_date)
    # Why it is used: Subtracting two date objects calculates a timedelta object
    # How it works : Accessing .days attribute extracts total elapsed days count
    total_days_lived = (today - birth_date).days

    # What is used : Age calculation tuple comparison logic
    # How it works : Compares (today.month, today.day) < (birth_date.month, birth_date.day) to adjust for unreached birthday in current year
    age_years = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

    # What is used : Next birthday date calculation
    # How it works : Constructs date for current year birthday; if already passed, rolls forward to next year
    this_year_bday = date(today.year, birth_date.month, birth_date.day)
    if this_year_bday < today:
        next_bday = date(today.year + 1, birth_date.month, birth_date.day)
    else:
        next_bday = this_year_bday

    # What is used : Days remaining math (next_bday - today).days
    days_until_next_bday = (next_bday - today).days

    return age_years, total_days_lived, days_until_next_bday

def main():
    print("=== Birthday Countdown & Age Calculator ===")
    try:
        bday_input = input("Enter Birthday (DD-MM-YYYY, e.g. 19-06-2005): ").strip()

        # What is used : Function invocation with tuple unpacking
        age, total_days, days_until = calculate_birthday_stats(bday_input)

        print("\n---------------- RESULTS ----------------")
        print(f"Age in Years           : {age} years old")
        print(f"Total Days Lived       : {total_days:,} days")
        print(f"Days Until Next Birthday: {days_until} days")
        print("-----------------------------------------")

    except ValueError as e:
        print(f"Error: {e}. Please enter date in DD-MM-YYYY format!")

if __name__ == "__main__":
    main()
