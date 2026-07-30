# ==============================================================================
# Program    : Birthday Countdown & Age Calculator
# Objective  : Calculate days until next birthday, age in years, and total days lived.
# Concept    : Date Math & Parsing (datetime.strptime, date math)
# Why Used   : datetime.strptime parses input date strings; subtracting dates yields timedelta object.
# ==============================================================================

# What is used : from datetime import datetime, date
# Why it is used: Provides datetime parsing and date arithmetic capabilities
from datetime import datetime, date

def calculate_birthday_stats(birth_date_str):
    # What is used : datetime.strptime(str, format)
    # Why it is used: Parses input string "DD-MM-YYYY" into a structured datetime object
    # How it works : Matches %d (day), %m (month), %Y (four-digit year)
    birth_date = datetime.strptime(birth_date_str, "%d-%m-%Y").date()
    today = date.today()

    if birth_date > today:
        raise ValueError("Birth date cannot be in the future!")

    # Calculate Total Days Lived
    # Subtracting two date objects returns a timedelta object with a .days attribute
    total_days_lived = (today - birth_date).days

    # Calculate Age in Years
    age_years = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

    # Calculate Next Birthday
    this_year_bday = date(today.year, birth_date.month, birth_date.day)
    if this_year_bday < today:
        next_bday = date(today.year + 1, birth_date.month, birth_date.day)
    else:
        next_bday = this_year_bday

    days_until_next_bday = (next_bday - today).days

    return age_years, total_days_lived, days_until_next_bday

def main():
    print("=== Birthday Countdown & Age Calculator ===")
    try:
        bday_input = input("Enter Birthday (DD-MM-YYYY, e.g. 19-06-2005): ").strip()
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
