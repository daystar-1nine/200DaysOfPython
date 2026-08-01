# ==============================================================================
# Program    : Resume Information Extractor (Challenge Project)
# Objective  : Parse candidate resume text and extract Name, Email, Phone, Skills, and Experience using regex.
# Concept    : Complex Multi-Pattern Regex Parsing
# Why Used   : Extracts structured candidate metadata from unstructured text documents.
# ==============================================================================

import re

# Sample candidate resume text
resume_text = """
SURAJ SAWANT
Senior Software Engineer
Email: suraj.sawant19@gmail.com | Phone: 9876543210
Location: Mumbai, India

SUMMARY:
Passionate Software Engineer with 5 years of professional experience building scalable web apps and AI agents.

TECHNICAL SKILLS:
Programming: Python, JavaScript, C++, SQL
Frameworks & Libraries: Django, React, PyTorch, TensorFlow, Pandas
Tools: Git, Docker, AWS, Linux

EXPERIENCE:
Lead Developer at Tech Corp (2021 - Present)
- Developed enterprise Python applications.
"""

def extract_resume_details(text):
    # What is used : Regex patterns for each candidate field
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    phone_match = re.search(r"\b[6-9]\d{9}\b", text)
    exp_match = re.search(r"(\d+)\s*(?:years?|yrs?)\s*(?:of)?\s*experience", text, re.IGNORECASE)
    
    # Extract candidate name (First line of resume)
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    name = lines[0] if lines else "Not Found"

    # Extract skills section using regex search
    skills_search = re.search(r"TECHNICAL SKILLS:[\s\S]*?(?=\n\n|\n[A-Z]+:|$)", text)
    skills = "Not Found"
    if skills_search:
        skills = re.sub(r"TECHNICAL SKILLS:\s*", "", skills_search.group(0)).strip()

    return {
        "Name": name,
        "Email": email_match.group(0) if email_match else "Not Found",
        "Phone": phone_match.group(0) if phone_match else "Not Found",
        "Experience": f"{exp_match.group(1)} Years" if exp_match else "Not Found",
        "Skills": skills
    }

def main():
    print("==========================================================")
    print("             RESUME INFORMATION EXTRACTOR                 ")
    print("==========================================================")

    candidate_info = extract_resume_details(resume_text)

    for key, val in candidate_info.items():
        print(f"{key:<15}: {val}")
    print("==========================================================\n")

if __name__ == "__main__":
    main()
