# ==============================================================================
# Program    : Find intersection of two sets
# Objective  : Practice and master find intersection of two sets logic.
# Concept    : Intersection returns common elements using intersection() or & operator
# Why Used   : Demonstrates fundamental Python programming logic and best practices.
# ==============================================================================

devs_frontend = {"HTML", "CSS", "JavaScript", "Python"}
devs_backend = {"Python", "Java", "C++", "JavaScript"}

print("Frontend Stack:", devs_frontend)
print("Backend Stack:", devs_backend)

# Method 1: Using intersection() method
fullstack_skills = devs_frontend.intersection(devs_backend)
print("Common Skills (using .intersection()):", fullstack_skills)

# Method 2: Using & operator
fullstack_skills_op = devs_frontend & devs_backend
print("Common Skills (using & operator):", fullstack_skills_op)
