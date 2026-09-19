"""Text normalization utilities."""
import re
import string

def normalize_text(text: str, remove_punctuation: bool = True, remove_numbers: bool = False) -> str:
    """
    Normalizes input text by:
    - handling non-string inputs safely
    - converting to lowercase
    - stripping extra whitespace
    - removing URLs and emails
    - handling punctuation and special characters
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
        
    text = text.lower().strip()
    
    # Replace URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Replace email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    if remove_numbers:
        text = re.sub(r'\d+', ' ', text)
        
    if remove_punctuation:
        # Replace punctuation with spaces to avoid concatenating words
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(translator)
        
    # Collapse multiple whitespaces into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
