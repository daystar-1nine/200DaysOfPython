"""Text normalization utilities."""
import re
import string

def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punct: bool = True,
    remove_urls: bool = True,
    remove_numbers: bool = False,
    strip_whitespace: bool = True
) -> str:
    """Normalizes input text according to specified pipeline parameters."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
        
    if lowercase:
        text = text.lower()
        
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        
    if remove_numbers:
        text = re.sub(r'\d+', ' ', text)
        
    if remove_punct:
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(translator)
        
    if strip_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
        
    return text
