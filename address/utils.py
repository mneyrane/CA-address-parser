"""Text and numerical processing utilities."""
import re
from unidecode import unidecode

def get_digits(num):
    x = 1
    d = 0
    
    while num >= x:
        x *= 10
        d += 1

    return d
    
def normalize_text(s):
    """Normalize text in a string.
    
    The expected output contains only uppercase letters, digits, or a space.
    Two consecutive spaces cannot occur, nor can spaces appear in the first
    or last character of the output.
    
    The text is transformed by (in order):
        - non-ASCII characters to their ASCII "equivalent"
        - convert to uppercase
        - replace dash with a space
        - delete characters that are not alphanumeric or a space
        - strip spaces at the beginning and end of the text
        - reduce consecutive spaces to a single space
        
    Raises:
        RuntimeError: Normalized string contains characters that are not
            uppercase letters, digits or a space.
    """
    s = unidecode(s)
    s = s.upper()
    s = re.sub(r'[^A-Z0-9 ]', ' ', s, flags=re.ASCII)
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)

    check = re.search(r'[^A-Z0-9 ]', s)

    if check is None:
        return s
    else:
        raise RuntimeError(f"The string '{s}' could not be normalized.")
        
def build_vocabulary(chars):
    """Build mapping of characters to non-negative integer indices."""
    vocab = {}
    n_tokens = 0
    
    for c in chars:
        if c not in vocab:
            vocab[c] = n_tokens
            n_tokens += 1
            
    return vocab
