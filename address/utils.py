"""
TO DO ...
"""
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
    s = unidecode(s)
    s = s.upper()
    s = s.replace('-', ' ')
    s = re.sub(r'[^A-Z0-9 ]', '', s, flags=re.ASCII)
    #s = s.strip()
    s = re.sub(r'\s+', ' ', s)

    check = re.search(r'[^A-Z0-9 ]', s)

    if check is None:
        return s
    else:
        raise RuntimeError(f"The string '{s:s}' could not be normalized.")
