"""
TO DO ...
"""
from copy import deepcopy

_EN_US_KBD_NEIGHBOURS = {
    'A' : ['Q','W','S','Z'],
    'B' : ['G','H','V','N'],
    'C' : ['D','F','X','V'],
    'D' : ['E','R','S','F','X','C'],
    'E' : ['3','4','W','R','S','D'],
    'F' : ['R','T','D','G','C','V'],
    'G' : ['T','Y','F','H','V','B'],
    'H' : ['Y','U','G','J','B','N'],
    'I' : ['8','9','U','O','J','K'],
    'J' : ['U','I','H','K','N','M'],
    'K' : ['I','O','J','L','M'],
    'L' : ['O','P','K'],
    'M' : ['J','K','N'],
    'N' : ['H','J','B','M'],
    'O' : ['9','0','I','P','K','L'],
    'P' : ['0','O','L'],
    'Q' : ['1','2','W','A'],
    'R' : ['4','5','E','T','D','F'],
    'S' : ['W','E','A','D','Z','X'],
    'T' : ['5','6','R','Y','F','G'],
    'U' : ['7','8','Y','I','H','J'],
    'V' : ['F','G','C','B'],
    'W' : ['2','3','Q','E','A','S'],
    'X' : ['S','D','Z','C'],
    'Y' : ['6','7','T','U','G','H'],
    'Z' : ['A','S','X'],
    '0' : ['9','O','P'],
    '1' : ['2','Q'],
    '2' : ['1','3','Q','W'],
    '3' : ['2','4','W','E'],
    '4' : ['3','5','E','R'],
    '5' : ['4','6','R','T'],
    '6' : ['5','7','T','Y'],
    '7' : ['6','8','Y','U'],
    '8' : ['7','9','U','I'],
    '9' : ['8','0','I','O'],
}

def _get_alphabet_indices(chars):
    return [i for i, c in enumerate(chars) if c != ' ']

def delete(chars, clf, rng):
    i = rng.choice(len(chars))
    chars.pop(i)
    clf.pop(i)
    
def replace(chars, clf, rng):
    alpha_idxs = _get_alphabet_indices(chars)
    i = rng.choice(alpha_idxs)
    c = rng.choice(_EN_US_KBD_NEIGHBOURS[chars[i]])
    chars[i] = c
    
def duplicate(chars, clf, rng):
    alpha_idxs = _get_alphabet_indices(chars)
    i = rng.choice(alpha_idxs)
    c, v = chars[i], clf[i]
    chars.insert(i, c)
    clf.insert(i, v)
    
def swaps(chars, clf, rng):
    while True:
        i = rng.choice(len(chars)-1)
        if (i == 0 and chars[i+1] == ' ') or (i == len(chars)-2 and chars[i] == ' '):
            continue
        else:
            chars[i], chars[i+1] = chars[i+1], chars[i]
            clf[i], clf[i+1] = clf[i+1], clf[i]
            break
