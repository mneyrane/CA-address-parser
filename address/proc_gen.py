"""Procedural generation of Canadian addresses.

Based on Canada Post's addressing guidelines and a very common English civic
address format used throughout Canada and the United States.
"""
from abc import ABC, abstractmethod
import enum
import string

import numpy as np

from address.utils import get_digits


@enum.verify(enum.UNIQUE)
@enum.verify(enum.CONTINUOUS)
class TokenCategory(enum.Enum):
    # Character-level address token classification.
    SEPARATOR = 0
    UNIT = 1
    HOUSE_NUM = 2
    ST_NAME = 3
    ST_TYPE = 4
    DIR_PREFIX = 5
    DIR_SUFFIX = 6
    CITY = 7
    PROVINCE = 8
    POST_CODE = 9

# mapping of template entries to TokenCategory enum
map_token_id_to_enum = {
    'un' : TokenCategory.UNIT,
    'hn' : TokenCategory.HOUSE_NUM,
    'sn' : TokenCategory.ST_NAME,
    'st' : TokenCategory.ST_TYPE,
    'dp' : TokenCategory.DIR_PREFIX,
    'ds' : TokenCategory.DIR_SUFFIX,
    'ci' : TokenCategory.CITY,
    'pr' : TokenCategory.PROVINCE,
    'po' : TokenCategory.POST_CODE,
}

class TokenGenerator(ABC):
    """Abstract base class for generating tokens.
    
    Attributes:
        name (str) : Token generator identifer.
        rng (np.random.Generator) : NumPy random number generator.
        data (dict) : Data used to generate tokens.
    """
    def __init__(self, name, rng, **data):
        self.name = name
        self.rng = rng
        self.data = data
        
    @abstractmethod
    def sample(self):
        pass
        
        
class AlphanumGenerator(TokenGenerator):
    """Token generator for units and house numbers.
    
    The units and house numbers can be (or contain as a suffix,) alphabet
    characters, not just numbers. The numbers themselves are generated in 
    a way that numbers differing in number of digits will occur with equal
    probability.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        _vec_get_digits = np.vectorize(get_digits)
        
        self._num_range = np.arange(kwargs['nmin'], kwargs['nmax'])
        d_arr = _vec_get_digits(self._num_range)
        d_u, d_indices, d_count = np.unique(d_arr, return_inverse=True, return_counts=True)
        # numbers grouped by number of digits occur with equal probability
        self._num_range_probs = 1 / (len(d_u) * d_count[d_indices])
        
        self._alpha_range = list(string.ascii_uppercase)
        
    def sample(self):
        templates = ('N', 'NA', 'A')
        tp = self.rng.choice(templates)
        
        num, alpha, desig = '', '', ''
        
        if 'N' in tp:
            num = self.rng.choice(self._num_range, p=self._num_range_probs)
            num = str(num)
            
        if 'A' in tp:
            alpha = self.rng.choice(self._alpha_range)
        
        if 'desig' in self.data:
            desig = self.rng.choice(self.data['desig'])
        
        if tp == 'N':
            subtokens = [desig, num]
        elif tp == 'NA':
            subtokens = [desig, ''.join([num, alpha])]
        else: # tp == 'A'
            subtokens = [desig, alpha]
        
        return ' '.join([t for t in subtokens if len(t) > 0])
        

class UniformListSampler(TokenGenerator):
    """Token generator from strings stored in a list or array-like.
    
    Used for street names, street types, directions, cities and provinces.
    """
    def sample(self):
        return self.rng.choice(self.data['values'])
        

class PostalCodeGenerator(TokenGenerator):
    """Token generator for postal codes.
    
    Postal codes are of the form ADADAD (or ADA DAD), where A is a letter 
    and D is a digit.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._letters = list(string.ascii_uppercase)
        self._digits = list(string.digits)
        
    def sample(self):
        postcode = []
        
        for i in range(6): # six characters; ADADAD
            if i % 2 == 0:
                postcode.append(self.rng.choice(self._letters))
            else:
                postcode.append(self.rng.choice(self._digits))
                
        x = self.rng.random()
        if x < self.data['sep_prob']:
            postcode.insert(3, ' ')
        
        return ''.join(postcode)


def sample_address_tokens(generator_map, template):
    """Generate address tokens from generators.
    
    Args:
        generator_map (list) : TokenGenerator subclasses.
        template (list) : Token IDs (see keys of `map_token_id_to_enum`).
            
    Returns:
        dict: Address tokens.
    """
    tokens = {}
    
    for t_id in template:
        tokens[t_id] = generator_map[t_id].sample()
    
    return tokens

def join_address_tokens(tokens, template, strict=True):
    """Create address line and per-character classification.
    
    Args:
        tokens (dict): Address tokens.
        template (list): Token IDs (see keys of `map_token_id_to_enum`).
        strict (bool): Raise an exception if a template entry does not
            appear in `tokens`.
    
    Returns:
        list: Address line characters.
        list: Address character token classification.
        
    Raises:
        ValueError: Template entry is not a key in `tokens`.
    """  
    address = []
    clf = []
    
    first = True
    for v in template:
        if v not in tokens:
            if strict:
                raise ValueError(f"Template entry {v} is not in 'tokens'.")
            else:
                continue
            
        if not first:
            address.append(' ')
            clf.append(TokenCategory.SEPARATOR)
            
        t = tokens[v]
        address.extend(list(t))
        clf.extend(len(t)*[map_token_id_to_enum[v]])
        
        first = False
        
    for i, c in enumerate(address):
        if c == ' ':
            clf[i] = TokenCategory.SEPARATOR

    if not strict and address[-1] == ' ':
        address.pop()
        clf.pop()
    
    return address, clf
