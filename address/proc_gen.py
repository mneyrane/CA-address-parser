"""
TO DO ...
"""
from abc import ABC, abstractmethod
import enum
import re
import string

import numpy as np
from unidecode import unidecode

from address.utils import get_digits


@enum.verify(enum.UNIQUE)
@enum.verify(enum.CONTINUOUS)
class TokenCategory(enum.Enum):
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


class TokenGenerator(ABC):
    def __init__(self, name, rng, **data):
        self.name = name
        self.rng = rng
        self.data = data
        
    @abstractmethod
    def sample(self):
        pass
        
        
_vec_get_digits = np.vectorize(get_digits)

class AlphanumGenerator(TokenGenerator):
    """
    ...
    
    Suitable for units and house numbers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._num_range = np.arange(kwargs['nmin'], kwargs['nmax'])
        d_arr = _vec_get_digits(self._num_range)
        d_u, d_indices, d_count = np.unique(d_arr, return_inverse=True, return_counts=True)
        # numbers grouped by number of digits occur with equal probability
        self._num_range_probs = 1 / (len(d_u) * d_count[d_indices])
        
        self._alpha_range = list(string.ascii_uppercase)
        
    def sample(self):
        templates = ('N', 'NA', 'A')
        #probs = (.9, .05, .05)
        #tp = self.rng.choice(templates, p=probs)
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
    """
    Generator for tokens stored in a list or array-like.
    
    Suitable for street names, street types, directions, cities and provinces.
    """
    def sample(self):
        return self.rng.choice(self.data['values'])
        

class PostalCodeGenerator(TokenGenerator):
    """
    TO DO ...
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.letters = list(string.ascii_uppercase)
        self.digits = list(string.digits)
        
    def sample(self):
        postcode = []
        
        for i in range(6): # 6 characters; postal code is ADADAD (A : alphabet, D : digit) 
            if i % 2 == 0:
                postcode.append(self.rng.choice(self.letters))
            else:
                postcode.append(self.rng.choice(self.digits))
                
        x = self.rng.random()
        if x < self.data['sep_prob']:
            postcode.insert(3, ' ')
        
        return ''.join(postcode)


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

def sample_address_tokens(generator_map, template):
    tokens = {}
    
    for t_id in template:
        tokens[t_id] = generator_map[t_id].sample()
    
    return tokens

def join_address_tokens(tokens, template):
    """
    `template` should be a sequence of TokenCategory enums.
    """  
    address = []
    clf = []
    
    first = True
    for v in template:        
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
    
    return address, clf
