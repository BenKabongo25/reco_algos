# Ben Kabongo
# December 2024

import numpy as np
import random
import re
import string
import torch
import unicodedata


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def normalize(s):
    return re.sub(' +', ' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters + " " + string.punctuation)).strip()