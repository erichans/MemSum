# From https://github.com/AIPHES/emnlp19-moverscore/

from itertools import zip_longest
from typing import List, Union, Iterable
from collections import defaultdict
from moverscore_v2 import get_idf_dict, word_mover_score
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

def compute(predictions, references):
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    return np.mean(word_mover_score(references, predictions, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True))
