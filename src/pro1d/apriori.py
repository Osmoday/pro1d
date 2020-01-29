from dataclasses import dataclass
import numpy as np
import pandas as pd
import time
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


class _AprioriMlxtend:
    def __init__(self):
        pass

    @staticmethod
    def do_apriori(data, min_supp):
        t1 = time.time()
        frequent_itemsets = apriori(data, min_support=min_supp, use_colnames=True)
        t2 = time.time()
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        return Result(t2 - t1, frequent_itemsets)

    @staticmethod
    def find_assoc_rules(frequent_itemsets, metric, threshold):
        t1 = time.time()
        assoc = association_rules(frequent_itemsets, metric=metric,
                                  min_threshold=threshold)
        t2 = time.time()
        return Result(t2 - t1, assoc)


class _AprioriInterpreted:
    def __init__(self):
        pass

    @staticmethod
    def do_apriori(data, min_supp, metric, threshold):
        pass

    @staticmethod
    def find_assoc_rules(frequent_itemsets, metric, threshold):
        pass


class _AprioriNumpy:
    def __init__(self):
        pass

    @staticmethod
    def do_apriori(data, min_supp, metric, threshold):
        pass

    @staticmethod
    def find_assoc_rules(frequent_itemsets, metric, threshold):
        pass


@dataclass
class Result:
    exec_t: float
    dataframe: pd.DataFrame


_engines = {'mlxtend': _AprioriMlxtend, 'vanilla_python': _AprioriInterpreted, 'numpy': _AprioriNumpy}


def engine(provided_engine):
    engine.engine = provided_engine


def do_apriori(data, min_supp, metric, threshold):
    return _engines[engine.engine].do_apriori(data, min_supp, metric, threshold)

# TODO: add compiled cython apriori engine, numpy-enhanced apriori interpreted engine,
# maybe multithreaded or CUDA accelerated engines too?
# TODO: create a predictor of minimum reasonable support
