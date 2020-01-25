import numpy as np
import pandas as pd
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules


class Apriori:
    def __init__(self, engine):
        self.engine = engine

    def switch_engine(self, engine):
        self.engine = engine


class AprioriMlxtend:
    def __init__(self, min_supp, metric, threshold):
        self.min_supp = min_supp
        self.metric = metric
        self.threshold = threshold


class AprioriInterpreted:
    def __init__(self, min_supp, metric, threshold):
        self.min_supp = min_supp
        self.metric = metric
        self.threshold = threshold

# TODO: add compiled cython apriori engine, numpy-enhanced apriori interpreted engine,
# maybe multithreaded or CUDA accelerated engines too?
