import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

with open("../data/retail.dat.txt", "r") as f:
    data = f.readlines()
data = [line.strip().split(" ") for line in data]

te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)

frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
# frequent_itemsets = frequent_itemsets.drop()

print(frequent_itemsets)

print(association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05))

# df.to_csv('readout.csv')
