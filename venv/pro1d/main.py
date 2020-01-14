import numpy as np
import pandas as pd
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

with open("../data/retail.dat.txt", "r") as f:
    data = f.readlines()
data = [line.strip().split(" ") for line in data]

te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)
print("Dataset loaded and encoded")
print(df)

apriori_time_start = time.time()
frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
apriori_time = time.time() - apriori_time_start
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
# frequent_itemsets = frequent_itemsets.drop()

print("Apriori frequent itemsets")
print(frequent_itemsets)
print("Apriori execution time: ", apriori_time, " seconds")
print("")

association_apriori_time_start = time.time()
association_rules_apriori = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05)
association_apriori_time = time.time() - association_apriori_time_start

print("Apriori frequent itemsets: association rules found:")
print(association_rules_apriori)
print("Apriori association rule mining execution time: ", association_apriori_time, " seconds")
print("")

fpg_time_start = time.time()
fpg = fpgrowth(df, min_support=0.05, use_colnames=True)
fpg_time = time.time() - fpg_time_start

print("FP-growth")
print(fpg)
print("FP-growth execution time:", fpg_time, " seconds")
print("")

association_fpg_time_start = time.time()
association_rules_fpg = association_rules(fpg, metric="confidence", min_threshold=0.05)
association_fpg_time = time.time() - association_fpg_time_start

print("FP-growth frequent itemsets: association rules found:")
print(association_rules_fpg)
print("FP-growth association rule mining execution time: ", association_fpg_time, " seconds")
print("")

inp = input("Save results as .csv files? [Y/N]")
if inp.lower() == "y" or inp.lower() == "yes":
    filename_apriori = input("Please name the file for the apriori results (without file format extension) ")
    frequent_itemsets.to_csv(filename_apriori + ".csv")

    filename_a_association = input("Please name the file for the apriori association rules (without file format "
                                   "extension) ")
    association_rules_apriori.to_csv(filename_a_association + ".csv")

    filename_fpg = input("Please name the file for the FP-growth results (without file format extension) ")
    fpg.to_csv(filename_fpg + ".csv")

    filename_dpg_association = input("Please name the file for the FP=growth association rules (without file format "
                                     "extension) ")
    association_rules_fpg.to_csv(filename_dpg_association + ".csv")

inp = input("Save encoded dataframe as a .csv file? [Y/N] WARNING: Large file size measured in gigabytes, saving it "
            "will take a very long time. ")
if inp.lower() == "y" or inp.lower() == "yes":
    filename_dataframe = input("Please name the file for the dataframe (without file format extension)")
    df.to_csv(filename_dataframe + ".csv")

