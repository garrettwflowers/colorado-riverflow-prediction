import pandas as pd
import ast
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def process_chunk(chunk, min_support=0.1, min_threshold=0.5):
    # Convert the string representation of lists back into actual lists if needed
    chunk['transactions'] = chunk['cleaned_content'].apply(ast.literal_eval)

    # Prepare the transaction encoder
    te = TransactionEncoder()
    te_ary = te.fit(chunk['transactions']).transform(chunk['transactions'])
    df_chunk = pd.DataFrame(te_ary, columns=te.columns_)

    # Apply the Apriori algorithm to find frequent itemsets
    frequent_itemsets_chunk = apriori(df_chunk, min_support=min_support, use_colnames=True)

    # Generate association rules with a minimum confidence of min_threshold
    rules_chunk = association_rules(frequent_itemsets_chunk, metric="confidence", min_threshold=min_threshold)

    return frequent_itemsets_chunk, rules_chunk

def process_chunk_wrapper(args):
    return process_chunk(*args)