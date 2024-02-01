import pandas as pd

df_train = pd.read_feather("/beegfs/scratch/user/rdeffaye/baidu-bert/features/train-features.feather", 
                     columns=["query_no", "query_md5", "url_md5", "text_md5", "position", "click"])
df_test = pd.read_feather("/beegfs/scratch/user/rdeffaye/baidu-bert/features/test-features.feather", 
                     columns=["query_no", "query_md5", "url_md5", "text_md5", "position", "click"])
df = pd.concat([df_train, df_test])

# Total sessions:
dfs_train = df_train.value_counts("query_no")
# In train set: 1,779,017
print("Total sessions in train set:", len(dfs_train))
dfs_test = df_test.value_counts("query_no")
# In val/test set: 593,930
print("Total sessions in val/test set:", len(dfs_test))
print("")

# Unique queries:
dfq_train = df_train.value_counts("query_md5")
# In train set: 1,378,901
print("Unique queries in train set:", len(dfq_train))
dfq_test = df_test.value_counts("query_md5")
# In val/test set: 501,215
print("Unique queries in val/test set:", len(dfq_test))
print("")

# Total impressions:
# In train set: 14,526,276
print("Total impressions in train set:", len(df_train))
# In val/test set: 4,848,878
print("Total impressions in val/test set:", len(df_test))
print("")

# Unique URLs:
dfd_train = df_train.value_counts("url_md5")
# In train set: 9,455,953
print("Unique URLs in train set:", len(dfd_train))
dfd_test = df_test.value_counts("url_md5")
# In val/test set: 3,557,825
print("Unique URLs in val/test set:", len(dfd_test))
print("")

# Total query/document pairs:
dfqd_train = df_train.groupby("query_md5", as_index=False)["url_md5"].value_counts()
# In train set: 11,715,447
print("Total query/document pairs in train set:", len(dfqd_train))
dfqd_test = df_test.groupby("query_md5", as_index=False)["url_md5"].value_counts()
# In val/test set: 4,209,900
print("Total query/document pairs in val/test set:", len(dfqd_test))
print("")

# Average number of docs per session: 8.165
dfs = df.value_counts("query_no")
print("Avg number of docs per session:", round(dfs.mean(), 3))

# Average number of clicks per session: 0.688
dfc = df[df["click"] == 1]
print("Average number of clicks per session:", round(len(dfc) / len(dfs), 3))

# Click-through rate of the whole dataset: 0.084
print("Click-through rate of the whole dataset:", round(df["click"].mean(), 3))
print("")


# Proportion of sessions with at least one click: 0.466
dfcc = dfc.value_counts("query_no")
print(f"% of sessions with at least one click: {round(len(dfcc) / len(dfs) * 100, 3)}%")

# Proportion of sessions with more than one click: 0.131
print(f"% of sessions with more than one click: {round(len(dfcc[dfcc > 1]) / len(dfs) * 100, 3)}%")

# Average CTR within a session: 0.083
dfqc = df.groupby("query_no", as_index=False)["click"].mean()
print("Average CTR within a session:", round(dfqc["click"].mean(), 3))
print("")


df_val = pd.read_feather("/beegfs/scratch/user/rdeffaye/baidu-bert/features/annotations-features.feather", 
                         columns = ["query_md5", "text_md5"])
dfq_val = df_val.value_counts("query_md5")
dfd_val = df_val.value_counts("text_md5")
dfqd_val = df_val.groupby("query_md5", as_index=False)["text_md5"].value_counts()

dfd_text = df_train.value_counts("text_md5")
dfq_text = df_train.value_counts("query_md5")
dfqd_text = df_train.groupby("query_md5", as_index=False)["text_md5"].value_counts()

# Queries: 6,985
print("Queries in annotated set:", len(dfq_val))

# Docs: 381,552
print("Documents in annotated set:", len(dfd_val))

# Total query/document pairs: 382,038
print("Total query/document pairs in annotated set:", len(dfqd_val))
print("")

# % of train queries occurring in the annotated set: 0.064%
dfqjoin = dfqd_text.merge(dfqd_val, "inner", on = "query_md5").value_counts("query_md5")
print(f"% of train queries occurring in the annotated set: {len(dfqjoin) / len(dfq_text) * 100:.3f}%")

# % of annotated queries occurring in the train set: 12.713%
print(f"% of annotated queries occurring in the train set: {len(dfqjoin) / len(dfq_val) * 100:.3f}%")
print("")

# % of train docs occurring in the annotated set: 0%
dfdjoin = dfqd_text.merge(dfqd_val, "inner", on = "text_md5").value_counts("text_md5")
print(f"% of train docs occurring in the annotated set: {len(dfdjoin) / len(dfqd_train) * 100:.3f}%")

# % of annotated docs occurring in the train set: 0%
print(f"% of annotated docs occurring in the train set: {len(dfdjoin) / len(dfqd_val) * 100:.3f}%")
print("")

# % of train pairs occurring in the annotated set: 0%
dfqdjoin = dfqd_text.merge(dfqd_val, "inner", on = ["query_md5", "text_md5"]).value_counts("text_md5")
print(f"% of train pairs occurring in the annotated set: {len(dfqdjoin) / len(dfd_text) * 100:.3f}%")

# % of annotated pairs occurring in the train set: 0%
print(f"% of annotated pairs occurring in the train set: {len(dfqdjoin) / len(dfd_val) * 100:.3f}%")


