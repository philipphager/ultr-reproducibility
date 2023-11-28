MODELS = ["pbm", "two-towers", "naive"]
METRICS = ["dcg@01", "dcg@03", "dcg@05", "dcg@10", "ndcg@10", "mrr@10"] 
           #"click_loss", "BC_dcg@01", "BC_dcg@03", "BC_dcg@05", "BC_dcg@10", "BC_ndcg@10", "BC_mrr@10"]
LOSSES = ["listwise", "listwise-dla", "listwise-em", "pointwise", "pointwise-em", "pointwise-bc", "pairwise-bc"]
