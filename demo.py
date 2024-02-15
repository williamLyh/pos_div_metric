from metrics import calculate_positional_divergence
from datasets import load_dataset

news_data = load_dataset('json', data_files="news_annotations.jsonl")['train']
def filter_nan(datapoint):
    for k, v in datapoint.items():
        if v == None:
            return False
    return True

news_data = news_data.filter(filter_nan)
pred_discourse = news_data['discourse_v1'] 
ref_discourse = news_data['discourse'] 

print(
    calculate_positional_divergence(
        predictions=pred_discourse, 
        references=ref_discourse, 
        num_class=8, 
        num_bins_default=5
    )
)
