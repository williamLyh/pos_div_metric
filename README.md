# pos_div_metric


```python
# Minimum xample
from metrics import calculate_positional_divergence

predictions = [[1,2,1,1,0,2,2,4,0,0,5]]
references = [[1,1,1,1,0,0,2,3,3,0,5,5]]
PDD = calculate_positional_divergence(
    predictions=predictions, 
    references=references, 
    num_class=6, 
    num_bins_default=3
)
```
