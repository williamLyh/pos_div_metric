<p align="center">
<img src="asset/icon.png" alt="Alt Text" width="200px" height="200px">
</p>

# PDD: Positional Discourse Divergence
This is the code repo for paper [**Unlocking Structure Measuring: Introducing PDD, an Automatic Metric for Positional Discourse Coherence**](https://arxiv.org/abs/2402.10175).

## What's PDD
PDD is a novel automatic metric designed to quantify the **discourse divergence** between two long-form articles.
It partitions the sentences of an article into multiple position bins and calculates the divergence in discourse structures within each bin.
PDD can
- have certain level of tolerance on local discourse variations.
- handling misaligned numbers of sentences between prediction and reference.


## Example
A minimum working example
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

## Citation
Preprint version
```
@misc{liu2024unlocking,
      title={Unlocking Structure Measuring: Introducing PDD, an Automatic Metric for Positional Discourse Coherence}, 
      author={Yinhong Liu and Yixuan Su and Ehsan Shareghi and Nigel Collier},
      year={2024},
      eprint={2402.10175},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
