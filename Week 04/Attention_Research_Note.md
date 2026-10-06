# Week 03 research note: attention and causal prediction

Date: 2026-10-05. Based on our paper discussion and the accompanying NumPy experiments.

## Paper question and method

Vaswani et al. ask whether sequence transduction can be effective without recurrence or convolution. Their Transformer uses attention, learned projections, feed-forward layers, positional information, residual connections, and normalization. Parallel computation across known training positions avoids recurrent hidden-state dependencies; autoregressive generation still depends on previously generated tokens.

## Evidence and limitation

For WMT 2014 English-to-French, Table 2 reports ConvS2S Ensemble at 41.29 BLEU and approximately 1.2e21 training FLOPs, versus Transformer (big) at 41.8 BLEU and 2.3e19 FLOPs: 0.51 BLEU higher with about 52 times less estimated computation. This is not a 52-fold wall-clock speed claim. The comparison concerns complete systems, not attention in isolation.

BLEU measures reference n-gram overlap with a brevity penalty; 41.8 is not a percentage correct. Valid alternative translations can be penalized. These results do not establish superiority on every task or human-level translation.

Sources: [Attention Is All You Need, Sections 3–6 and Table 2](https://arxiv.org/html/1706.03762v7); [original BLEU paper](https://aclanthology.org/P02-1040/).

## Equation explained through our implementation

`O = softmax_rows(Q @ K.T / sqrt(d_k) + M) @ V`

A query at destination i is compared with every source key j. Dividing the dot products by sqrt(d_k) controls their scale. Row-wise softmax produces a Gibbs distribution over source positions, independently for each query. The causal mask adds zero at allowed positions and negative infinity at future positions, whose exponential becomes zero. Multiplication by V forms a weighted average of value vectors. O is a representation matrix, not a probability distribution over vocabulary tokens.

For n positions represented by d_model features, Q and K have shape (n, d_k), V has shape (n, d_v), attention weights A have shape (n, n), and O has shape (n, d_v). The learned projection matrices stay fixed during ordinary prediction; Q, K, and V depend on the input.

## Our experiment and results

We implemented stable softmax, temperature comparisons, single-query attention, row-wise softmax, learned-projection-style matrix calculations with manually selected weights, and a causal mask for variable sequence lengths. Weight rows sum to one; future-position weights are zero. Changing the last input leaves earlier outputs unchanged. The notebook was rerun from a fresh namespace and the attention checks were also verified independently during the previous save session.

This establishes properties of our implementation, not useful language prediction. We have not trained embeddings or projection matrices, implemented a complete Transformer, or reproduced translation benchmarks.

## Research question and hypothesis

How much does prediction performance depend on input-dependent attention compared with uniform averaging over allowed positions?

Our hypothesis is that adaptive weights can emphasize contextually relevant information that uniform averaging dilutes. To test predictive usefulness, compare trained systems with otherwise matched architecture, data, and training budget, then evaluate on held-out examples. Hand-chosen matrices demonstrate behavior but cannot establish a prediction-quality advantage.

## Evaluation lesson

Training data adjusts parameters; validation data guides model and setting choices; test data assesses the final choice. Duplicates shared with training contaminate a test: low loss on seen text does not establish generalization. Repeated selection using a test set likewise weakens its independence.
