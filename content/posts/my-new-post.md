+++
date = '2025-07-30T15:20:50+08:00'
draft = false
math=true
title = 'Scaled Dot-Product Attention'
+++

The Self Attention Mechanism (SA) is a fundamental part of the Transformer model, which is universally applied in large language models. Below is my view of how SA works, and why it works.

![Self attention mechanism](/images/self_attention.png "Self attention mechanism")

The basic structure of SA is shown above. Given a embedded input vector $X$, we apply $3$ learnable linear layers: $W_Q, W_K$ and $W_V$, to generate three new output vectors: Query, Key and Value. This process can be represented as:

$$
Query=XW_Q^T, 
Key=XW_K^T, 
Value=XW_V^T
$$

Then, calculate the attention weight through this function:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \ V
$$

where $Q, K, V$ denotes the Query, Key and Value vector, $K^T$ represents the transposed Key vector. $d_k$ is the dimension of the $Q$ and $K$. 
Every row of $Q$ and $K$ is the semantic feature representation of each input token, so the result of $QK^T$ is the similarity matrix of all the tokens in the input. For example: `"I love tennis, it is my favorite sport."` In this sentence's similarity matrix, the value of $("tennis", "it")$ will be high because they have strong semantic relavance.

After obtaining the similarity matrix, we need to scale it. Assume that each component of $Q$ and $K$ is an independent and identically distributed random variable with mean 0 and variance $1$. For vectors of dimension $d_k$, the expected value of the dot product is still $0$, but its variance increases as the dimension increases:

$$
\text{Var}(Q \cdot K) = \text{Var}\left(\sum_{i=1}^{d_k} Q_i K_i\right) = \sum_{i=1}^{d_k} \text{Var}(Q_i K_i) = \sum_{i=1}^{d_k} \text{Var}(Q_i) \text{Var}(K_i) = d_k
$$

The standard deviation of the dot product is $\sqrt{d_k}$, which means that the value of the dot product can become very large, especially in high dimensions. Therefore, The variance of the dot product is normalized by dividing by $\sqrt{d_k}$. The standard deviation of the scaled dot product is close to $1$, and the distribution of dot product values is more stable and falls within a reasonable range.

Then, we need to utilize the Softmax function to convert the dot product attention score into probability distribution to determine the contribution weight of each key to the final output:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

In self-attention, the input $x$ is a scaled dot product score matrix (of shape $[n, n]$, where $n$ is the sequence length), and Softmax is typically applied along the last dimension (i.e., the scores of all keys corresponding to each query) to produce a probability matrix.

From the function we can also see that the Softmax function is very sensitive to the absolute size of the input value. If the dot product value is too large (for example, far exceeding $1$), the output of Softmax will tend to be "hard-selected", causing the attention weights to be concentrated in a few positions, reducing the model's attention to diverse contexts. Excessively large dot product values can also cause the gradient vanishing during Softmax backpropagation, because the derivative of Softmax for large input values is close to $0$. 

![Softmax Sensitivity](/images/softmax_sensitivity.png "Softmax Sensitivity")

When I was first learning Self Attention mechanism, I was confused about why there is a $V$ matrix here, because the semantic similarity have already been calculated. However, the value vector $V$ is introduced to separate the roles of keys and values:

- $K$: Used to calculate similarity with the query and determine which locations are more relevant.
- $V$: Provide actual feature information for weighted combination.
Benefits of this separation:

Keys and values can represent different information. For example, the key can capture the contextual relevance of a location, while the value can retain richer semantic information. This separation increases the flexibility of the model, enabling it to learn different attention patterns and representations.

## References
> [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.