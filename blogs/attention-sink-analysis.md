---
title: Attention Sink in Large Language Models
subtitle: An Experimental Analysis of Attention Sink Across Layers
date: 2026-01-05
tags: [LLM, Attention, NLP, AI, Experiment]
---

# Attention Sink in Large Language Models

## Introduction

The attention sink phenomenon in large language models (LLMs) refers to the tendency of these models to allocate a disproportionately high amount of attention to the initial tokens in a sequence, even when those tokens lack significant semantic relevance. This behavior was first highlighted in the StreamingLLM framework, where retaining the key-value states of the first few tokens (attention sinks) enables stable generation over extremely long contexts without fine-tuning.

The phenomenon arises partly due to the softmax normalization in the attention mechanism, which requires attention scores to sum to 1. When no strongly relevant prior tokens exist, excess attention is "dumped" into the initial tokens as a sink. This effect is observed across many decoder-only transformer-based LLMs and has implications for long-context modeling, KV cache management, and inference efficiency.

In this blog post, I conduct a small-scale experimental analysis using the locally downloaded Qwen2.5-0.5B-Instruct model. The goal is to visualize and quantify the attention sink behavior when processing a simple prompt.

## Experimental Setup

- **Model**: Qwen2.5-0.5B-Instruct (24 layers)
- **Input Prompt**: "who are you?"

The input sequence after tokenization consists of a beginning-of-sequence token followed by the prompt tokens. Attention patterns were extracted during a forward pass with attention weight recording enabled.

## Results

### Average Attention to the First Token Across Layers

The following figure illustrates the average attention weight directed to the first token (averaged over all attention heads) as a function of layer depth.

![Average attention to first token across layers](blogs/images/sink.png)

Observations:
- In the early layers (approximately layers 0–3) and the final layers (approximately layers 22–23), the average attention to the first token remains relatively low.
- In the intermediate layers (roughly layers 4–21), the value peaks significantly, indicating strong attention sink behavior in the middle of the network.

This pattern aligns with prior findings: attention sinks are most pronounced beyond the initial few layers but may diminish near the output as the model focuses on prediction-relevant information.

### Attention Maps Per Layer

Below are the visualized attention maps for each layer (all heads combined or displayed in a grid, depending on the generation method). A prominent vertical line at the first token position in many layers confirms the attention sink effect.

![Layer 0 attention](blogs/images/layer_0_all_heads.png)

![Layer 1 attention](blogs/images/layer_1_all_heads.png)

![Layer 2 attention](blogs/images/layer_2_all_heads.png)

![Layer 3 attention](blogs/images/layer_3_all_heads.png)

![Layer 4 attention](blogs/images/layer_4_all_heads.png)

![Layer 5 attention](blogs/images/layer_5_all_heads.png)

![Layer 6 attention](blogs/images/layer_6_all_heads.png)

![Layer 7 attention](blogs/images/layer_7_all_heads.png)

![Layer 8 attention](blogs/images/layer_8_all_heads.png)

![Layer 9 attention](blogs/images/layer_9_all_heads.png)

![Layer 10 attention](blogs/images/layer_10_all_heads.png)

![Layer 11 attention](blogs/images/layer_11_all_heads.png)

![Layer 12 attention](blogs/images/layer_12_all_heads.png)

![Layer 13 attention](blogs/images/layer_13_all_heads.png)

![Layer 14 attention](blogs/images/layer_14_all_heads.png)

![Layer 15 attention](blogs/images/layer_15_all_heads.png)

![Layer 16 attention](blogs/images/layer_16_all_heads.png)

![Layer 17 attention](blogs/images/layer_17_all_heads.png)

![Layer 18 attention](blogs/images/layer_18_all_heads.png)

![Layer 19 attention](blogs/images/layer_19_all_heads.png)

![Layer 20 attention](blogs/images/layer_20_all_heads.png)

![Layer 21 attention](blogs/images/layer_21_all_heads.png)

![Layer 22 attention](blogs/images/layer_22_all_heads.png)

![Layer 23 attention](blogs/images/layer_23_all_heads.png)

The intermediate layers exhibit a clear dark vertical band corresponding to the first token, confirming substantial attention allocation independent of semantic content.

## Discussion

Even in a relatively small model such as Qwen2.5-0.5B-Instruct, the attention sink phenomenon is readily observable. The concentration of attention on the first token in middle layers suggests that this behavior emerges naturally during pre-training and is not exclusive to larger models.

This experiment provides empirical evidence supporting the broader literature on attention sinks. Future work could explore how this pattern varies with longer inputs, different prompts, or across model families.

## References

- [1] Xiao et al. (2023). Efficient Streaming Language Models with Attention Sinks. arXiv:2309.17453.
- [2] Gu et al. (2024). When Attention Sink Emerges in Language Models: An Empirical View. arXiv:2410.10781.

## Conclusion

The attention sink phenomenon highlights an intriguing artifact of the transformer attention mechanism in autoregressive language models. Through simple visualization on a compact model, we confirm its presence and layer-dependent strength. Understanding and leveraging this behavior remains valuable for advancing long-context and efficient inference techniques in LLMs.