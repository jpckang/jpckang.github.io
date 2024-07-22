---
layout: post
title: Transformers
date: 2023-02-01 15:47:57 +0800
tags: deep learning
description: Transformers introduces attention mechanism to efficiently represent sentences.
---

> We will start by introducing the idea of pretraining to explaining why we need pretrained language models, but here we only focus on the most common architecture to implement this idea.
> We firstly introduce the core of attention mechanism, and how self-attention layer works. Next, we describe the structure of a transformer block, which contains self-attention layers. We then move to two other designs of transformer: multi-heads attention and positional embeddings.


## Introduction
### Distribution Hypothesis
The idea is that word meanings (loosely called) can be learned even without any grounding in the real world, soley based on the content of the text we have encountered in our lives. Via the words they co-occur with, we can obtain their word meanings.
This means, long after some words' initial acquisition in novel context, that we can still learn knowledge through this process (through context) during language processing.

### Pretraining
We can formalize the above idea to pretraining, which is learning some representation of word meanings (or sentences) by very large amounts of texts.
The most common architecture for language modelling is **transformer**. It offers new mechanisms like **self-attention and positional encodings** that can represent time and help us focus on how words relate to each other over long distances.

## Self-Attention Networks
> architecture of transformers are unlike LSTMs, which are based on recurrent connections (they are hard to parallelize), instead, they can be trained more efficiently.

### Components of a transformer
What transformers do is to map sequences of input vectors ($\textbf{x}_1,...,\textbf{x}_n$) into sequences of output vectors ($\textbf{y}_1,...,\textbf{y}_n$) of the **same length**. 
A transformer is made up of stacks of transformer **blocks**. Each block is a multi-layer networks, consisting of simple linear layers, fnn, and **self-attention layers**, which is the focus of this section.

### Self-attention layers
We firstly describe the core and a simplified version of how self-attention approach works, then we introduce the innovation of such process.
![[information flow in a causal self-attention model.png]]
The figure above illustrates the information flow in a self-attention model. 
- We call it causal or masked, or backward-looking self-attention layer, because for each item being processed, the model has access to all the previous inputs (including the current one). 
	- this means that we can use them for autoregressive generation
- We can also see that a self-attention layer takes sequences of input and maps them into the same length output sequences as well.
- Computation of each item is independent of each other
	- this means that we can parallelize the both forward inference and training of such models.

#### Core of attention-based approach
- **compare** an item of interest to a collection of other items in a way that reveals their relevance in the current context
	- which means that through those comparisons, we can have a kind of representation for a sentence using words in the sentence. 
		- so, in the case of self-attention, those comparisons are to other elements within a given sequence
	- results of such comparisons are used to compute an output for the current input
	- example below is computation for one of the output sequences $y_i$
		- we use a simple form of **comparison**, dot product:  
			- $score(\textbf{x}_i,\textbf{x}_j) = \textbf{x}_i\cdot\textbf{x}_j$
			- in this case, the $i$th element of x is the current focus of attention, $x_3$, and we compare it with the two previous items $x_1,x_2$ and $x_3$ itself, so $j$=1,2,3
		- to avoid numerical values being too large, we pass them through a softmax to **normalize** them, giving a probability distribution and **create a vector of weights**: $$\begin{align} \alpha_{ij}&=softmax(score(x_i, x_j))\ \forall j\leq i \\ &=\frac{exp(score(x_i,x_j))}{\sum_{k=1}^{i}exp(score(x_i,x_k))}\ \forall j<i \end{align}$$
		- now we have a set of results of comparisons as weights, we can **straighforwardly use weighted sum** to generate an output from inputs:
			- $y_i = \sum_{j\leq i}\alpha_{ij}x_j$    

#### Transformers
We now have an overview of how attention works, but transformers have a more sophisticated way to represent how words can be contributed to the represention of longer inputs. Transformers do this by assigning the input embeddings with three different roles (and this is done by three different weight matrices).
- For an input embedding, during the whole process of attention, it has three roles:
	- when it is being compared to other preceding inputs, it is the current focus of attention. We call it **query**.
	- when it is being compared to the current focus of attention, it is a preceding input. We call it **key**.
	- when it is being used to compute the output for the current focus of attention, we call it **value**.
- to capture the different roles, we give them different weight matrices to project each input embedding into a representation of its role:
	- $\textbf{q}_i = \textbf{W}^Q\textbf{x}_i;\ \textbf{k}_i=\textbf{W}^K\textbf{x}_i;\ \textbf{v}_i=\textbf{W}^V\textbf{x}_i$ 
- now we will compute the comparisons and output using the right roles:
	- $score(\textbf{x}_i,\textbf{x}_j)= \textbf{q}_i\cdot\textbf{k}_j$
	- before passing scores to softmax, we have to consider the result of dot product being too large, and exponentiating large values will cause numerical issues and gradient loss
		- so we will scale the dot product down with:
			- $score(\textbf{x}_i,\textbf{x}_j)=\frac{\textbf{q}_i\cdot\textbf{k}_j}{\sqrt{d_k}}$ 
			- $d_k$ is the dimensionality of query and key vectors
	- the softmax calculation resulting in $\alpha_{ij}$ is the same, we just want to normalize the scores
	- $\textbf{y}_i=\sum_{j\leq i}\alpha_{ij}\textbf{v}_j$
- up to this point, we know how to calculate a single output at a single time step *i*, however, every $\textbf{y}_i$ can be computed in parrallel because we have access to the entire sequence of input tokens all the time. We use efficient matrix muliplication routines:
	- pack all the input tokens into one input matrix $\textbf{X}\in \mathbb{R}^{N\times d}$ , so we have N input tokens, dimension of each is d, one row of **X** is one token.
	- the key, query, value weight matrices now have the dimensionality of $d\times d$, we will mulitply them to input matrix
		- $\textbf{Q}=\textbf{X}\textbf{W}^Q; \ \textbf{K}=\textbf{X}\textbf{W}^K; \ \textbf{V}=\textbf{X}\textbf{W}^V$
		- three matrices $\in \mathbb{R}^{N\times d}$ 
	- now we can reduce the entire self-attention step for an entire sequence of N input tokens to:
		- $SelfAttention(\textbf{Q},\textbf{K},\textbf{V})=softmax(\frac{\textbf{Q}\textbf{K}^{\intercal}}{\sqrt{d_k}})\mathbf{V}$ 
		- we should be able to describe the above computation step by step now
			- we calculate the comparison result(using dot product) for all the requisite query-key vectors in one matrix multiplication using the query and key matrices, which gives a product of shape $N\times N$
			- then we scale the scores down before passing them to the softmax function to avoid numerical issues and gradient loss
			- next we softmax it for normalization and create a weight matrix
			- finally we calculate the weight sum for each output in one matrix multiplication using the value matrix
		- however, $\mathbf{QK}^{\intercal}$ is inappropriate 
			- because it actually calculate the scores for each query with every key, *including those that follow the query.*
			- this means we are predicting the next word under the condition that we already know the next word
			- to fix this, we will eliminate all the scores which represent the following tokens by zeroing out the upper-triangular elements in the $\mathbf{QK}^{\intercal}$ matrix.![[attention comparison matrix.png]]
			- limit the input length
				- one more issue we can see from the figure above, the attention is quadratic in the length of the input because we calculate the dot products between each pair of the query-key value. This will lead to extremely expensive calculation if the input consists of long documents. 

#### Transformer Blocks
![[transformer block.png]]
Now we discuss the structure of transformer and how each component works briefly.
From the figure above, we have a clear overview of what a transfomer block consists of:
- self-attention layer, where the core of attention mechanism lies.
- residual connections 
	- pass information from a lower layer to a higher layer without going throught the intermediate layers
	- more details of how residual connections can help, see the paper [[deep residual learning for image recognition.pdf]]
- layer norm [[layer normalization.pdf]]
	- layer normalization can help improve the training performance in deep neural networks by keeping the values of a hidden layer in a range that facilitates gradient-based training
	- first, we calculate the mean and std
		- $\mu = \frac{1}{d_h}\sum_{i=1}^{d_h}x_i$
		- $\sigma=\sqrt{\frac{1}{d_h}\sum_{i=1}^{d_h}(x_i-\mu)^2}$ 
	- then we normalize the values:
		- $\hat{x}=\frac{x-\mu}{\sigma}$ 
	- to implement the layer normalizaiton, we have two learnable parameters:
		- $LayerNorm=\gamma\hat{x}+\beta$ , (gain and offset values) 
- we can represent a transformer block as:
	- $\mathbf{z}=LayerNorm(\mathbf{x}+SelfAttention(\mathbf{x}))$ 
	- $\mathbf{y}=LayerNorm(\mathbf{z}+FNN(\mathbf{z}))$ 

#### Multihead attention
Why we need multihead attention and what they are:
- Words in a sentence can hold different relationships simultaneously, including syntactic, semantic, and discourse relationships.
- A single transformer block cannot learn to capture all these relationships.
- so we introduce the multihead self-attention layers:
	- Multihead attention layers are sets of self-attention layers, we call heads, that reside in parallel layers at the same depth in a model, each with its own set of parameters.
	- with these distinct parameters, we can capture the different aspects of relationships at the same level of abstraction

**How multiheads attention works**
$$
\begin{align}
MultiHeadAttention(\mathbf{X})=(\mathbf{head}_1\oplus\mathbf{head}_2...\oplus\mathbf{head}_h)\mathbf{W}^O \\
Q=\mathbf{XW}_i^Q; \ \mathbf{K}=\mathbf{XW}_i^K; \ \mathbf{V}=\mathbf{XW}_i^V \\
\mathbf{head}_i=SelfAttention(\mathbf{Q,K,V})
\end{align}
$$

- for each head *i*, it has its own set of key, query, value matrices $\mathbf{W}_i^Q,\mathbf{W}_i^K,\mathbf{W}_i^V$
- we calculate the output of each head using such matrices:
	- unlike a single head attention, they do not have the same dimensionality of input and output vectors *d*, but have their own
		- $\mathbf{W}_i^Q\in \mathbb{R}^{d\times d_k}$, $\mathbf{W}_i^K\in \mathbb{R}^{d\times d_k}$, $\mathbf{W}_i^V\in \mathbb{R}^{d\times d_v}$
		- they can project the packed input matrix **X** into $\mathbf{Q}\in \mathbb{R}^{N\times d_k}$, $\mathbf{K}\in \mathbb{R}^{N\times d_k}$, $\mathbf{V}\in \mathbb{R}^{N\times d_v}$ as different roles in the attention mechanism
		- $\mathbf{Q,K,V}$ are used to compute the self-attention, we review this here:
			- $SelfAttention(\mathbf{Q,K,V})=softmax(\frac{\mathbf{QK}^{\intercal}}{\sqrt{d_k}})\mathbf{V}$ 
		- the output of each of the *h* heads is in shape $N\times d_v$
- then, we combine the those output vectors and scale them down to *d*
	- by concatenating all the heads' outputs
	- and using another linear project $\mathbf{W}^O\in \mathbb{R}^{hd_v\times d}$ 
	- so we reduce it to the original output dimension for each token, $N\times d$  ![[multihead attention layer.png]]

#### Positional embeddings
To model the position of each token in an input sequence, we moodify the input embeddings by combining them with positional embeddings which are specific to each position.
**How are positional embeddings generated?**
- a simple method
	- start with randomly initialized embeddings according to each possible input position up to some maximum length (we certainly can have an embedding for some position n, they are learned as with word embeddings during training)
	- add them into the corresponding input word embedding (just add, not concatenating)
	- and we have a new embedding for further processing![[simple way for positional embedding.png]]
- we have some problems in this approach:
	- initially, we may have plenty of training examples, which requires lots of initial positions in our inputs
	- but we might have correspondingly fewer positions at the outer length limits(?unclear)
	- there will be some latter embeddings that are poorly trained so it may not well generalize during testing
- An alternative approach:
	- choose a static function that maps integer inputs to real-values vectores in a way that captures the inherent relationshipsa among positions

> refer to the original transformer paper for more detailed design
> [[attention is all you need.pdf]]
