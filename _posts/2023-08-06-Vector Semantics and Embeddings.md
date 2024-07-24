---
layout: post
title: Vector-Semantics-and-Embeddings
date: 2023-08-06 15:47:57 +0800
tags: NLP
description: Texts can be represented by vectors, called embeddings.
---

>Vector Semantics: learns representation of the meanings of words, called **embeddings** directly from their distributions in texts

- ## Lexical Semantics - several concepts to know first
	- #### lemmas and senses
		- lemma is also called citation form, the original form of a word, a word can have muliple word forms.
		- A word can have different meanings, each aspect of the meaning of a word is a word sense. A word being polysemous makes interpretation difficult because word sense is ambiguious (the task of word sense disambiguation)
	- #### synonyms
		* it is used to describe a relationship of approximate or rough synonym.
			* **principle of contrast:** a difference in linguistic form is always associated with some difference in meaning
			* two words that are synonyms can have the same **propositional meaning**
	- #### word similarity
		- synonym is a relation between word senses, similarity is a relation between words
		- cat and dog are not synonyms, but they are similar words
	- #### word relatedness
		- a relation between words (traditionally called association), coffee and cup
		- belong to the same **semantic field**, which is a set of words that cover a particular semantic domain and bear structured relations with each other
		- topic models
	- #### semantic frames and roles
		- semantic frame is a set of words that denote perspectives or participants in a particular type of event
		- frames have semantic roles which words in a sentence can take on
	- #### connotations
		- words that have affective meanings
		- dimensional models of emotion, representing a word as a point in space.


- ## Vector semantics
> represent a word as a point in a multi-dimensional semantic space that is derived from the ***distributions of word neighbors***. Vectors for representing words are called ***embeddings.***
- The core idea is that two words that appear in very similar distributions (whose neighboring words are similar) have similar meanings

- ## Words and Vectors
	- ### Term-document Matrix
		- ![[term-document matrix.png]]
		- column vectors that represent each document would have dimensionality of |V|, size of vocabulary, so we can think of a document as a point in a |V| dimensional space
			- similar documents have similar vectors, because they tend to have similar words
		- row vectors that represent each word would have dimensionality of |D|, number of documents
			- similar words have similar vectors because they tend to appear in similar documents
	- ### Term-term Matrix
		- $|V| \times|V|$ 
		- each cell records the number of time the row word (target) and the column word (context) co-occur in some context (could be a document) in training corpus
		- usually use smaller context, such as a window around target word
		- those vectors have large dimensionalities and most of the numbers are zero, so they are **sparse** vectors

- ## Cosine Similarity
	- the dot product (inner product) acts as a similarity because it wll tend to be high just when the two vectors have large values in the same dimension, meanwhile, if vectors have zeroes in different dimensions will have a dot product of 0 (strong dissimilarity)
	- but raw dot product tend to favor long vectors, so we need to normalize it, which leads to the definition of cosine of the angle between the two vectors
	- cosine metric for measuring similarity: $$ cosine(\textbf{v, w})=\frac{v\cdot w}{|\textbf{v}||\textbf{w}|}=\frac{\sum_{i=1}^{N}v_iw_i}{\sqrt{\sum_{i=1}^{N}v_i^2}\sqrt{\sum_{i=1}^{N}w_i^2}} $$
- ### TF-IDF: weighting terms in the vector
	- raw frequency is not very discriminative, becasue some words are ubiquitous
	- tf-idf is the product of two terms
		- tf: term frequency $$tf_{t,d}=count(t,d)$$
		- but a word that appears 100 times in a document doesn't make that word 100 times more relevant to the meaning of the document, so we use log10: $$tf_{t,d}=log_{10}(count(t,d)+1)$$plus one because we cannot log 0
		- idf is a factor to give higher weights to those occur only in a few documents, they are very useful to discriminate documents
			- document frequency: the number of documents a term t occur in
			- we want to emphasize such words via **inverse document frequency**$$\frac{N}{df_{t}}$$ the fewer documents in which a term occurs, the higher this weight
				- because of the large number of documents in many collections:$$idf_t=log_{10}(\frac{N}{df_t})$$
		- the tf-idf weighted value for word t in document d is: $$w_{t,d}=tf_{t,d}\times idf_{t}$$
		- ![[tf-idf example.png]]

- ## Pointwise Mutual Information PMI
	- Positive Pointwise Mutual Information PPMI for term-term matrices
	- PPMI: the best way to weigh the association between two words is to ask how much ***more*** the two words co-occur in our corpus than we would have a priori expected them to appear by chance
	- **PMI**
		- a measure of how often two events x and y occur, compared with what we would expect if they were independent
		- PMI between a target word w and a context word c is: $$PMI(w,c)=log_2\frac{P(w,c)}{P(w)P(c)}$$
			- numerator tells us how often w and c are observed to co-occur (could be computed using MLE)
			- denominator tells us how often we would expect w and c to co-occur assuming they occurred independently
			* **so the ratio gives us an estimation of how much more the two-words co-occur than we expect by chance**
			* PMI is very useful to find words that are strongly associated
			* we don't trust negative PMI values unless our corpora are enormous (if individual probability is very small, it requires us to be certain P(w,c) is different than the even smaller product of probability.)
		* PPMI $$PPMI(w,c)=max(log_2\frac{P(w,c)}{P(w)P(c)},0)$$
		* a co-occurence matrix F with W rows(words) and C columns(contexts)
			* individual probabilities: $$p_{ij}=\frac{f_{ij}}{\sum_{i=1}^{W}\sum_{j=1}^{C}f_{ij}},\ p_{i*}=\frac{\sum_{j=1}^Cf_{ij}}{\sum_{i=1}^{W}\sum_{j=1}^{C}f_{ij}}, \ p_{*j}=\frac{\sum_{i=1}^Wf_{ij}}{\sum_{i=1}^{W}\sum_{j=1}^{C}f_{ij}} $$
			* PPMI is formally defined as $$PPMI_{ij} = max(log_2\frac{p_{ij}}{p_{i*}p_{*j}},0)$$
		* PPMi is biased towards infrequent events, very rare words tend to have very high PMI values, to improve this: $$PPMI_\alpha(w,c)=max(log_2\frac{P(w,c)}{P(w)P_\alpha(c)},0)$$$$P_\alpha(c) = \frac{count(c)^\alpha}{\sum_ccount(c)^\alpha} $$ $\alpha$ = 0.75 improved the performance
		* another alternative solution is laplace smoothing
* ## Application
	* tf-idf model of meaning can be used to decide whether two documents are similar
		* centroid of all those vectors: has the minimum sum of squared distances to each of the vectors in the set $$d=\frac{w_1+w_2+...+w_k}{k}$$
		* information retrieval, plagiarism detection, news recommender systems

- ## Word2Vec [word2vec](https://code.google.com/archive/p/word2vec/ "google code")  
	- a more powerful word representation: ***embeddings, short dense vectors*** (d=50-1000)
	- most of the vector values are real-valued numbers that can be negative
	-  dense vectors are better
		- less dimensions requires the classifier to learn far fewer weights
		- smaller parameter space possibly helps with generalization and avoid overfitting
		- are better at capturing synonyms
	- #### the idea of self-supervision
		- we train a classifier that predicts whether a word A appears near B, we do not care about the prediction task, but the weights learned by the classifiers
		- **we use the running text as implicitly supervised training data**
		- A appears near B is the golden correct answer
	- ### Skip-gram with negative sampling(SGNS) algorithm
		- treat the target word and a neighboring context word as positive examples
		- randomly sample other words in the lexicon as negative samples
		- use logistic regression to train a classifier to distinguish those two cases
		- use the learned weights as the embeddings
	- ### classifier
		- classification task
			- given a tuple (w, c) of target word w and candidate context word c
			- returns the probability that c is a real context word of w
				- $P(+|w, c)$
				- while $P(-|w,c)=1-P(+|w,c)$
		- how to compute the probability?
			- use the embedding similarity:
				-  a word is likely to occur near the target if its embedding vector is similar to the target embedding, and two vectors are similar if they have a high dot product $$Similarity(w,c)\approx\textbf{w}\cdot\textbf{c}$$ which ranges from $-\infty\ to\ \infty$ 
				- use logistic or sigmoid function to turn it to a probability $$\sigma(x) = \frac{1}{1+exp(-x)}$$
				- so we have the following equation: $$P(+|w,c)=\sigma(\textbf{w}\cdot\textbf{c})=\frac{1}{1+exp(-\textbf{w}\cdot\textbf{c})}$$
				- There are many context words within a window, skip-grams assumes that they are independent: $$P(+|w,c_{1:L}) = \prod_{i=1}^{L}\sigma(\textbf{c}_{i}\cdot\textbf{w})$$
				- In summary, skip-gram algorithm assigns a probability based on how similar the context window (with multiple context words) is to the target word
				- to compute the probability, we need
					- embeddings for each target word
					- embeddings for each context word
	- ### Classifier learned weights as skip-gram embeddings
		- Skip-gram stores two embeddings for each word (one as target and one as context), so we have two matrices to learn: **W** and **C**, each containing an embedding for every one of the |*V*| words![[skip-gram embeddings.png]]
		- How to learn skip-gram embeddings
			- assigns a random embedding vector for N words (vocabulary)
			- to train a binary classifiers, we need positive and negative examples:
				- window size of L, then nearby the target word w, we have L positive examples
				- for negative examples, we have a parameter k, SGNS uses $k\times L$ negative examples, so for every positive example, we have k negative examples
					- the noise word is a random word from the lexicon (not the target word)
					- noise words are chosen according to their weighted unigram frequency, in practice, we choose $\alpha=0.75$:
						- $P_\alpha(w)=\frac{count(w)^\alpha} {\sum_{w'}count(w')^\alpha}$ 
						- because it gives rare words slightly higher probability:
							-   $P_\alpha(a)=\frac{.99^.75}{.99^.75+.01^.75}=.97$
							- $P_\alpha(b)=\frac{.01^.75}{.01^.75+.99^.75}=.03$ 
			- having two sets of instances and an initial set of embeddings, we need to shift the embeddings of each word to be more like the embeddings of words that occur nearby in texts
				- maximize the similarity of the target word, context word pairs from the positive examples
				- minimize the similarity of the word pairs from negative exmaples
				- we can do this by minimizing a loss function
					- $$\begin{aligned} L_{CE} &=-log[P(+|w,c_{pos})\prod_{i=1}^kP(-|w,c_{neg_i})] \\ &=-[logP(+|w,c_{pos})+\sum_{i=1}^{k}logP(-|w,c_{neg_i})] \\ &= - [logP(+|w,c_{pos}+\sum_{i=1}^k(1-P(+|w,c_{neg_i}))] \\ &=-[log\sigma(c_{pos}\cdot w) + \sum_{i=1}^{k}log\sigma(-c_{neg_i}\cdot w] \end{aligned}$$ the first term is to assign a high probability for a real context word of being a neighbor, the second term is to assign a high probability for a noise word of being a non-neighbor
					- so, in the end we need to maximize the dot product of the word with actual context words, minimize the dot products of the word with noise words
					- we use stochastic gradient descent to minimize the loss function, derivatives are: 
					- $$\begin{aligned} \frac{\partial L_{CE}}{\partial c_{pos}} &=[\sigma(\textbf{c}_{pos}\cdot \textbf{w})-1]\textbf{w} \\ \frac{\partial L_{CE}}{\partial c_{neg}}&=[\sigma(\textbf{c}_{neg}\cdot \textbf{w})]\textbf{w} \\ \frac{\partial L_{CE}}{\partial w} &=[\sigma(\textbf{c}_{pos}\cdot\textbf{w})-1]\textbf{c}_{pos}+\sum_{i=1}^k[\sigma(\textbf{c}_{neg_i}\cdot \textbf{w})]\textbf{c}_{neg_i}] \end{aligned}$$
					- we update the above equations from time step t to t+1
		- other static embeddings
			- fasttext
				- improved word2vec in terms of dealing unknown words (unseen in test corpus)
			- GloVe
				- combine the intuitions of count-based models like PPMI and capture the linear structures used by methods like word2vec