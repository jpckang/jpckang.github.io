---
layout: post
title: Question Answering
date: 2023-01-11 15:48:22 +0800
tags: question answering
description: Notes of Chapter Question Answering from Speech and Language Processing 3rd.ed
---

> factoid questions: questions that can be answered with simple facts expressed in short text
>
> - information retrieval based QA
> - knowledge-based QA

## information retrieval

> retrieving all manner of media based on user information needs (search engine)
> ![[Screenshot 2022-11-20 at 14.11.48.png]]

- ### how a query and a document match

  - #### term weight for document words

    - tf-idf
      - term frequency tell us how frequent the word is (more frequent words are likely to be more informative about the document's contents)
        - we use $log_{10}$ instead of raw count(one word appearing 100 times does not mean it is 100 times more important), and add 1 to the count because we can't log 0 $tf_{t,d} = log_{10}(count(t,d)+1)$
      - document frequency of a term is the number of document it occurs in (words that occur only in a few documents are more useful for discriminating the documents)
        - define inverse document frequency $idf_t = log_{10}\frac{N}{df_t}$ the fewer documents a word occurs in, the higher this weight (more discriminating)
      - tf-idf value $tf-idf(t,d) = tf_{t,d}\cdot idf_t$

  - document scoring

    - cosine similarity for scoring a document 

      $$score(q,d)=cos(q,d)=\frac{q\cdot d}{|q|\cdot|d|}  = \frac{q}{|q|}\cdot\frac{d}{|d|}$$ 

      where q is a query vector and d is document vector, spell out equation using td-idf values and dot product as a sum of products: 

      $$score(q,d) = \sum_{t\in q}\frac{tf\text{-}idf(t,q)}{\sqrt{\sum_{q_i\in q}tf\text{-}idf^2(q_i,q)}}\cdot \frac{tf\text{-}idf(t,d)}{\sqrt{\sum_{d_i\in d}tf\text{-}idf^2(d_i,d)}} $$

    - However, we can simplify the query processing, use the following simple score:

      $$score(q,d) = \sum_{t\in q}\frac{tf\text{-}idf(t,d)}{|d|}$$

  - #### **BM25** as another weight scheme

    - k as the knob that adjust the balance between term frequency and IDF
    - b controls the importance of document length normalization

  - #### Inverted Index

    - given a query term, gives a list of candidate documents
    - compositions:
      - dictionary: a list of terms, each pointing to a positings list for the term
      - postings: the list of document IDs associated with each term ![[inverted index.png]] 
      - indexing based on bigrams works better than unigram
      - hashing algorithms are better than the inverted index

- ### evaluation of IR systems

  - **Precision and Recall** 
    
    $$Precision = \frac{|R|}{|T|} \ \ \ \ \ Recall = \frac{|R|}{|U|}$$ 
    
    is relevant documents, T is returned ranked documents, U is all relevant documents in the whole collection
    
    - Precision and recall are not adequate, to capture how well a system does at putting relevant documents higher in the ranking:
      - interpolated precision: choose the maximum precision value achieved at any level of recall at or aobve the one we are calculating
      
        $$IntPrecision = \mathop{max}\limits_{i>=r}Precision(i) $$
      
      - compare two systems or approaches by comparing their precision-recall curves
    
  - **Mean average precision** (MAP)
    - gives a single metric that can be used to compare systems
    
    - note precision only at those points where a relevant item has been encountered
      - assume $R_r$ is the set of relevant documents at or above r, average precision is:
      
        $$AP = \frac{1}{|R_r|}\sum_{d\in R_r}Precision_r(d) $$
      
      - $Precision_r(d)$ is the precision measured at the rank where document d was found, for an ensemble of queries, we average over averages:
      
        $$MAP = \frac{1}{|Q|}\sum_{q\in Q}AP(q) $$

- #### IR with dense vectors

>***tf-idf and BM25 algorithms only work if there is exact overlap of words between the query and document, it is likely to have vocabulary mismatch problem***

- modern approaches use encoders like BERT
  - more complex versions like using averaging pooling over the BERT outputs or add extra weight matrices after encoding or dot product steps
  - for efficiency, modern systems use nearest neighbor vector search algorithms like **Faiss**.



- ## IR-based Factoid Question Answering

  - also **open domain QA**, answer a user's questions by finding short text segments from the web or other large collection of documents
  - **retrieve and read model** ![[retrieve and read model.png]]
  - text retrieval
  - reading comprehension

- #### Datasets

  - SQuAD
  - HotpotQA
  - TriviaQA
  - Natural Questions 
  - TyDi QA (non-English)

- #### Reader

  - for extractive QA, the answer that reader produces is a span of text in the passage
  - standard baseline algorithm is to pass the question and passage to any encoder like BERT
  - BERT allows up to 512 tokens, for longer passages, we create multiple pseudo-passage observations

- ## Entity Linking

  - EL is the task of associating a mention in text with representation of some real-world entity in an ontology

  - EL is done in two stages:

    - mention detection
    - mention disambiguation

  - #### Linking based on Anchor Dictionaries and Web Graph

    - TAGME linker [[Fast and accurate annotation of short texts with Wikipedia pages.pdf]]
      - anchor dictionary
      - linke probabillity
      
    - Mention Detection
      - query the anchor dictionary for each token sequence
      
    - Mention Disambiguation
      - Spans match anchors for multiple Wikipedia entities/pages
      
      - prior probability 
      
        $$prior(e\rightarrow a) = p(e|a) = \frac{count(a \rightarrow e)}{link(a)} $$ 
      
        This gives the link of the highest probability, but it is not always correct.
      
      - relatedness/coherence 
      
        $$ rel(A,B)=\frac{log(max(|in(A)|,|in(B)|))-log(|in(A)|\cap |in(B)|)}{log(|W|)-log(min(|in(A)|, |in(B)|))} $$ 
      
        W is the collection of all pages
      
      - vote given by anchor b to the candidate annotation a to X is 
      
        $$vote(b,X)=\frac{1}{|\mathcal{E}(b)|}\sum_{Y\in \mathcal{E}(b)}rel(X,Y)p(Y,b) $$
      
      - Total relatedness score for a to X is 
      
        $$relatedness(a\rightarrow X)=\sum_{b\in \mathcal{X}\backslash a}vote(b,X) $$
      
      - see book for more references.

  - #### Neural Graph-based linking

    - **biencoders** allows embeddings for all the entities in the knowledge based to be prcomputed and cached [[Scalable Zero-shot Entity Linking with Dense Entity Retrieval.pdf]]
    - ELQ linking algorithm [[Efficient One-Pass End-to-End Entity Linking for Questions.pdf]]
      - Entity Mention Detection
      - Entity Linking
      - Training

- ## Knowledge-based Question Answering ^1719df

  - **Graph-based QA**
    - from RDF triple stores: a set of factoids
    - RDF: a predicate with two arguements, expressing some simple relation or proposition $$<subject, predicate, object>$$
    - datasets
      - SimpleQuestions-Freebase
      - FreebaseQA
      - WEBQUESTIONS
      - COMPLEXWEBQUESTIONS
    - steps:
      - entity linking
      - mapping from question to canonical relations in knowledge base (triple)
      - relation detection and linking
        - compute similarity (dot product) between the encoding of the question text and an encoding for each possible relation
      - ranking of answers
        - heuristic
        - train a classifier (concatenated entity/relation encodings -> predict a probability)
  - **QA by semantic parsing**
    - uses a semantic parser to map the question to a structured program to produce an answer
    - predicate calculus
      - can be converted to SQL
    - query language
      - SQL
      - SPARQL
    - Semantic parsing algorithms
      - fully supervised with questions paried with a hand-built logical form [[Logical Representations of Sentence Meaning]]
        - a set of question paired with their correct logical form
          - GEOQUERY
          - DROP
          - ATIS
        - take those pairs of training tuples and produce a system that maps from new questions to their logical forms
          - baseline: simple sequence-to-sequence model 
          - [[Computational Semantics and Semantic Parsing]]
      - weakly supervised by questions paired with an answer 

- ## Using Language Models to do QA

  - query a pretrained language model, answer a question solely from information stored in its  parameters
    - T5 langauge model, encoder-decoder architecture [[How Much Knowledge Can You Pack Into the Parameters of a Language Model.pdf]]

- ## Classic QA Models - Watson DeepQA system

  - **Question Processing**
    - named entities are extracted
    - question focus is the string of words in the question that corefers with the answer
    - lexical answer type: a word or words which indicate the smenatic type of the answer
  - **Candidate Answer Generation**
    - query structured resources with relations and known enetities
    - extract answers from text
      - IR stage to get passages
      - extract anchor texts and all noun phrases from passages
  - **Candidate Answering Scoring**
    - a classifier that scores whether the candidate answer can be interpreted as a subcalss or instance of the potential answer type
    - use time and space relations extracted from structured database
    - use text retrieval to retrieve evidence
    - output is a set of candidate answers, each with a vector of scoring features
  - **Answer merging and scoring**

- ## Evaluation of Factoid Answers
