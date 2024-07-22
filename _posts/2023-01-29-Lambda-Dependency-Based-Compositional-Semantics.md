---
layout: post
title: "Lambda Dependency-Based Compositional Semantics"
date: 2023-01-29 22:05:55 +0800
tags: semantic parsing
description: Like Lambda calculus, lambda DCS is a formal language designed to query Freebase.
---

> Lambda DCS is a kind of formal language to query the knowledge base. It has simplified expressions compared to lambda calculus, due to elimination of variables and making of implicit existential quantifications. We briefly list the fundamentals here by examples comparing with lambda calculus.

- Semantic parsing is the task of transforming the natural utterances into the logical forms. Those forms are in some of formal language such as  $\lambda-$ calculus. So is  $\lambda-$ DCS.
- lambda DCS was designed to query the Freebase.



## Fundamentals

- We have a knowledge base of assertions, namely $\mathcal{K}$ , a set of entities(nodes) $\mathcal{E}$ , a set of properties (edges) $\mathcal{P}$ , then we have $\mathcal{K}\subset \mathcal{E}\times \mathcal{P}\times \mathcal{E}$ . 

- We let [condition] denotes the truth value of the condition. ( $\lambda x.[x=3]$ ) denotes the function returns true if and only if x=3. 
- We let $\textlbrackdbl z \textrbrackdbl$ be the lambda DCS corresponded to the lambda calculus of z.

#### Unary base case

a simple entity in lamdba DCS: 

  $$Seattle\Longleftrightarrow\lambda x.[x=Seattle] \\
\textlbrackdbl e \textrbrackdbl = \lambda x.[x=e]$$

### Binary base case

For a property $p\in \mathcal{P}$  ,  p is a binary logical form, which denotes a function mapping two arguments to whether p holds:

$$PlaceOfBirth \Longleftrightarrow \lambda x.\lambda y.PlaceOfBirth(x, y)\\
\textlbrackdbl p\textrbrackdbl = \lambda x.\lambda y.p(x, y)$$

### Join *

$$PlaceOfBirth.Seattle \Longleftrightarrow \lambda x.PlaceOfBirth(x, Seattle) \\
\textlbrackdbl b.u \textrbrackdbl = \lambda x.\exists y. \textlbrackdbl b \textrbrackdbl(x,y)\and \textlbrackdbl u \textrbrackdbl(y)$$

where b is a binary logical form and u is a unary logical form, and b.u is a unary logical form.

**This is a key feature of join (the central operation of lambda DCS). Implicit existential quantification over argument y shared by b and u. This makes it more apparent when binaries are chained.**

### Intersection 

a set of scientists born in Seattle:

$$Profession.Scientists \sqcap PlaceOfBirth.Seattle\\
\Longleftrightarrow\\
\lambda x. Profession(x,Scientist)\and PlaceOfBirth(x, Seattle)\\
\textlbrackdbl u_1\sqcap u_2 \textrbrackdbl = \lambda x.\textlbrackdbl u_1 \textrbrackdbl(x)\and\textlbrackdbl u_2 \textrbrackdbl(x)$$

From the perspective of graph pattern, intersection allow tree-structured graph patterns, where branch points correspond to the intersections.

### Union

Intersection corresponds to conjunction, union corresponds to disjunction.

$$Oregon \sqcup Washington \Longleftrightarrow \lambda x.[x=Oregon]\or [x=Washington]\\
\textlbrackdbl u_1\sqcup u_2 \textrbrackdbl = \lambda x.\textlbrackdbl u_1 \textrbrackdbl(x)\or\textlbrackdbl u_2 \textrbrackdbl(x)$$



### Negation

US states not bordering California

$$Type.USState\sqcap \neg Border.California\\
\Longleftrightarrow \lambda x.Type(x, USState)\and \neg Border(x, California)\\
\textlbrackdbl \neg u \textrbrackdbl = \lambda x.\neg\textlbrackdbl u \textrbrackdbl(x)$$



