# 【译文】了解GPT-3、BERT、T5背后的模型：Transformer解析

> 原文链接：["Transformers, explained." by Dale Markowitz](https://daleonai.com/transformers-explained) 最好对比原文食用。
>
> If you have no problem reading in original language, always read the original article. Even large language models can not capture all the differences.
>
> 非直译的使用【】说明，一些文字表达上也有本菜鸡自由发挥的成分，但请放心，一定不涉及原理解释的部分：）

​	你听过一句谚语：“当你有一个锤子，所有东西看起来都像根钉子” 吗？在机器学习领域，我们似乎真的已经找到了那个魔法的锤子，能让一切都变成一根钉子【能够解决所有的问题】，它们就是Transformers。Transformers是一类模型，经过设计，它们能翻译文本，写诗，评论，甚至生成计算机代码。实际上，我在daleonai.com上写过的很多牛逼的研究都是基于Transformers的，比如AlphaFold 2这个模型，它能从基因序列预测蛋白质的结构。还有强大的自然语言处理模型，比如GPT-3，BERT，T5，Switch，Meena等等。你可能会说它们比表面上看到的还要...呃，算了。【more than meets the eyes是完整的句子，意译表示transformers确实比想象的更复杂】

​	如果你想在机器学习领域，尤其是NLP领域保持前沿，你至少得懂一丢丢Transformers。所以在这个帖子里，我们会说说它们到底是啥，它们咋工作的，以及它们为何这么牛逼的。

​	Transformer是一种神经网络架构。简单回顾下，在分析图片，视频，音频和文本这些复杂的数据类型时，神经网络是一种非常有效的模型。但针对不同的数据类型，也有相应优化的不同类型的神经网络。比如，要分析图片，我们一般用卷积神经网络CNN。简单来说，CNN能模仿人脑处理视觉信息的方式。



![image-20240725155640909](/Users/kj/Library/Application Support/typora-user-images/image-20240725155640909.png)

​	在2012年左右，我们已经能利用CNNs很好的解决一些视觉类的问题，比如识别照片里的物体，认识人脸，还有认识手写的数字。但很长一段时间以来，语言类任务中（翻译，文本归纳，文本生成，命名实体识别等等）一直没出现类似CNN般优秀的模型。不幸的是，语言是我们人类主要的交流方式。

​	在2017年Transformers出世之前，用来理解文本的深度学习技术是一种叫做循环神经网络RNN的模型，它长这样：

![image-20240725160529417](/Users/kj/Library/Application Support/typora-user-images/image-20240725160529417.png)

​	如果你想把一句英语翻译成法语，RNN就会把这句英文里的单词一个一个地吃进去，并且*按顺序*把相应的法语单词吐出来。敲重点，“按顺序”。在语言里，词语的顺序是很重要的，不能随机排列。举个例子，“Jane went looking for trouble”和“Trouble went looking for Jane”意思完全就不一样。

​	因此，要想理解语言，模型就必须能捕捉到语序。RNN按顺序一次处理一个单词，才能做到这点。

​	但是RNN也有问题。首先，遇到长段落或者整篇论文这类很长的文本时，RNN就够呛了。当处理到段落末尾的时候，它们就把开头给忘了，比如一个基于RNN的翻译模型可能就记不住一大段话里主角的性别。

​	更麻烦的是，RNNs特别难以训练。最烦人的就是，它们很容易就遇到梯度消失/爆炸的问题（有时候你只能重新开始训练然后祈祷别又爆炸了）。更令人头疼的是，因为它们是按顺序处理词语的，RNNs很难被并行化。这意味着，你没发通过增加GPU的数量来加速训练过程。换句话说，你压根没法使用大量的数据来训练它们。



## 进入Transformers的时代

​	Transformers终于横空出世了...2017年，为了做翻译，Google和多伦多大学的大佬们开发了Transformers。但不同于RNN，Transformers尤其适合并行化。这意味着，只要你有合适的硬件，你就能训练特别大的模型。

​	特别大是多大？

​	很大的大。

​	GPT-3，这个能写的像真人一样好的语言生成模型用了差不多45TB的文本数据训练，几乎囊括了整个公共互联网的文字。

​	你这么理解Transformers为啥如此厉害就行了：当一个扩展性很强的模型和一大堆数据合在一块的时候，它们产生的结果很可能惊呆你。



## Transformers是怎么工作的？

![Diagram of a Transformer](https://daleonai.com/images/screen-shot-2021-05-06-at-12.12.21-pm.png)

​	这是原论文中关于Transformer的架构图。看起来确实有点可怕，但Transformers的创新之处其实主要在三个方面：

1. Positional Encodings 位置编码
2. Attention 注意力
3. Self-Attention 自注意力

### 位置编码

​	先来看看第一个，位置编码。我们继续把英文翻译成法文。还记得RNN的做翻译的老方法吗---按顺序处理词语以理解语序，但这也让RNN很难并行化。

​	Transformers用位置编码来解决这个问题。思路是，给输入的英文语句里的每个单词，都配上它的位置的数字。然后你喂给神经网络的文本序列就会是这样：

```[("Dale",1),("says",2),("hello",3),("world",4)]```

​	这个方法，就是将理解语序这个任务，从神经网络身上移交到了数据本身。

​	初始状态下，Transformers还没被训练的时候，它并不知道如何理解位置编码。但是当模型见过越来越多的句子和相应的位置编码后，它就懂得如何用这些编码了。

​	在此我有点过于简化位置编码的工作原理了。原作者实际上用了sine函数生成位置编码，而不是一些简单的整数1，2，3，4--总之意思是一样的啦。将词语顺序存储为数据，而不是结构，神经网络就更容易训练了。



### 注意力

**下面，重要的一趴，Attention注意力。** 

懂？

​	注意力是一个神经网络的结构，现在只要说机器学习，你就能听到它。实际上呢，2017年那篇介绍Transformers的论文，标题不叫 “我们为您呈现Transformer”...而是，“注意力是你所需的一切（Attention is all you need）”。

​	注意力是在两年前的2015年在翻译领域被引入的。拿原论文中的例句来简单理解下：

​	*The agreement on the European Economic Area was signed in August 1992.*

​	ok，怎么把这句话翻译成法语？

​	*L’accord sur la zone économique européenne a été signé en août 1992.* 【我真不懂法语】

​	笨办法，把英文句子里的单词一个个过一遍，然后翻译成法语，一个单词一个单词来。这样翻出来不会好，有个原因就是，一些词语在法语和英语中语序是颠倒的：在英语中是“European Economic Area”，但法语里是这样“la zone économique européenne”。而且，法语单词分阴阳性，形容词“économique”和“européenne”必须用阴性形式以与阴性名词“la zone”匹配。

​	注意力机制，能让文本模型在决定如何翻译的时候关注到原句中的每一个单词。原论文里有个很好的图：

![image-20240725171024155](/Users/kj/Library/Application Support/typora-user-images/image-20240725171024155.png)

​	有点热力图，显示了模型在输出法语单词的时候在关注哪些英文单词。如你所想，当模型输出词语 “européenne”时，它非常关注输入词“European” 和 “Economic.”

​	模型咋知道每次它该关注哪些词呢？从训练数据里学到的。经过成千上万个法语和英语语句的训练，模型知道哪些类型的词语是相互依赖的。它学会了如何使用阴阳词性，单复数以及其他的语法规则。

​	注意力机制在2015年被建立以来，已经成为了NLP领域里一个非常有用的工具。但是在它最初的形式中，仍然是伴随着RNN一起使用的。因此，2017年的那篇Transformers论文有一部分的创新之处，就是完全舍弃了RNN。这也是为啥2017的论文叫做“Attenion is all you need”。



### 自注意力

​	最后，也是Transformer最牛逼的一点，是注意力机制的一种变体，叫做“Self-Attention自注意力”。

​	上面说的那类基本的注意力机制能帮助我们对齐英语和法语的单词，这对翻译任务很重要。但如果你不是为了单纯翻译词语，而是要建立一个能够理解一门语言背后的意义和模式的模型呢，一种能够完成任何语言任务的模型。

​	广泛的说，神经网络最强大最令人兴奋最酷的一点，就是它们通常会自动构建所训练数据的有意义的内部表示【从大量数据中找到数据内部的特征联系】。比如，仔细看一个视觉神经网络的各层级，你会发现一些神经元能“识别” 边缘，形状，甚至更高级的结构，例如眼睛和嘴巴。同理，一个文本模型经过训练也许能自动学习词性part of speech，语法规则，以及近义词。

​	总之，学到的内部表示（internal represenation）越好越完善，神经网络就能越好的完成任意语言任务。然而，如果将注意力机制用于输入文本自身，它会是个建立内部表示特别有效方法。

​	拿两句话举例：

​	“Server, can I have the check?” 【服务员，麻烦结账。】

​	“Looks like I just crashed the server.”【我刚好像把服务器搞崩了。】

​	“server”这个词语在这有俩非常不同的意思。人类通过句子中其他词语能够很容易分别出来。自注意力机制也能让神经网络通过语境来理解词义。

​	当模型处理第一句话中的“server”时，它可能会关注到“check”，从而将一个人（服务员）与金属（服务器）区分开来。

​	对于第二句话，模型可能关注“crashed”来理解“server”指代的是机器。

​	自注意力机制帮助神经网络辨别词义，进行词性标注，实体解析，学习语义角色的多种任务。【这些是NLP领域里的一系列任务】

​	总而言之，Transformers，三个重点：

1. 位置编码
2. 注意力
3. 自注意力

如果想要了解更深的技术细节，我强烈安利你去看看Jay Alammar的博客：[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

### Transformers能干啥？

​	有个特别流行的基于Transformer的模型叫做BERT（基于Transformer的双向编码器表示模型，Bidirectional Encoder Representation from Transformers）。这个模型也是Google的大佬们做的，在2018年我刚加入Google的那段时间，很快BERT就被使用在几乎所有的NLP项目里，包括谷歌搜索。

​	BERT不仅仅是个模型架构，同时也是一个训练好的模型。你可以[在这](https://github.com/google-research/bert)免费下载使用。谷歌的研究者们使用了海量的文字语料来训练它，它已经成为了NLP领域的一个多功能工具了。通过扩展，BERT可以解决一系列不同的问题，比如：

- 文本归纳 text summarization
- 问题回答 question answering
- 分类 classification
- 命名实体解析 named entity resolution
- 攻击性信息/脏话检测 offensive message/profanity detection
- 理解用户查询 understanding user query
- 还有更多....

​	BERT证明了你可以使用未标注的数据建立一些非常好的语言模型，比如用Wikipedia和Reddit上爬下来的文本，同时这些巨大的基础模型，经过具体细分领域中的数据调试后，可以应用于很多不同的场景。

​	最近，由OpenAI构建的GPT-3模型用它生成真实文本的能力惊艳了众人。 Meena，去年由Google研究院发布的基于Transformer的聊天机器人（aka.对话代理 conversational agent），它几乎可以就任何话题进行引人入胜的对话（这位作者曾经花了二十分钟与 Meena 讨论什么是人类）。

​	Transformers同时在NLP之外的领域产生了十足的影响，比如作曲，文生图片，还有预测蛋白质结构。

### 如何使用Transformers？

​	ok，现在你已经相信Transformers的力量了吧，你可能会想知道你怎样在自己的应用里使用它？冇问题。

​	你可以在[TensorFlow Hub](https://tfhub.dev/)下载一些常见的基于Transformer的模型，比如BERT。如果需要一些代码上的指导，可以看看这篇我的写的博客，是关于使用语义语言构建应用程序的。



​	如果你会真的想处于前沿并且你能写Python，我强烈安利由[HuggingFace](https://huggingface.co/)维护的非常流行的“Transformer”模型库。在这个平台上，你能训练并使用当下流行的NLP模型，比如BERT，Roberta，T5，GPT-2等等，非常适合开发者。

​	如果你想学更多如果使用Transformers构建应用，别离开！更多教程在路上了。

































