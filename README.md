# CE7455: Deep Learning for Natural Language Processing: From Theory to Practice

## Course Objectives
Natural Language Processing (NLP) is one of the the most important fields in Artificial Intelligence (AI). It has become very crucial in the information age because most of the information is in the form of unstructured text. NLP technologies are applied everywhere as people communicate mostly in language: language translation, web search, customer support, emails, forums, advertisement, radiology reports, to name a few.

There are a number of core NLP tasks and machine learning models behind NLP applications. Deep learning, a sub-field of machine learning, has recently brought a paradigm shift from traditional task-soecific feature engineering to end-to-end systems and has obtained high performance across many different NLP tasks and downstream applications. Tech companies like Google, Baidu, Alibaba, Apple Amazon, Facebook, Tencent, and Microsoft are now actively working on deep learning methods to improve their products. For example, Google recently replaced its traditional statistical machine translation and speech-recognition systems with systems based on deep learning methods.

### Optional Textbooks
- Deep Learning by Goodfellow, Bengio, and Courville [free online](http://www.deeplearningbook.org/)
- Machine Learning - A Probabilistic Perspective by Kevin Murphy [online](https://doc.lagout.org/science/Artificial%20Intelligence/Machine%20learning/Machine%20Learning_%20A%20Probabilistic%20Perspective%20%5BMurphy%202012-08-24%5D.pdf)
- Natural Language Processing by Jacob Eisenstien [free online](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)
- Speech and Language Processing by Dan Jurafsky and James H. Martin [(3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/)

## Intended Learning Outcome
In this course, students will learn state-of-the-art deep learning methods for NLP. Through lectures and practical assignments, students will learn the necessary tricks for making their models work on practical problems. They will learn to implement, and possibly to invent their own deep learning models using available deep learning libraries like [Pytorch](https://pytorch.org/).

### Our Approach
- _Thorough and Detailed_: How to write from scratch, debug and train deep neural models
- _State of the art_: Most lecture materials are new from research world in the past 1-5 years.
- _Practical_: Focus on practical techniques for training the models, and on GPUs.
- _Fun_: Cover exciting new advancements in NLP (e.g., Transformer, BERT).

## Assessment Approach
### Weekly Workload
- Every two-hour lecture will be accompanied by practice problems implemented in PyTorch.
- There will be a 30-min office per week to discuss assignments and project.
- There will be `5%` marks for class participation.

### Assignments (individually graded)
- There will be three (3) assignments contributing to `3 * 15% = 45%` of the total assessment.
- Late day policy
  - 2 free late days; afterwards, `10%` off per day late
  - Not accepted after 3 late days
- Students will be graded individually on the assignments. They will be allowed to discuss with each other on the homework assignments, but they are required to submit individual write-ups and coding exercises.

### Final Project (Group work but individually graded)
- There will be a final project contributing to the remaining 50% of the total course-work assessment.
  - `1-3` people per group
  - Project proposal: `5%`, updated: `5%`, presentation: `10%`, report: `30%`
- The project will be a group or individual work depending on the student's preference. Students will be graded individually. The final project presentation will ensure the student's understanding of the project

## Course Prerequisites
- Proficieny in Python (using numpy and PyTorch). There is a lecture for those who are not familiar with Python.
- College Calculus, Linear Algebra
- Basic Probability and Statistics
- Machine Learning basics

## Teaching
__Instructor__:
- Shafiq Rayhan Joty

__Teaching Assistants__:
- Tasnim Mohuiddin
- Nguyen Thanh Tung
- Xuan Phi Nguyen
- Hancheol Moon
- Lin Xiang

## Schedule & Course Content

### Week 1: Introduction [[Lecture Slides]](https://www.dropbox.com/s/3o9zo7yljht0g4b/Lecture-1.pdf?dl=0)
__Lecture Content__
- What is Natural Language Processing?
- Why is language understanding difficult?
- What is Deep Learning?
- Deep learning vs. other machine learning methods?
- Why deep learning for NLP?
- Applications of deep learning to NLP
- Knowing the target group (background, field of study, programming experience)
- Expectation from the course
__Python & PyTorch Basic__
- Programming in Python
  - Jupyter Notebook and [google colab](https://colab.research.google.com/drive/16pBJQePbqkz3QFV54L4NIkOn1kwpuRrj
  - [Introduction to python](https://colab.research.google.com/drive/1bQG32CFoMZ-jBk02uaFon60tER3yFx4c)
  - Deep Learning Frameworks
  - Why Pytorch?
  - [Deep learning with PyTorch](https://drive.google.com/file/d/1c33y8bkdr7SJ_I8-wmqTAhld-y7KcspA/view)
- [Supplementary]
  - Numerical programming with numpy/scipy - [Numpy intro](https://drive.google.com/file/d/1cUzRzQGURrCKes8XynvTTA4Zvl_gUJdc/view)
  - Numerical programmig with Pytorch - [Pytorch intro](https://drive.google.com/file/d/18cgPOj2QKQN0WR9_vXoz6BoravvS9mTm/view)
  
### Week 2: Machine Learning Basic [[Lecture Slides]](https://www.dropbox.com/s/hzizorgpwsbgi58/Lecture-2.pdf?dl=0)
__Lecture Content__
- What is Machine Learning?
- Supervised vs. unsupervised learning
- Linear regression
- Logistic regression
- Multi-class classification
- Parameter estimation (MLE & MAP)
- Gradient-based optimization & SGD
__Practical exercise with Pytorch__
- [Deep learning with PyTorch](https://colab.research.google.com/drive/1tRayE1KjvmENZJe9oGwtnRNw_sbrwb5d)
- [Linear Regression](https://colab.research.google.com/drive/1krekWlJPHjvxH6fMzjYohR-1OX43bp67)
- [Logistic Regression](https://colab.research.google.com/drive/1rpvMmkYgU3LWnmsG7xaowDhAdJCWFzOU)
- [Supplementary]
  - Numerical programming with Pytorch - [Pytorch intro](https://drive.google.com/file/d/18cgPOj2QKQN0WR9_vXoz6BoravvS9mTm/view)

### Week 3: Neural Network & Optimization Basics [[Lecture Slides]](https://www.dropbox.com/s/bqlf5kmvr2mcyjd/Lecture-3.pdf?dl=0)
__Lecture Content__
- Why Deep Learning for NLP?
- From Logistic Regression to Feed-forward NN
  - Activation functions
- SGD with Backpropagation
- Adaptive SGD (Adagrad, adam, RMSProp)
- Regularization (Weight Decay, Dropout, Batch normalization, Gradient clipping)
- Introduction to Word Vectors
__Assignment `1` out__

__Practical exercise with Pytorch__

[Numpy notebook](https://colab.research.google.com/drive/1IAonxZnZjJb0_xUVWHt5atIxaI5GTJQ2#scrollTo=IuC1D60M82lg) [Pytorch notebook](https://colab.research.google.com/drive/1YzZrMAmJ3hjvJfNIdGxae9kxGABG6yaT)
- Backpropagation
- Dropout
- Batch normalization
- Initialization
- Gradient clipping
__Suggessted Readings__
- SGD optimization [blog](https://ruder.io/optimizing-gradient-descent/)
- [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747)
- [Adam: a method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf)
- [Neural Optimizer Search with Reinforcement Learning](http://proceedings.mlr.press/v70/bello17a.html)

### Week 4: Word Vectors [[Lecture Slides]](https://www.dropbox.com/s/2m8pyvjfs5m6zpd/Lecture-4.pdf?dl=0)
__Lecture Content__
- Word meaning
- Denotational semantics
- Distributed representation of words
- Word2Vec models (Skip-gram, CBOW)
- Negative sampling
- Glove
- FastText
- Evaluating word vectors
  - Intrinsic evaluation
  - Extrinsic evaluation
- Cross-lingual word vectors
__Practical exercise with Pytorch__
- [Skip-gram training](https://colab.research.google.com/drive/164dB-Vemzwavf1ffqDDVNtx7Y5VtcmQh)
- Visualization
__Suggested Readings__
- Word2Vec Tutorial - The Skip-Gram Model [blog](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [Efficient Estimation of Word Represenations in Vector Space](https://arxiv.org/abs/1301.3781 - Original word2vec paper
- [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) - negative sampling paper
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
- [FastText: Enriching Word Vectors with Subword Information](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00051?mobileUi=0&)
- [Linguistic Regularities in Sparse and Explicit Word Representations.](https://levyomer.files.wordpress.com/2014/04/linguistic-regularities-in-sparse-and-explicit-word-representations-conll-2014.pdf)
- [Neural Word Embeddings as Implicit Matrix Factorization.](https://levyomer.files.wordpress.com/2014/09/neural-word-embeddings-as-implicit-matrix-factorization.pdf)
- [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://www.aclweb.org/anthology/Q15-1016/)
- [Survey on Cross-lingual embedding methods](https://arxiv.org/abs/1706.04902)
- [Slides on Cross-lingual embedding](https://www.dropbox.com/s/3eq5apr75yrz9ix/Cross-lingual%20word%20embeddings%20and%20beyond.pdf?dl=0)
- [Adversarial autoencoder for unsupervised word translation](https://arxiv.org/abs/1904.04116)
- [Evaluating Cross-Lingual Word Embeddings](https://www.aclweb.org/anthology/P19-1070/)
- [Linear Algebraic Structure of Word Senses, with Applications to Polsemy](https://transacl.org/ojs/index.php/tacl/article/viewFile/1346/320)

### Week 5: Cross-lingual Word Vectors & CNNs [[Lecture Slides]](https://www.dropbox.com/s/at3wztn2dn7h2k2/Lecture-5.pdf?dl=0) [[Slides with recording]](https://drive.google.com/open?id=18hsT0Pm3yGGliCCmoapW7XenEIEzP51L) [[Slides with video]](https://drive.google.com/file/d/1PU-8asYE4LdfBRbKa-CPD-b2gip0k7pQ/view)
__Lecture Content__
- Cross-lingual word embeddings
- Classification tasks in NLP
- Window-based Approach for language modelling
- Window-based Approach for NER, POS tagging, and Chunking
- Convolutional Neural Net for NLP
- Max-margin Training
- Scaling Softmax (Adaptive input & output)
__Assignment `1` in__

__Invited talk on cross-lingual word vectors__
- [Tasnim Modiuddin](https://taasnim.github.io/)
- [Talk Slides](https://www.dropbox.com/s/al987q6ltv3zpfv/word-tr-Tasnim.pdf?dl=0)
__Practical exercise with Pytorch__
- [CNN for word encoding](https://github.com/FengZiYjun/CharLM)
__Suggested Readings__
- [Natural Language Processing (Almost) from Scratch](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615)
- [Fast and Accurate Entity Recognition with Iterated Dilated Convolutions](https://arxiv.org/abs/1702.02098)
- [Adaptive Softmax Paper](https://arxiv.org/abs/1609.04309)
- [Adaptive Input representation paper](https://openreview.net/pdf?id=ByxZX20qFQ)
- [KNN-LM paper](https://openreview.net/forum?id=HklBjCEKvH)

### Week 6: Recurrent Neural Nets [[Lecture Slides]](#)
__Lecture Content__
- Basic RNN structures
- Language modeling with RNNs
- Backpropagation through time
- Text generation with RNN LM
- Issues with Vanilla RNNs
- Exploding gradient
- Gated Recurrent Units (GRUs) and LSTMs
- Bidirectional RNNs
- Multi-layer RNNs
- Sequence labeling with RNNs
- Sequence classification with RNNs
__Assignment `2` out__

__Practical exercise with Pytorch__
- Opinion analysis
- Part-of-speech (POS) tagging
- Named Entity Recognition (NER)
- Sentiment classification
- Text generation
__Suggested Readings__
- [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
- [Karpathy's nice blog on Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Building an Efficient Neural Language Model](https://research.fb.com/building-an-efficient-neural-language-model-over-a-billion-words/)
- [On the difficulty of training recurrent neural networks](http://proceedings.mlr.press/v28/pascanu13.pdf)
- [Colah's blog on LSTMs/GRUs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Neural Architectures for Named Entity Recognition](https://www.aclweb.org/anthology/N16-1030/)
- [Fine-grained Opinion Mining with Recurrent Neural Networks and Word Embeddings](https://www.aclweb.org/anthology/D15-1168/)
- [Zero-Resource Cross-Lingual NER](https://arxiv.org/abs/1911.09812)
