# WinoGrande: An Adversarial Winograd Schema Challenge at Scale

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, Yejin Choi

(Submitted on 24 Jul 2019 (v1), last revised 21 Nov 2019 (this version, v2))

## Abstract
The Winograd Schema Challenge (WSC) (Levesque, Davis, and Morgenstern 2011), a benchmark for commonsense reasoning, is a set of 273 expert-crafted pronoun resolution problems originally designed to be unsolvable for statistical models that rely on selectional preferences or word associations. However, recent advances in neural language models have already reached around 90% accuracy on variants of WSC. This raises an important question whether these models have truy acquired robust commonsense capabilities or whether they rely on spurious bias in the datasets that lead to an overestimation of the true capabilities of machine commonsense. To investigate this question, we introduct WinoGrande, a large-scale dataset of 44k problems, inspired by the original WSC design, but adjusted to improve both the scale and the hardness of the dataset. The key steps of the dataset construction consist of (1) a carefully designed crowdsourcing procedure, followed by (2) systematic bias reduction using a novel AfLite algorithm that generalizes human-detectable word associations to machine-detectable embedding associations. The best state-of-the-art methods on WinoGrande achieve 59.4%-79.1%, which are 15-35% below human performance of 94.0%, depending on the amount of the training data allowed. Furthermore, we establish new state-of-the-art results on five related benchmarks - WSC (90.1%), DPR (93.1%), COPA (90.6%), KnowRef (85.6%), and Winogender (97.1%). These results have dual implications: on on hand, they demonstrate the effectiveness of WinoGrande when used as a resource for transfer learning. On the other hand, they raise a concern that we are likely to be overestimating the true capabilities of machine commonsense across all these benchmarks. We emphasize the importance of algorithmic bias reduction in existing and future benchmarks to mitigate such overestimation.

## Question About Potential Overestimation
Recent advances in neural language models have already reported around 90% accuracy on a variant of WSC dataset. This raises an important question: Have neural language models successfully acquired commonsense or are we overestimating the true capabilities of machine commonsense?

This question about the potential overestimation leads to another crucial question regarding potential unwanted biases that the large-scale neural language models might be exploiting, essentially solving the problems _right_, but for _wrong_ reasons.

While WSC questions are expert-crafted, recent studies have shown that they are nevertheless prone to incidental biases. Trichelair et al. (2018) have reported _word-association_ (13.5% of the cases) as well as other types of _dataset-specific_ biases. While such biases and annotation artifacts are not apparent for individual instances, they get introduced in the dataset as problem authors subconsciously repeat similar problem-crafting strategies.

## Algorithmic Data Bias Reduction
Several recent studies (Gururangan et al. 2018; Poliak et al. 2018; Tsuchiya 2018; Niven and Kao 2019l Geva, Goldberg, and Berant 2019) have reported the presence of _annotation artifacts_ in large-scale datasets. Annotation artifacts are unintentional patterns in the data that leak information about the target label in an undesired way. State-of-the-art neural models are highly effective at exploiting such artifacts to solve problems _correctly_, but for _incorrect_ reasons. To tackle this persistent challenge with dataset biases, we propose AFLITE - a novel algorithm that can systematically reduce biases using state-of-the-art contextual representation of words.

## Light-weight adversarial filtering
Our approach builds upon the adversarial filtering (AF) algorithm proposed by Zellers et al. (2018), but makes two key improvements: (1) AFLITE is much more broadly applicable (by not requiring over generation of data instances) and (2) it is considerably more lightweight (not requiring re-training a model at each iteration of AF). Overgenerating machine text from a language model to use in test instances runs the risk of distributional bias where a discriminator can learn to distinguish between machine generated instances and human-generated ones. In addition, AF depends on training a model at each iteration, which comes at extremely high computation cost when being adversarial to a model like BERT (Devlin et al. 2018).

## WinoGrande
WinoGrande is a new dataset with 44k problems that are inspired by the original design of WSC, but modified to improve both the scale and hardness of the problems. The key steps in WinoGrande construction consist of (1) a carefully designed crowdsourcing procedure, followed by (2) a novel algorithm AFLITE that generalizes human-detectable biases based on _word_ occurrences to machine-detectable biases based on _embedding_ occurences. The key motivation of our approach ist that it is difficult for humans to write problems without accidentally inserting unwanted biases.

## Performance
While humans find WINOGRANDE problems trivial with 94% accuracy, best state-of-the-art results, including those from RoBERTa (Liu et al. 2019) are considerably lower, ranging between 59.4%-79.1% depending on the amount of training data provided (from 800 to 41k instances), which falls 15-35% (absolute) below the human-level performance.

## Roberta
This variant aggregates the original WSC, PDP (Morgenstern, Davis, and Ortiz 2016) and additional PDP-style examples, and recasts them into True/False binary problems.
