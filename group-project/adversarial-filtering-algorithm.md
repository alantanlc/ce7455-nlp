# Implementation of the Adversarial Filtering Algorithm

The algorithm takes as input:
# _pre-computed_ embeddings __X__ 
# labels __y__
# size _n_ of the ensemble
# training size _m_ for the classifiers in the ensemble
# size _k_ of the filtering cutoff
# filtering threshold _t_

At each filtering phase, we train _n_ linear classifiers on different random partitions of the data and we collect their predictions on their corresponding validation set. For each instance, we compute its _score_ as the ratio of correct predictions over the total number of predictions. We rank the instances according to their score and remove the top-_k_ instances whose score is above threhold _t_. We repeat this process until we remove fewer than _k_ instances in a filtering phase or there are fewer than _m_ remaining instance. When applying AFLite to WinoGrande, we set _m_ = 10,000, _n_ = 64, _k_ = 500, and _t_ = 0.75.
