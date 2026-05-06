# Feasibility Study of Representational Redundancy

The very first step of this project is to evaluate if there is redundancy present in the output projection matrices $W_O$ of some models. If the $W_O$ are low rank, this means that it is doing alot of redundant work. In particular, for comparison purposes I explore the BERT (large), GTP2 and the focus of this investigation DeepSeek V2 (Lite).

The second strategy is to explore the concatenated output projections associated activations, in this case the concatenated output attention heads. In order to do this a batch of texts needs to be passed through each model (BERT (large), GTP2, DeepSeek V2 (Lite)), hooks are added to capture the concatenated output attention heads activations. These activations are reshaped across each token (-1, hidden_size) and SVD is applied to these flattened data, with the aim to identify if any of the 2048 dimensions are redundant.

Finally, I will explore the internal head redundancy and inter-head redundancy for each of the models (BERT (large), GTP2, DeepSeek V2 (Lite)).

Overall, it would appear that redundancy is a common trend among these models, with varying levels of dimension redundancy seen across the weights and activations for the three models. Interestingly, we can see that Bert Large and DeepSeek V2 Lite have similar levels of activation effective rank across the different layers for a supplied batch of wikitext-103-raw-v1. However, this could be dataset dependent given that DeepSeek V2 Lite was trained an much more varied and diverse set of data. However, the signal is there for me to continue my research into applying some form of compression/decompression to the output projections. As such I will next train some models and compare there results.
