
# Sarcasm detection

## Introduction
Sarcasm detection in text remains a challenging problem in natural language processing (NLP). Recognizing sarcasm involves understanding the subtle nuances and context-dependent nature of language, where the intended meaning often diverges from the literal expression.

In this project, the aim is to explore various NLP techniques and models to effectively detect sarcasm in textual data. By leveraging advancements in machine learning and deep learning, the goal is to develop a robust system capable of discerning sarcastic statements from non-sarcastic ones.

## Dataset

The dataset used for this project is the News Headlines Sarcasm Detection dataset from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection), containing over 26,000 examples of news headlines labeled as sarcastic or non-sarcastic. The data was divided into training, validation, and test sets in a 90%, 5%, and 5% split, respectively

## Preprocessing

The preprocessing phase involved converting raw text data into a format suitable for model training. Initially, the text was tokenized using the `TreebankWordTokenizer` from the [NLTK](https://www.nltk.org/) library. This tokenizer splits the text into individual words or tokens, handling punctuation and special characters appropriately to ensure accurate tokenization. After tokenization, the tokens were converted into numerical representations using one-hot encoding.

## Models

Every model used in this project is a probabilistic model that outputs the probability of an input being sarcastic. For any input $x$, the model outputs $\sigma(F(x))$, which represents the probability that $x$ is sarcastic, where $F(x)$ is the raw output of the network and $\sigma$ is sigmoid function.

The models were optimized for maximum log-likelihood of the data, leading to the following loss function:


$$\mathcal{L}(x_k, y_k) = y_k \log \left[ 1 + e^{-F(x_k)} \right] + (1-y_k) \left[ F(x_k) + \log \left(1 + e^{-F(x_k)}\right) \right]$$


### Vanilla LSTM

The vanilla Long Short-Term Memory (LSTM) network was chosen as a basic model. LSTMs are a type of Recurrent Neural Network (RNN) specifically designed to address the vanishing gradient problem, allowing them to learn long-range dependencies.



### RNN Search

RNN Search model is an attention-based architecture inspired by the work "[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)". This model enhances traditional Recurrent Neural Networks (RNNs) by incorporating an attention mechanism, allowing the model to focus on specific parts of the input sequence when making predictions.

In this architecture, the attention mechanism assigns a weight to each input token, indicating its relevance to the current output prediction. The RNN Search model then computes a context vector as a weighted sum of the input annotations, where the weights are determined by the attention scores. This context vector is then used to compute the output.

## Results

| Model | Accuracy     |
| :-------- | :------- |
| Vanilla LSTM | 85.2% |
| RNN Search | 87% |
