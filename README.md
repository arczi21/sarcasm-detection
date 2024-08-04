
# Sarcasm detection

## Introduction
Sarcasm detection in text remains a challenging problem in natural language processing (NLP). Recognizing sarcasm involves understanding the subtle nuances and context-dependent nature of language, where the intended meaning often diverges from the literal expression.

In this project, the aim is to explore various NLP techniques and models to effectively detect sarcasm in textual data. By leveraging advancements in machine learning and deep learning, the goal is to develop a robust system capable of discerning sarcastic statements from non-sarcastic ones.

## Dataset

The dataset used for this project is the News Headlines Sarcasm Detection dataset from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection), containing over 26,000 examples of news headlines labeled as sarcastic or non-sarcastic. The data was divided into training, validation, and test sets in a 90%, 5%, and 5% split, respectively

## Preprocessing

The preprocessing phase involved converting raw text data into a format suitable for model training. Initially, the text was tokenized using the `TreebankWordTokenizer` from the [NLTK](https://www.nltk.org/) library. This tokenizer splits the text into individual words or tokens, handling punctuation and special characters appropriately to ensure accurate tokenization. Following tokenization, embeddings for each token were learned.

## Models

Every model used in this project is a probabilistic model that outputs the probability of an input being sarcastic. For any input $x$, the model outputs $\sigma(F(x))$, which represents the probability that $x$ is sarcastic, where $F(x)$ is the raw output of the network and $\sigma$ is sigmoid function.

The models were optimized for maximum log-likelihood of the data, leading to the following loss function:


$$\mathcal{L}(x_k, y_k) = y_k \log \left[ 1 + e^{-F(x_k)} \right] + (1-y_k) \left[ F(x_k) + \log \left(1 + e^{-F(x_k)}\right) \right]$$


### Vanilla LSTM

The vanilla Long Short-Term Memory (LSTM) network was chosen as a basic model. LSTMs are a type of Recurrent Neural Network (RNN) specifically designed to address the vanishing gradient problem, allowing them to learn long-range dependencies.



### RNN Search

RNN Search model is an attention-based architecture inspired by the work "[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)". This model enhances traditional Recurrent Neural Networks (RNNs) by incorporating an attention mechanism, allowing the model to focus on specific parts of the input sequence when making predictions.

In this architecture, the attention mechanism assigns a weight to each input token, indicating its relevance to the current output prediction. The RNN Search model then computes a context vector as a weighted sum of the input annotations, where the weights are determined by the attention scores. This context vector is then used to compute the output.

### TransformerEncoder

The Transformer Encoder is a fundamental component of the Transformer architecture introduced in [Attention is All You Need](https://arxiv.org/abs/1706.03762). The Transformer Encoder leverages self-attention mechanisms to capture intricate dependencies and contextual information from the input text. The Transformer Encoder consists of a stack of identical layers, each containing two primary components:

**1.** Multi-Head Self-Attention Mechanism: This mechanism allows the model to attend to different parts of the input sequence simultaneously, capturing various aspects of contextual information. It computes attention scores for each token with respect to every other token in the sequence, enabling the model to weigh their relevance dynamically.

**2.** Position-Wise Feed-Forward Network: This component is applied to each position independently and identically. It consists of two linear transformations with a ReLU activation in between, enhancing the model's ability to learn complex patterns and interactions within the data.

Additionally, the Transformer Encoder incorporates residual connections and layer normalization to stabilize training and improve convergence.


### TransformerDecoder

The Transformer Decoder is another key component of the Transformer architecture introduced in Attention is All You Need. It is primarily designed for sequence generation tasks, utilizing self-attention and encoder-decoder attention mechanisms to generate output sequences. The Transformer Decoder consists of a stack of identical layers, each containing three main components:

**1.** Masked Multi-Head Self-Attention: This mechanism allows the model to attend to previous tokens in the output sequence while masking future tokens, ensuring that predictions are based solely on known information. It computes attention scores for each token with respect to every other token in the sequence up to the current position, dynamically weighing their relevance.

**2.** Encoder-Decoder Attention: This component attends to the encoder's output representations, integrating contextual information from the input sequence. For models using only the Transformer Decoder, this layer is adapted to focus on the input sequence itself, allowing the decoder to capture relevant information without a separate encoder.

**3.** Position-Wise Feed-Forward Network: Similar to the encoder, this component applies two linear transformations with a ReLU activation to each position independently, enhancing the model's ability to learn complex patterns and interactions within the data.

Additionally, the Transformer Decoder incorporates residual connections and layer normalization to stabilize training and improve convergence.

The Transformer Decoder has been successfully used in applications such as [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198) and [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf).

### TransformerDecoder with Pre-Training

The Transformer Decoder was also evaluated in a pre-trained configuration, inspired by the methodology described in the paper [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). In this approach, the Transformer Decoder was initially pre-trained on the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). This dataset, containing 50,000 movie reviews labeled as positive or negative, provides a rich source of textual data for learning general language representations.

After pre-training, the Transformer Decoder was fine-tuned on the sarcasm detection dataset. Fine-tuning involved training the model specifically on the task of detecting sarcasm in news headlines, leveraging the rich language representations learned during the pre-training phase. This two-step process of pre-training followed by fine-tuning helped the model achieve better performance by building on a solid foundation of general language understanding.

## Results

| Model | Accuracy     |
| :-------- | :------- |
| Vanilla LSTM | 85.2% |
| RNN Search | 87% |
| TransformerEncoder with MaxPooling | 85.3% |
| TransformerDecoder | 79% |
| TransformerDecoder (pre-trained) | 85% | 