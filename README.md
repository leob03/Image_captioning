# Neural Image Caption Generation with Visual Attention
Implementation of neural network model that can generate natural language captions for images. Three different architectures are proposed and compared: first one uses vanilla recurrent neural networks (RNNs), second one long-short term memory networks (LSTMs), and third one attention-based LSTMs. 

# Contents

[***Objective***](https://github.com/leob03/Image_captionning#objective)

[***Concepts***](https://github.com/leob03/Image_captionning#concepts)

[***Overview***](https://github.com/leob03/Image_captionning#overview)

[***Dependencies***](https://github.com/leob03/Image_captionning#dependencies)

[***Getting started***](https://github.com/leob03/Image_captionning#getting-started)

# Objective

**To build a model that can generate a descriptive caption for an image we provide it.**

In this project, we implemented a vanilla recurrent neural networks (RNNs), [long-short term memory networks (LSTMs)](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory), and [attention-based LSTMs](https://arxiv.org/abs/1409.0473) to train a model that can generate natural language captions for images.

Models in this exercise are highly similar to very early works in neural-network based image captioning. If you are interested to learn more, check out these two papers:

1. [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
2. [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)

The Attention model learns _where_ to look.

As you generate a caption, word by word, you can see the model's gaze shifting across the image.

This is possible because of its _Attention_ mechanism, which allows it to focus on the part of the image most relevant to the word it is going to utter next.

Here is a caption generated example:


---

![](./img/babycake.png)

---


# Concepts

* **Encoder-Decoder architecture**. Typically, a model that generates sequences will use an Encoder to encode the input into a fixed form and a Decoder to decode it, word by word, into a sequence.

* **Attention**. The use of Attention networks is widespread in deep learning, and with good reason. This is a way for a model to choose only those parts of the encoding that it thinks is relevant to the task at hand. The same mechanism you see employed here can be used in any model where the Encoder's output has multiple points in space or time. In image captioning, you consider some pixels more important than others. In sequence to sequence tasks like machine translation, you consider some words more important than others.

* **Transfer Learning**. This is when you borrow from an existing model by using parts of it in a new model. This is almost always better than training a new model from scratch (i.e., knowing nothing). As you will see, you can always fine-tune this second-hand knowledge to the specific task at hand. Using pretrained word embeddings is a dumb but valid example. For our image captioning problem, we will use a pretrained Encoder, and then fine-tune it as needed.

# Overview

The pipeline for the project looks as follows:

- The **input** is a dataset of images and 5 sentence descriptions that were collected with Amazon Mechanical Turk. We will use the 2014 release of the [COCO Captions dataset](http://cocodataset.org/) which has become the standard testbed for image captioning. The dataset consists of 80,000 training images and 40,000 validation images, each annotated with 5 captions.
- In the **training stage**, the images are fed as input to RNN (or LSTM/LSTM with attention depending on the model) and the RNN is asked to predict the words of the sentence, conditioned on the current word and previous context as mediated by the hidden layers of the neural network. In this stage, the parameters of the networks are trained with backpropagation.
- In the **prediction stage**, a witheld set of images is passed to RNN and the RNN generates the sentence one word at a time. The code also includes utilities for visualizing the results.

# Dependencies
**Python 3.10**, modern version of **PyTorch**, **numpy** and **scipy** module. Most of these are okay to install with **pip**. To install all dependencies at once, run the command `pip install -r requirements.txt`

I only tested this code with Ubuntu 20.04, but I tried to make it as generic as possible (e.g. use of **os** module for file system interactions etc. So it might work on Windows and Mac relatively easily.)


# Getting started

1. **Get the code.** `$ git clone` the repo and install the Python dependencies
2. **Get the data.** We don't distribute the data in the Git repo, instead download the `data/` folder from [here]([http://web.eecs.umich.edu/~justincj/teaching/eecs498/coco.pt]). Also, this download does not include the raw image files, so if you want to visualize the annotations on raw images, you have to obtain the images from Flickr8K / Flickr30K / COCO directly and dump them into the appropriate data folder.
3. **Train the models.** Run the training `$ python train_rnn.py` or `$ python train_lstm.py` or `$ python train_lstm_attention.py`, depending on the model that you want to try (see many additional argument settings inside the file) and wait. You'll see that the learning code writes checkpoints into `cv/` and periodically print its status. 
4. **Evaluate the models checkpoints and Visualize the predictions.** To evaluate a checkpoint from `checkpoints/`, run the scripts `$ python test_rnn.py` or `$ python test_lstm.py` or `$ python test_lstm_attention.py` and pass it the path to a checkpoint ( by adding --checkpoint /path/to/the/checkpoint after your python command).


