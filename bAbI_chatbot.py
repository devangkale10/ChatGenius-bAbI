# -*- coding: utf-8 -*-
"""
# Question and Answer Chat Bots

## Loading the Data

I'll be working with the Babi Data Set from Facebook Research.

Full Details: https://github.com/facebookarchive/bAbI-tasks?tab=readme-ov-file

- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698
"""

import pickle

import numpy as np

with open("train_qa.txt", "rb") as fp:  # Unpickling
    train_data = pickle.load(fp)

with open("test_qa.txt", "rb") as fp:  # Unpickling
    test_data = pickle.load(fp)

"""----

## Exploring the Format of the Data
"""

type(test_data)

type(train_data)

len(test_data)

len(train_data)

train_data[0]

" ".join(train_data[0][0])

" ".join(train_data[0][1])

train_data[0][2]

"""-----

## Setting up Vocabulary of All Words
"""

# Create a set that holds the vocab words
vocab = set()

all_data = test_data + train_data

for story, question, answer in all_data:
    # In case you don't know what a union of sets is:
    # https://www.programiz.com/python-programming/methods/set/union
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))

vocab.add("no")
vocab.add("yes")

vocab

vocab_len = (
    len(vocab) + 1
)  # we add an extra space to hold a 0 for Keras's pad_sequences

max_story_len = max([len(data[0]) for data in all_data])

max_story_len

max_question_len = max([len(data[1]) for data in all_data])

max_question_len

"""## Vectorizing the Data"""

vocab

# Reserve 0 for pad_sequences
vocab_size = len(vocab) + 1

"""-----------"""

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# integer encode sequences of words
tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

tokenizer.word_index

train_story_text = []
train_question_text = []
train_answers = []

for story, question, answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)

train_story_seq = tokenizer.texts_to_sequences(train_story_text)

len(train_story_text)

len(train_story_seq)

# word_index = tokenizer.word_index

"""### Functionalize Vectorization"""


def vectorize_stories(
    data,
    word_index=tokenizer.word_index,
    max_story_len=max_story_len,
    max_question_len=max_question_len,
):
    """
    INPUT:

    data: consisting of Stories,Queries,and Answers
    word_index: word index dictionary from tokenizer
    max_story_len: the length of the longest story (used for pad_sequences function)
    max_question_len: length of the longest question (used for pad_sequences function)


    OUTPUT:

    Vectorizes the stories,questions, and answers into padded sequences. We first loop for every story, query , and
    answer in the data. Then we convert the raw words to an word index value. Then we append each set to their appropriate
    output list. Then once we have converted the words to numbers, we pad the sequences so they are all of equal length.

    Returns this in the form of a tuple (X,Xq,Y) (padded based on max lengths)
    """

    # X = STORIES
    X = []
    # Xq = QUERY/QUESTION
    Xq = []
    # Y = CORRECT ANSWER
    Y = []

    for story, query, answer in data:

        # Grab the word index for every word in story
        x = [word_index[word.lower()] for word in story]
        # Grab the word index for every word in query
        xq = [word_index[word.lower()] for word in query]

        # Grab the Answers (either Yes/No so we don't need to use list comprehension here)
        # Index 0 is reserved so we're going to use + 1
        y = np.zeros(len(word_index) + 1)

        # Now that y is all zeros and we know its just Yes/No , we can use numpy logic to create this assignment
        #
        y[word_index[answer]] = 1

        # Append each set of story,query, and answer to their respective holding lists
        X.append(x)
        Xq.append(xq)
        Y.append(y)

    # Finally, pad the sequences based on their max length so the RNN can be trained on uniformly long sequences.

    # RETURN TUPLE FOR UNPACKING
    return (
        pad_sequences(X, maxlen=max_story_len),
        pad_sequences(Xq, maxlen=max_question_len),
        np.array(Y),
    )


inputs_train, queries_train, answers_train = vectorize_stories(train_data)

inputs_test, queries_test, answers_test = vectorize_stories(test_data)

inputs_test

queries_test

answers_test

sum(answers_test)

tokenizer.word_index["yes"]

tokenizer.word_index["no"]

"""## Creating the Model"""

from keras.layers import (
    LSTM,
    Activation,
    Dense,
    Dropout,
    Embedding,
    Input,
    Permute,
    add,
    concatenate,
    dot,
)
from keras.models import Model, Sequential

"""### Placeholders for Inputs

Recall we technically have two inputs, stories and questions. So we need to use placeholders. `Input()` is used to instantiate a Keras tensor.

"""

input_sequence = Input((max_story_len,))
question = Input((max_question_len,))

"""### Building the Networks

To understand why we chose this setup, make sure to read the paper we are using:

* Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895

## Encoders

### Input Encoder m
"""

# Input gets embedded to a sequence of vectors
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64))
input_encoder_m.add(Dropout(0.3))

# This encoder will output:
# (samples, story_maxlen, embedding_dim)

"""### Input Encoder c"""

# embed the input into a sequence of vectors of size query_maxlen
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3))
# output: (samples, story_maxlen, query_maxlen)

"""### Question Encoder"""

# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_question_len)
)
question_encoder.add(Dropout(0.3))
# output: (samples, query_maxlen, embedding_dim)

"""### Encode the Sequences"""

# encode input sequence and questions (which are indices)
# to sequences of dense vectors
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

"""##### Use dot product to compute the match between first input vector seq and the query"""

# shape: `(samples, story_maxlen, query_maxlen)`
match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation("softmax")(match)

"""#### Add this match matrix with the second input vector sequence"""

# add the match matrix with the second input vector sequence
response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

"""#### Concatenate"""

# concatenate the match matrix with the question vector sequence
answer = concatenate([response, question_encoded])

answer

# Reduce with RNN (LSTM)
answer = LSTM(32)(answer)  # (samples, 32)

# Regularization with Dropout
answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer)  # (samples, vocab_size)

# we output a probability distribution over the vocabulary
answer = Activation("softmax")(answer)

# build the final model
model = Model([input_sequence, question], answer)
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

model.summary()

# train
history = model.fit(
    [inputs_train, queries_train],
    answers_train,
    batch_size=32,
    epochs=100,
    validation_data=([inputs_test, queries_test], answers_test),
)

"""### Saving the Model"""

filename = "chatbot_10.h5"
model.save(filename)

"""## Evaluating the Model

### Plotting Out Training History
"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt

# %matplotlib inline
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

"""### Evaluating on Given Test Set"""

model.load_weights(filename)
pred_results = model.predict(([inputs_test, queries_test]))

test_data[0][0]

story = " ".join(word for word in test_data[0][0])
print(story)

query = " ".join(word for word in test_data[0][1])
print(query)

print("True Test Answer from Data is:", test_data[0][2])

# Generate prediction from model
val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

print("Predicted answer is: ", k)
print("Probability of certainty was: ", pred_results[0][val_max])

"""## Writing Your Own Stories and Questions

Remember you can only use words from the existing vocab
"""

vocab

# Note the whitespace of the periods
my_story = "John left the kitchen . Sandra dropped the football in the garden ."
my_story.split()

my_question = "Is the football in the garden ?"

my_question.split()

mydata = [(my_story.split(), my_question.split(), "yes")]

my_story, my_ques, my_ans = vectorize_stories(mydata)

pred_results = model.predict(([my_story, my_ques]))

# Generate prediction from model
val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

print("Predicted answer is: ", k)
print("Probability of certainty was: ", pred_results[0][val_max])
