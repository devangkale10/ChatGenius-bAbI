# ChatGenius-bAbI

An innovative question-and-answer chatbot developed using the cutting-edge bAbI dataset from Meta.

This is an innovative question-and-answer chatbot developed using the bAbI dataset from Meta. This repository houses a sophisticated AI model designed to understand and respond to a wide array of questions with remarkable accuracy. Leveraging the structured yet diverse nature of the bAbI dataset, this chatbot offers an interactive learning and querying experience that mimics human-like conversation patterns.

## Features

### Why chat with ChatGenius-bAbI?

- **He Understands:**: ChatGenius-bAbI is powered by an NLP model that can understand and respond to a wide array of questions with remarkable accuracy.
- **Chat Away**: From wondering about the weather to unraveling riddles, this chatbot loves a good chat across any topic you can think of.
- **Interactive Learning**: ChatGenius-bAbI is designed to learn from every interaction, making it smarter and more intuitive with every conversation.

## Model Architecture

The model takes a discrete set of inputs x1, x2, ..., xT that are stored in memory and a query q. The model processes the inputs and the query to produce an answer a. The model is trained end-to-end with backpropagation.
Each of the x, q, a contain symbols coming from a dictionary with V words

### End to End Memory Networks

- **Input Memory Representation**: The input memory is represented as a sequence of vectors x1, x2, ..., xT, converted into memory vectors m1, m2, ..., mT.
- **Output Memory Representation**: Each x has a corresponding output vector c.
- **The Output**: The sum of the output vector o and the input embedding u is passed through a final weight matrix and a softmax to product predicted label a.
