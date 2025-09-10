# AI_Algos

Test of various AI techniques and models.

## Yolo

YoloV11 use on two various videos for object detection though a video.
Test of training my own Yolo code on data I've labbelled. 

## Whisper

Script that I use to synthetize quicker book that i've read and on which I want to have notes on.

## Decision Tree

Decision tree algorithm implementation to detect, based on input, pattern of restaurant choice (exemple base on 'AI : Modern Approach' book)


# Andrej Kharparty's tutorial

Tutorials implementation to understand the various concepts of AI, from bag of words to GPT basic implementation.

## BackPropagation - Micrograd

Implementation of MultiLayerPerceptron (MLP) to copy XOR function. Based on 
Andrej Karpathy's micrograd online tutorial (https://www.youtube.com/watch?v=VMj-3S1tku0&t)

## MakeMore

### Part One
https://www.youtube.com/watch?v=PaCmpygFfXo \
\
Implementation of bigram, which is a character prediction model based on probability encountered in dataset. Two solutions are provided:\
\
Solution 1
* Construction of a matrix of duo's of character based on dataset ("ab has been met 2500 times, ac 750 times..").
* Usage of torch.multinominal to create normal distribution upon these duo's
* New name generation by randomness based on computed probabilities

\
Solution 2
* Single layer of 27 neurons [27, 27] with gradient descent to compare results
* After few iterations, an approximatively equal loss has been archived

### Part Two
https://www.youtube.com/watch?v=TCH_1BHY58I \
\
Implementation of the MLP from Bengio, al. paper. Instead of word generation based on previous words in context, the current solution still generates next character on contextual previous character. 
* Creation of an embedding of two for each character [27, 2].
* Embedding of each contextual character met in the provided dataset [X, 3 , 2]. 3 being the number of characters used for next character prediction, 2 being the emebedding size and X the total amount of contextual character met. This number cannot be unique because contextual character of 3 has to be put in correlation with an output (true next character from dataset).
* Creation of a tanh layer processing. Each generated tuple is transformed to be processed [X, 3 * 2 ] and is input into layer.
* OutputLayer of size [Y, 27]. Y for the defined tanh layer and 27 for corresponding to the next word generation.
* backward propagation into the aforementioned architecture.
* Computer loss below the Bigram solution (the next character prediction is closer to the truth).\
Some optimizations:
* Utilization of batch for processing gradient instead of the whole. (More iteration on a less precise gradient but still true at the begining phases are better than perfect gradient with heavy computing processes)
* Could use cross_entropy and tensor.view to reduce the number of code.

### Interlude

Proposition of selection of best model based on hyperparameters optimization.

