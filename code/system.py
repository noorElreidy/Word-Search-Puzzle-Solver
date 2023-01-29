"""Dummy classification system.

Dummy solution the COM2004/3004 assignment.

REWRITE THE FUNCTIONS BELOW AND REWRITE THIS DOCSTRING

version: v1.0
"""

from typing import List

import numpy as np
from utils import utils
from utils.utils import Puzzle
import scipy.linalg

# The required maximum number of dimensions for the feature vectors.
N_DIMENSIONS = 20
DIRECTIONS = [[-1, 0], [1, 0], [1, 1],[1, -1], [-1, -1], [-1, 1], [0, 1], [0, -1]]


def load_puzzle_feature_vectors(image_dir: str, puzzles: List[Puzzle]) -> np.ndarray:
    """Extract raw feature vectors for each puzzle from images in the image_dir.

    OPTIONAL: ONLY REWRITE THIS FUNCTION IF YOU WANT TO REPLACE THE DEFAULT IMPLEMENTATION

    The raw feature vectors are just the pixel values of the images stored
    as vectors row by row. The code does a little bit of work to center the
    image region on the character and crop it to remove some of the background.

    You are free to replace this function with your own implementation but
    the implementation being called from utils.py should work fine. Look at
    the code in utils.py if you are interested to see how it works. Note, this
    will return feature vectors with more than 20 dimensions so you will
    still need to implement a suitable feature reduction method.

    Args:
        image_dir (str): Name of the directory where the puzzle images are stored.
        puzzle (dict): Puzzle metadata providing name and size of each puzzle.

    Returns:
        np.ndarray: The raw data matrix, i.e. rows of feature vectors.

    """
    return utils.load_puzzle_feature_vectors(image_dir, puzzles)


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """ This methods implements PCA dimentionality reduction. It
        reduces data to 20 dimentions.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.

    """
    training_data = model["training_data"]

    tr = np.array(training_data)
    covx = np.cov(tr, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N-N_DIMENSIONS, N-1))
    v = np.fliplr(v)
    v.shape
    reduced_data  = np.dot((data - np.mean(data)), v)

    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """ Processes training data by and stores it into model.
        Stores labels_train as model["labels_train"]
        Stores fvectors_train as model["training_data"]
        Stores reduced training data as model["fvevtors_train"]

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """


    model = {}
    model["labels_train"] = labels_train.tolist()
    model["training_data"] = fvectors_train.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """ Implements a nearest neighbour classifier. 

    Args:
        fvectors_train (np.ndarray): feature vectors that are to be classified, stored as rows.
        model (dict): a dictionary storing all the model parameters needed by your classifier.

    Returns:
        List[str]: A list of classifier labels, i.e. one label per input feature vector.
    """


    train_vectors = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])


    # Super compact implementation of nearest neighbour
    x = np.dot(fvectors_test, train_vectors.transpose())
    modtest = np.sqrt(np.sum(fvectors_test * fvectors_test, axis=1))
    modtrain = np.sqrt(np.sum(train_vectors * train_vectors, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())  # cosine distance
    nearest = np.argmax(dist, axis=1)
    labels = labels_train[nearest]
    return labels


def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:

    """ Finds the words passed in the words array within the labels array.
        Searches in all directions in labels array and starts allowing some
        letters to be wrong if word isn't found.
    Args:
        labels (np.ndarray): 2-D array storing the character in each
            square of the wordsearch puzzle.
        words (list[str]): A list of words to find in the wordsearch puzzle.
        model (dict): The model parameters learned during training.

    Returns:
        list[tuple]: A list of four-element tuples indicating the word positions.
    """
    lis = []
    #print(labels)
    for word in words:
        pos = find_one_word(labels, word.upper(), False,0)
        if (pos[0] == -1):
            wrong_allowed = 1
            while((pos[0] == -1) and  (wrong_allowed-1 < len(word)/2)):
                pos = find_one_word(labels, word.upper(),True,wrong_allowed)
                wrong_allowed += 1
            lis.append(pos)
            #print(pos)
        else :
            lis.append(pos)

    return lis

def find_one_word_one_point(labels: np.ndarray, word, row, column):

    """ Function that searches for a single word within the 2-D array from a single
        starting point and into all direction. Doesn't allow wrong letters

        Args:
            labels (np.ndarray): 2-D array storing the character in each
                square of the wordsearch puzzle.
            word : a word to find .
            model (dict): The model parameters learned during training.

        Returns:
            a tuple : (start letter x, start letter y, end letter x, end letter y)
            returns (-1,-1,-1,-1) if not found
    """
    pos = [-1,-1,-1,-1]
    maxRow = len(labels)
    maxCol = len(labels[0])
    wrong_count = 0

    #already not word if first character doesnt match
    if (labels[row][column] != word[0]):
        return (-1,-1,-1,-1)

    pos = [row,column,-1,-1]

    #checks all directions
    for x, y in DIRECTIONS:

        this_direction = True
        row_direction = row + x
        column_direction =  column + y

        for k in range(1, len(word)):
            #if within labels bounds and character matches
            if ((0 <= row_direction < maxRow) and (0 <= column_direction <  maxCol) and word[k] == labels[row_direction][column_direction]):
                    pos[2] = row_direction
                    pos[3] = column_direction
                    column_direction += y
                    row_direction += x

            else:
                this_direction = False
                break

        if (this_direction):
            pos2 = (pos[0],pos[1],pos[2],pos[3])
            return pos2
    return (-1,-1,-1,-1)

def find_one_word_one_point_mis(labels: np.ndarray, word, row, column,wrong_allowed):
    """ Function that searches for a single word within the 2-D array from a single
        starting point and into all direction

        Args:
            labels (np.ndarray): 2-D array storing the character in each
                square of the wordsearch puzzle.
            word : a word to find .
            model (dict): The model parameters learned during training.

        Returns:
            a tuple : (start letter x, start letter y, end letter x, end letter y)
            returns (-1,-1,-1,-1) if not found
    """
    pos = [-1,-1,-1,-1]
    maxRow = len(labels)
    maxCol = len(labels[0])
    wrong_count = 0
    wrong_ok = wrong_allowed
    wrong_first = 0

    if (labels[row][column] != word[0]):
        wrong_count += 1
        wrong_first +=1

    pos = [row,column,-1,-1]

    #checks all directions
    for x, y in DIRECTIONS:

        this_direction = True
        row_direction = row + x
        column_direction =  column + y
        wrong_count = wrong_first #resets before trying another direction

        for k in range(1, len(word)):
            #checks if within bounds of labels array
            if ((0 <= row_direction < maxRow) and (0 <= column_direction <  maxCol)) :
                if word[k] == labels[row_direction][column_direction] :
                    #sets end coordinate and moves to next coordinte in same direction
                    pos[2] = row_direction
                    row_direction += x
                    pos[3] = column_direction
                    column_direction += y
                elif  wrong_count < wrong_ok :
                    #does same as first condition but increments wrong count
                    wrong_count += 1
                    pos[2] = row_direction
                    row_direction += x
                    pos[3] = column_direction
                    column_direction += y
                else :
                    this_direction = False
                    break

            else:
                this_direction = False
                break

        if (this_direction):
            #turns into tuple. Needed array to be able to change it
            pos2 = (pos[0],pos[1],pos[2],pos[3])
            return pos2
    return (-1,-1,-1,-1)


def find_one_word(labels: np.ndarray, word, allow,wrong_allowed):

    """ Function that loops through entire word search labels array
        searching for word. Calls find_one_word_one_point or
        find_one_word_one_point_mis for each point in labels.

        Args :
             labels :  2-D array storing the character in each
                square of the wordsearch puzzle.
             word : the word to find .
        Returns :
             a tuple : the words position -> (start x, start y, end x, end y)
             returns (-1,-1,-1,-1) if word isn't found
    """
    #calls method that allows wrong letters if allow is true
    if allow :
        for i in range (len(labels)):
            for j in range (len(labels[0])):
                pos = find_one_word_one_point_mis(labels,word,i,j,wrong_allowed)
                if (pos[0]!= -1):
                    return pos
    else :
        for i in range (len(labels)):
            for j in range (len(labels[0])):
                pos = find_one_word_one_point(labels,word,i,j)
                if (pos[0]!= -1):
                    return pos


    return (-1,-1,-1,-1)

