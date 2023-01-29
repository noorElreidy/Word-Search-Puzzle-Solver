Run train.py and then evaluate.py. 
System.py is where the dimentionality reduction, classification and world finder is.

This code atttempts to solve one low-quality and one high-quality image of a word search puzzles.

Each letter in image is classified using a nearest neighbour classfier. Letters are stored in a 2D array 
after being classfied. Methof find_words goes through list of words to be found and attempts to find them.
find_words returns a list of start and end coordinate of each word in grid. If word is not found, 
(-1,-1,-1,-1) is returned for it. 

Letters are stored as a 30 x 30 array of pixels. They are reduced to a 20 dimensions vector using
PCA dimentionality reduction technique before being classified.