Old-fashion face detection algorithm, old-fashion, because the features are extracted by specific definition. This code as solution for ML competition of EESTEC challenge in Ankara 2017 Apr
 
Pre-requestes:

     Codes are in python 2.7
     the used libraries:
     PIL
     tensorflow
     OpenCV

how to run:
1- run feature-extraction.py in same folder as train data, to extract features
2- it creates out.csv file of features
3- run feature-extrac-from-testdata.py in directory of test images to extract test features
4- it creates test.csv file
5- run nn.py file to train on traindata features and crate txt file of predicted labels for test features
