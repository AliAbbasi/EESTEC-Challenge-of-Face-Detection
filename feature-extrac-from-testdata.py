import  glob
from  PIL  import  Image 
import cv2
import sys
import numpy as np

#----------------------------------------------------------------

def w_b(input): 
    input = np.reshape(input , (input.shape[0] * input.shape[1])) 
    b = 0
    w = 0
    for i in input:
        if i <= 127:
            b += i
        else:
            w += i 
    return w - b
    
#----------------------------------------------------------------

def filters(img):
    arr=[]
    
    kernel1 =np.array([[ 1,  1,  1,  1],
                       [ 1,  1,  1,  1],
                       [-1, -1, -1, -1],
                       [-1, -1, -1, -1]])  
    dst1 = cv2.filter2D(img, -1, kernel1) 
    
    
    kernel2 =np.array([[1, 1, -1, -1],
                       [1, 1, -1, -1],
                       [1, 1, -1, -1],
                       [1, 1, -1, -1]])  
    dst2 = cv2.filter2D(img, -1, kernel2) 
    
    
    kernel3 =np.array([[1, -1, -1, 1],
                       [1, -1, -1, 1],
                       [1, -1, -1, 1],
                       [1, -1, -1, 1]]) 
    dst3 = cv2.filter2D(img, -1, kernel3) 
    
    
    kernel4 =np.array([[ 1,  1,  1,  1],
                       [-1, -1, -1, -1],
                       [-1, -1, -1, -1],
                       [ 1,  1,  1,  1]]) 
    dst4 = cv2.filter2D(img, -1, kernel4) 
    
    
    kernel5 =np.array([[ 1,  1, -1, -1],
                       [ 1,  1, -1, -1],
                       [-1, -1,  1,  1],
                       [-1, -1,  1,  1]]) 
    dst5 = cv2.filter2D(img, -1, kernel5) 
    
    arr.append(dst1)
    arr.append(dst2)
    arr.append(dst3)
    arr.append(dst4)
    arr.append(dst5)
    
    return arr

#----------------------------------------------------------------

if __name__ == '__main__':

    output = open("test.csv" , 'w')
    outArr = []
    imageArr= []
    
    counter = 1
    for imageName in glob.glob('*.jpg'): 
    
        print imageName[:-4]
        imageArr.append( int(imageName[:-4]) )
        features = []  
        counter += 1 
        
        #1
        img = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)  
        arr = filters(img) 
        for item in arr:   
            features.append( w_b(item) ) 
        
        #2 
        for y in range(2):
            for x in range(2):
                tmp = img[y*100:(y+1)*100, x*100:(x+1)*100]
                arr = filters(tmp) 
                for item in arr:   
                    features.append( w_b(item) )
        
        #3 
        for y in range(4):
            for x in range(4):
                tmp = img[y*50:(y+1)*50, x*50:(x+1)*50] 
                arr = filters(tmp) 
                for item in arr:   
                    features.append( w_b(item) )
                    
        outArr.append(  str(features)[ 1 : len(str(features))-1 ]   )
        
    checker = 1
    i = 0 
    
    for i in range(len (outArr)):
        output.write (  outArr [imageArr.index(min(imageArr))] +"\n" ) 
        imageArr[   imageArr.index(min(imageArr)) ] = 200000       
        
    output.close()    
    