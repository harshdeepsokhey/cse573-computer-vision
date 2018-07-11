# PA2 
import cv2
import numpy as np


left_img = cv2.cvtColor(cv2.imread('./data/view1.png'), cv2.COLOR_BGR2GRAY)  #read it as a grayscale image
right_img = cv2.cvtColor(cv2.imread('./data/view5.png'), cv2.COLOR_BGR2GRAY)

OcclusionCost = 20

#For Dynamic Programming you have build a cost matrix. Its dimension will be numcols x numcols
N,M = left_img.shape

print ('Left Image : [{0},{1}]'.format(N,M))
print ('Right Image: [{0}]'.format(right_img.shape))


dm_left = np.zeros((N,M),dtype=np.float)
dm_right = np.zeros((N,M),dtype=np.float)

#We first populate the first row and column values of Cost Matrix
for k in range(N):

    if k % 30 == 0: print k, 'processed!'

    CostMatrix = np.zeros((M,M), dtype=np.float)
    DirectionMatrix = np.zeros((M,M), dtype=np.float)

    for i in range(M):
        CostMatrix[k,0] = i*OcclusionCost
        CostMatrix[0,k] = i*OcclusionCost

    # Now, its time to populate the whole Cost Matrix and DirectionMatrix
    for i in range(N):
        for j in range(M):
            min1 = CostMatrix[i-1,j-1] + np.abs(int(left_img[k,i]) - int(right_img[k,j]))
            min2 = CostMatrix[i-1,j] + OcclusionCost
            min3 = CostMatrix[i,j-1] + OcclusionCost
            cmin = np.min((min1, min2, min3))

            CostMatrix[i][j] = cmin

            if (min1 == cmin):
                DirectionMatrix[i,j] = 1
            if (min2 == cmin):
                DirectionMatrix[i,j] = 2
            if (min3 == cmin):
                DirectionMatrix[i,j] = 3


    p = N - 1
    q = M - 1

    while(p != 0 and q!= 0):
        #print "Inside Loop, p = {0}, q = {1}".format(p,q)
        if DirectionMatrix[p,q] == 1:
            #print('P matches Q')
            dm_left[k,p] = np.abs(p-q) 
            dm_right[k,q] = np.abs(p-q) 
            p = p - 1
            q = q - 1

        elif DirectionMatrix[p,q] == 2:
            #print('P is unmatched')
            p = p - 1
            
        elif DirectionMatrix[p,q] == 3:
            #print('Q is unmatched')
            q = q -1    
   
print dm_left
print dm_right

cv2.imwrite('./result/view1_dm.png', dm_left)
cv2.imwrite('./result/view5_dm.png', dm_right)

