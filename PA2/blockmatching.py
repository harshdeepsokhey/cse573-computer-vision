import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import os

def showImages(v1,v5,d1,d5):

	plt.figure(1)
	plt.subplot(221)
	plt.imshow(v1);
	plt.title('View 1')
	plt.subplot(222)
	plt.imshow(v5);
	plt.title('View 5')
	plt.subplot(223)
	plt.imshow(d1);
	plt.subplot(224)
	plt.imshow(d5);
	plt.show()
    
def getSize(v1,v5,d1,d5):
    print 'View 1:',v1.shape
    print 'View 5:',v5.shape
    print 'Disp 1:',d1.shape
    print 'Disp 5:',d5.shape
    

def get3x3block(img, r,c):
    '''
        returns a 3x3 block of the image based on r,c 
    '''
    return img[r-1:r+2,c-1:c+2]


def getDisparityMap(imgLeft, imgRight, mode='Left'):
    '''
    	Returns generated disparity map 
    '''
    N,M = imgLeft.shape
    print 'Image Size = ', N,',',M

    disparityMap = np.zeros(imgLeft.shape)

    plt.figure(2)
    plt.subplot(121)
    plt.imshow(imgLeft)
    plt.title('Left')
    plt.subplot(122)
    plt.imshow(imgRight)
    plt.title('Right')
    plt.show()

    print 'mode=',mode

    for row in range(1,N-1):

        if row % 30 == 0 : print 'row #',row, 'processed!'

        for col in range(1,M-1):
            
            imgLeft_block = get3x3block(imgLeft, row, col)

            minSAD, minSADidx = sys.maxint, -1

            factor = int(N//5)

            start = col - factor # default : left
            if mode == 'Right':
            	start = col + factor

            end = col

            inc = 1
            if mode == 'Right':
            	inc = -1

            for idx in range(start, end, inc):
            	if mode == 'Left':
                	if idx <= 1: idx = 1

                if mode == 'Right':
                	if idx >= M -1 : idx = M - 2

                imgRight_block = get3x3block(imgRight, row, idx)

                # calculate SAD for left image 
                diff = np.square(imgLeft_block - imgRight_block)
                SAD = np.sum(diff)

                if SAD < minSAD:
                    minSAD = SAD
                    minSADidx= idx

            
            disparityMap[row][col] = col - minSADidx

            if mode == 'Right':
            	disparityMap[row,col] = minSADidx - col

    return disparityMap



def getDisparityMapWrapper(img1, img2, disp_border, mode = 'Left'):
	# ## Disparity Map wrt MODE Image

	print 'Disparity Map with respect to',mode,' Image'

	str_filename = 'disparityMap'+str(mode)

	filename = os.path.join(os.getcwd(),'result/'+str_filename+'.png')

	# if os.path.exists(filename):
	# 	print 'File ',str_filename,'Exits! Reading ..'
	# 	disparityMap = cv2.imread(filename,0)
	# 	print 'type=',type(disparityMap)
	# else:
	# 	disparityMap = getDisparityMap(img1, img2, mode)
	disparityMap = getDisparityMap(img1, img2, mode)


	# Normalized Disparity Map
	MAXdisparityMap = np.amax(disparityMap)
	disp = disparityMap / MAXdisparityMap

	# Mean Squared Error calculation
	mse = np.mean(np.square(disp_border - disparityMap))

	str_mse = 'MSE'+str(mode)
	print '{} = {:4f}'.format(str_mse,mse)

	

	plt.figure(3)
	plt.subplot(121)
	plt.imshow(disparityMap)
	plt.title(str_filename)
	plt.subplot(122)
	plt.imshow(disp)
	plt.title(str_filename+'(Normalized)')
	plt.show()

	cv2.imwrite('./result/'+str_filename+'.png',disparityMap)


def get9x9block(img, r,c):
    '''
        returns a 9x9 block of the image based on r,c 
    '''
    return img[r-4:r+5,c-4:c+5]

def getDisparityMap9x9(imgLeft, imgRight, mode='Left'):
    '''
    	Returns generated disparity map 
    '''
    N,M = imgLeft.shape
    print 'Image Size = ', N,',',M

    disparityMap = np.zeros(imgLeft.shape)

    plt.figure(2)
    plt.subplot(121)
    plt.imshow(imgLeft)
    plt.title('Left')
    plt.subplot(122)
    plt.imshow(imgRight)
    plt.title('Right')
    plt.show()

    print 'mode=',mode

    for row in range(4,N-4):

        if row % 30 == 0 : print 'row #',row, 'processed!'

        for col in range(4,M-4):
            
            imgLeft_block = get9x9block(imgLeft, row, col)

            minSAD, minSADidx = sys.maxint, -1

            factor = int(N//5)

            start = col - factor # default : left
            if mode == 'Right':
            	start = col + factor

            end = col

            inc = 1
            if mode == 'Right':
            	inc = -1

            for idx in range(start, end, inc):
            	if mode == 'Left':
                	if idx <= 5: idx = 5

                if mode == 'Right':
                	if idx >= M -5 : idx = M - 6

                imgRight_block = get9x9block(imgRight, row, idx)

                # calculate SAD for left image 
                diff = np.square(imgLeft_block - imgRight_block)
                SAD = np.sum(diff)

                if SAD < minSAD:
                    minSAD = SAD
                    minSADidx= idx

            
            disparityMap[row][col] = col - minSADidx

            if mode == 'Right':
            	disparityMap[row,col] = minSADidx - col

    return disparityMap



def getDisparityMapWrapper9x9(img1, img2, disp_border, mode = 'Left'):
	# ## Disparity Map wrt MODE Image

	print 'Disparity Map (9x9) with respect to',mode,' Image'

	str_filename = 'disparityMap9x9'+str(mode)

	filename = os.path.join(os.getcwd(),'result/'+str_filename+'.png')

	# if os.path.exists(filename):
	# 	print 'File ',str_filename,'Exits! Reading ..'
	# 	disparityMap = cv2.imread(filename,0)
	# 	print 'type=',type(disparityMap)
	# else:
	# 	disparityMap = getDisparityMap9x9(img1, img2, mode)
	disparityMap = getDisparityMap9x9(img1, img2, mode)


	# Normalized Disparity Map
	MAXdisparityMap = np.amax(disparityMap)
	disp = disparityMap / MAXdisparityMap

	# Mean Squared Error calculation
	mse = np.mean(np.square(disp_border - disparityMap))

	str_mse = 'MSE'+str(mode)
	print '{} (9x9)= {:4f}'.format(str_mse,mse)

	

	plt.figure(3)
	plt.subplot(121)
	plt.imshow(disparityMap)
	plt.title(str_filename)
	plt.subplot(122)
	plt.imshow(disp)
	plt.title(str_filename+'(Normalized)')
	plt.show()

	cv2.imwrite('./result/'+str_filename+'.png',disparityMap)


def consistencyCheck(dm1, dm5, mode='Left'):
	N,M = dm1.shape

	consistency = np.zeros((N,M))

	for row in range(N):
	    for col in range(M):

			px_1 = int(dm1[row, col])

			check = (col - px_1 > 0 and M > col - px_1)
			if mode == 'Right':
				check = (col +px_1 < M)

			if check == True:

				start = row
				end = col - px_1

				if mode == 'Right':
					start = row
					end = col + px_1

				px_2 = dm5[start, end]

			else:
				px_2=dm5[row,col]
			    
			if(px_1 == px_2):
				consistency[row,col]= px_1
			else: 
				consistency[row,col]=0

	return consistency

def performConsistencyCheck(v1_border, v5_border, disp_border, mode = 'Left'):

	consistency = consistencyCheck(v1_border, v5_border)

	cv2.imwrite('./result/consistency'+mode+'.png', consistency)
	plt.figure(2)
	plt.imshow(consistency)
	plt.show()

	N,M = v1_border.shape
	s = 0
	for row in range (N):
	    for col in range(M):
	        if(consistency[row,col] != 0):

				temp=np.square(disp_border[row,col]-consistency[row,col])
				s = s + temp

	mse = s / (N*M)
	str_mse = 'MSE'+str(mode)+'(post-consistencyCheck)'

	print '{} = {:.4f}'.format(str_mse,mse)


def consistencyCheck9x9(dm1, dm5, mode='Left'):
	N,M = dm1.shape

	consistency = np.zeros((N,M))

	for row in range(4,N-4):
	    for col in range(4,M-4):

			px_1 = int(dm1[row, col])

			check = (col - px_1 > 0 and M > col - px_1)
			if mode == 'Right':
				check = (col +px_1 < M)

			if check == True:

				start = row
				end = col - px_1

				if mode == 'Right':
					start = row
					end = col + px_1

				px_2 = dm5[start, end]

			else:
				px_2=dm5[row,col]
			    
			if(px_1 == px_2):
				consistency[row,col]= px_1
			else: 
				consistency[row,col]=0

	return consistency

def performConsistencyCheck9x9(v1_border, v5_border, disp_border, mode = 'Left'):

	consistency = consistencyCheck(v1_border, v5_border)

	cv2.imwrite('./result/consistency9x9'+mode+'.png', consistency)
	plt.figure(2)
	plt.imshow(consistency)
	plt.show()

	N,M = v1_border.shape
	s = 0
	for row in range (N):
	    for col in range(M):
	        if(consistency[row,col] != 0):

				temp=np.square(disp_border[row,col]-consistency[row,col])
				s = s + temp

	mse = s / (N*M)
	str_mse = 'MSE'+str(mode)+'(post-consistencyCheck)'

	print '{} (9x9) = {:.4f}'.format(str_mse,mse)


def blockMatching3x3(v1,v5,d1,d5): 

	# Padded Images with pad=1
	view1_with_border = cv2.copyMakeBorder(v1, 1, 1, 1, 1,cv2.BORDER_CONSTANT,value=0)
	view5_with_border = cv2.copyMakeBorder(v5, 1, 1, 1, 1,cv2.BORDER_CONSTANT,value=0)
	disp1_border=cv2.copyMakeBorder(d1, 1, 1, 1, 1,cv2.BORDER_CONSTANT,value=0)
	disp5_border=cv2.copyMakeBorder(d5, 1, 1, 1, 1,cv2.BORDER_CONSTANT,value=0)

	getSize(view1_with_border,view5_with_border,disp1_border, disp5_border)


	# disparity calculcations with respect to the left image
	getDisparityMapWrapper(view1_with_border, view5_with_border, disp1_border)

	# disparity calculation with respect to the right image
	getDisparityMapWrapper(view5_with_border, view1_with_border, disp5_border, mode='Right')

	# consistency check

	# left image
	performConsistencyCheck(view1_with_border, view5_with_border, disp1_border)

	# right image
	performConsistencyCheck(view5_with_border, view1_with_border, disp5_border, mode='Right')


def blockMatching9x9(v1,v5,d1,d5): 

	# Padded Images with pad=1
	view1_with_border = cv2.copyMakeBorder(v1, 4, 4, 4, 4,cv2.BORDER_CONSTANT,value=0)
	view5_with_border = cv2.copyMakeBorder(v5, 4, 4, 4, 4,cv2.BORDER_CONSTANT,value=0)
	disp1_border=cv2.copyMakeBorder(d1, 4, 4, 4, 4,cv2.BORDER_CONSTANT,value=0)
	disp5_border=cv2.copyMakeBorder(d5, 4, 4, 4, 4,cv2.BORDER_CONSTANT,value=0)

	getSize(view1_with_border,view5_with_border,disp1_border, disp5_border)


	# disparity calculcations with respect to the left image
	getDisparityMapWrapper9x9(view1_with_border, view5_with_border, disp1_border)

	# disparity calculation with respect to the right image
	getDisparityMapWrapper9x9(view5_with_border, view1_with_border, disp5_border, mode='Right')

	# consistency check

	# left image
	performConsistencyCheck9x9(view1_with_border, view5_with_border, disp1_border)

	# right image
	performConsistencyCheck9x9(view5_with_border, view1_with_border, disp5_border, mode='Right')


def main():
	# Original Images
	view1 = cv2.imread('./data/view1.png',0)
	view5 = cv2.imread('./data/view5.png',0)
	disp1=cv2.imread('./data/disp1.png',0)
	disp5=cv2.imread('./data/disp5.png',0)


	blockMatching3x3(view1, view5, disp1, disp5)
	blockMatching9x9(view1, view5, disp1, disp5)

if __name__ == '__main__':
	main()