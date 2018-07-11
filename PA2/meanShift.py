import cv2
import numpy as np
from matplotlib import pyplot as plt
import random




def main():
	img = cv2.imread('./data/Butterfly.jpg')
	N,M,rgb = img.shape

	print img.shape

	# 5 features : R,G,B, x,y
	featureMatrix = np.zeros((N*M,5))
	result_img = np.zeros(img.shape,)

	idx = 0
	h = 50
	max_iter = 60

	for row in range(N):
		for col in range(M):
			# color (R,G,B)
			featureMatrix[idx,0] = img[row, col, 0]
			featureMatrix[idx,1] = img[row, col, 1]
			featureMatrix[idx,2] = img[row, col, 2]

			# position (x,y)
			featureMatrix[idx,3] = row
			featureMatrix[idx,4] = col

			idx = idx + 1

	canSelect = True
	while len(featureMatrix) > 0:

		i = random.randint(0, len(featureMatrix) - 1)

		if canSelect:
			R_mean = featureMatrix[i][0]
			G_mean = featureMatrix[i][1]
			B_mean = featureMatrix[i][2]

			x_mean = featureMatrix[i][3]
			y_mean = featureMatrix[i][4]

		cluster = []
		dist = 0

		for j in range(len(featureMatrix)):
			dist = np.sqrt(np.square(R_mean - featureMatrix[j][0]) +\
						np.square(G_mean - featureMatrix[j][1]) +\
						np.square(B_mean - featureMatrix[j][2]) +\
						np.square(x_mean - featureMatrix[j][3]) +\
						np.square(y_mean - featureMatrix[j][4]))
			
			#print 'Initial Euclidean Distance = {:.3f}'.format(d)
			if dist < h:
				cluster.append(j)

		#print "featureMatrix size = ",len(featureMatrix)
		#l_cluster = len(cluster)

		if len(cluster) > 0:
			R_mean_1 = 0
			G_mean_1 = 0
			B_mean_1 = 0
			x_mean_1 = 0
			y_mean_1 = 0

			for k in range(len(cluster)):
				R_mean_1 += featureMatrix[cluster[k]][0]
				G_mean_1 += featureMatrix[cluster[k]][1]
				B_mean_1 += featureMatrix[cluster[k]][2]
				x_mean_1 += featureMatrix[cluster[k]][3]
				y_mean_1 += featureMatrix[cluster[k]][4]

			R_mean_1 = R_mean_1 / len(cluster)
			G_mean_1 = G_mean_1 / len(cluster)
			B_mean_1 = B_mean_1 / len(cluster)
			x_mean_1 = x_mean_1 / len(cluster)
			y_mean_1 = y_mean_1 / len(cluster)

			dist_1 = np.sqrt(np.square(R_mean_1 - R_mean) +\
						np.square(G_mean_1 - G_mean) +\
						np.square(B_mean_1 - B_mean) +\
						np.square(x_mean_1 - x_mean) +\
						np.square(y_mean_1 - y_mean))

			print 'Updated Euclidean Distance  = {:.3f}'.format(dist_1)

			if dist_1 < max_iter:
				print "Cluster Size = ", len(cluster)

				for c in range(len(cluster)):
					x_pos = int(featureMatrix[cluster[c]][3])
					y_pos = int(featureMatrix[cluster[c]][4])

					#print x_pos, y_pos
					result_img[x_pos][y_pos][0] = R_mean_1
					result_img[x_pos][y_pos][1] = G_mean_1
					result_img[x_pos][y_pos][2] = B_mean_1

				featureMatrix  =np.delete(featureMatrix,cluster,0)
				canSelect = True

			else:
				R_mean = R_mean_1
				G_mean = G_mean_1
				B_mean = B_mean_1
				x_mean = x_mean_1
				y_mean = y_mean_1

				canSelect = False

			print 'featureMatrix size = ',featureMatrix.shape

	print result_img
	cv2.imwrite('./result/mean_shift_h'+str(h)+'_iter_'+str(max_iter)+'.png',result_img)


if __name__ == '__main__':
	main()


