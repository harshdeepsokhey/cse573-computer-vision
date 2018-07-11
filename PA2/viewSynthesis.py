import numpy as np
import cv2



def getViewSynthesis(v,disp, mode = 'Left'):
	views = np.zeros(v.shape, dtype=np.uint8)
	N,M, ch = v.shape

	for row in range(N):
		for col in range(M):
			d = disp[row,col]

			mid = d / 2

			if row - mid < 0 :
				continue

			views[row, col - mid] = v[row, col]

	return views


def main():
	view1 = cv2.imread('./data/view1.png')
	view5 = cv2.imread('./data/view5.png')
	disp1 = cv2.imread('./data/disp1.png',0)
	disp5 = cv2.imread('./data/disp5.png',0)

	N,M,ch = view1.shape


	views = getViewSynthesis(view1,disp1)
	cv2.imwrite('./result/views_after_view1.png',views)

	for  row in range(N):
		for col in range(M):
			d = disp5[row, col]

			mid = d / 2

			if col + mid >= M:
				continue

			if views[row, col + mid].all() == 0:
				views[row, col + mid] = view5[row,col]

	cv2.imwrite('./result/views_after_view2.png',views)


if __name__ == '__main__':
	main()
