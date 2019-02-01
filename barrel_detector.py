'''
ECE276A WI19 HW1
Blue Barrel Detector
'''
import numpy as np
from scipy.stats import multivariate_normal as mvn
import os, cv2
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

class BarrelDetector():
	def __init__(self):
		self.PARAMETER_PATH = "./trainset/parameter"
		self.HSV_SPACE = False

		self.true_foreground_mean = np.load(os.path.join(self.PARAMETER_PATH,"true_foreground_mean.npy"))
		self.true_foreground_cov = np.load(os.path.join(self.PARAMETER_PATH, "true_foreground_cov.npy"))
		self.true_foreground_pre = np.load(os.path.join(self.PARAMETER_PATH, "true_foreground_pre.npy"))

		self.false_foreground_mean = np.load(os.path.join(self.PARAMETER_PATH,"false_foreground_mean.npy"))
		self.false_foreground_cov = np.load(os.path.join(self.PARAMETER_PATH,"false_foreground_cov.npy"))
		self.false_foreground_pre = np.load(os.path.join(self.PARAMETER_PATH,"false_foreground_pre.npy"))

		self.background_mean = np.load(os.path.join(self.PARAMETER_PATH,"background_mean.npy"))
		self.background_cov = np.load(os.path.join(self.PARAMETER_PATH,"background_cov.npy"))
		self.background_pre = np.load(os.path.join(self.PARAMETER_PATH,"background_pre.npy"))

		self.prior_true_foreground = np.load(os.path.join(self.PARAMETER_PATH,"priorProbability_T_F_G.npy"))[0]
		self.prior_false_foreground = np.load(os.path.join(self.PARAMETER_PATH,"priorProbability_T_F_G.npy"))[1]
		self.prior_background = np.load(os.path.join(self.PARAMETER_PATH,"priorProbability_T_F_G.npy"))[2]
		self.counter = ""
	def segment_image(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img.astype(np.float32, casting = "safe") / 255
		image = img
		d1 = image.shape[0]
		d2 = image.shape[1]
		image = np.reshape(image, (d1 * d2, -1))
		ptf = np.zeros(shape=(d1 * d2,))
		pff = np.zeros(shape=(d1 * d2,))
		pbg = np.zeros(shape=(d1 * d2,))
		for hc in range(self.true_foreground_mean.shape[0]):
			ptf = ptf + self.true_foreground_pre[hc] * mvn.pdf(image, mean=self.true_foreground_mean[hc], cov=self.true_foreground_cov[hc], allow_singular=True)
		for hc in range(self.false_foreground_mean.shape[0]):
			pff = pff + self.false_foreground_pre[hc] * mvn.pdf(image, mean=self.false_foreground_mean[hc], cov=self.false_foreground_cov[hc], allow_singular=True)
		for hc in range(self.background_mean.shape[0]):
			pbg = pbg + self.background_pre[hc] * mvn.pdf(image, mean=self.background_mean[hc], cov=self.background_cov[hc], allow_singular=True)

		ptf = np.reshape(ptf, (d1, d2))
		# ptf = ptf * self.prior_true_foreground
		pff = np.reshape(pff, (d1, d2))
		# pff = ptf * self.prior_false_foreground
		pbg = np.reshape(pbg, (d1, d2))
		# pbg = pbg * (self.prior_background - 0.5)

		mask = np.zeros(ptf.shape, dtype = np.uint8)
		mask[ptf > pbg] = 1
		mask[pff > ptf] = 0
		# plt.imsave("result_pic/mask_" + (self.counter[0:-4]) + ".png", mask)
		print("show mask")
		plt.imshow(mask)
		plt.show()
		return mask

	def get_bounding_box(self, img):
		mask = self.segment_image(img)
		kernel = np.ones((3,3), np.uint8)
		mask = cv2.dilate(mask, kernel, iterations=3)
		kernel = np.ones((8, 8), np.uint8)
		mask = cv2.erode(mask, kernel, iterations=2)
		mask = cv2.dilate(mask, kernel, iterations=3)
		mask = label(mask)
		props = regionprops(mask)
		props = sorted(props, key=lambda x: abs((x.bbox[1] - x.bbox[3]) * (x.bbox[2] - x.bbox[0])), reverse = True)
		boxes = []
		counter = 0
		for prop in props:
			counter += 1
			if (counter == 3):
				break
			box = [prop.bbox[1],prop.bbox[0],prop.bbox[3], prop.bbox[2]]
			boxes.append(box)
		return boxes


if __name__ == '__main__':
	folder = "trainset/origin_picture"
	my_detector = BarrelDetector()
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder,filename))
		my_detector.counter = filename
		bboxs = my_detector.get_bounding_box(img)
		for bbox in bboxs:
			cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 3)
		# cv2.imwrite("result_pic/" + filename[0:-4] + "_bbx.png", img)
		cv2.imshow("image",img)
		cv2.waitKey(0)


