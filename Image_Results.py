import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

No_of_dataset = 3


def Image_Results():
    for s in range(2, No_of_dataset):
        Images = np.load('Images_' + str(s + 1) + '.npy', allow_pickle=True)
        GT = np.load('Ground_Truth_' + str(s + 1) + '.npy', allow_pickle=True)
        dehaz = np.load('De_Hazed_Images_' + str(s + 1) + '.npy', allow_pickle=True)
        Detection = np.load('Detected_Images_' + str(s + 1) + '.npy', allow_pickle=True)
        ind = [[0, 1, 2, 3, 5], [0, 1, 2, 3, 45], [5, 28, 61, 106, 116]]  # [0, 1, 2, 3, 45] [5, 6, 10, 11, 12]
        for j in range(5):
            image = Images[ind[s][j]]
            gt = GT[ind[s][j]]
            deh = dehaz[ind[s][j]]
            det = Detection[ind[s][j]]

            # Adjust figsize to reduce subplot size
            fig, ax = plt.subplots(2, 2, figsize=(8, 8))  # You can further adjust the size here
            plt.suptitle("Image %d" % (j + 1), fontsize=15)

            plt.subplot(2, 2, 1)
            plt.title('Original Image')
            plt.imshow(image)

            plt.subplot(2, 2, 2)
            plt.title('Groundtruth')
            plt.imshow(gt)

            plt.subplot(2, 2, 3)
            plt.title('De-Hazed Images')
            plt.imshow(deh)

            plt.subplot(2, 2, 4)
            plt.title('Detected Images')
            plt.imshow(det)

            path = "./Results/Dataset_%s_image_%s.png" % (s + 1, j + 1)
            plt.savefig(path)
            plt.show(block=False)
            plt.pause(2)  # Keeps the graph open for 2 seconds
            plt.close()

            cv.imwrite('./Results/Image Results/Dataset-' + str(s + 1) + 'Original-Images-' + str(j + 1) + '.png',
                       image)
            cv.imwrite('./Results/Image Results/Dataset-' + str(s + 1) + 'GT-Images-' + str(j + 1) + '.jpg', gt)
            cv.imwrite('./Results/Image Results/Dataset-' + str(s + 1) + 'De-Hazed-Images-' + str(j + 1) + '.jpg', deh)
            cv.imwrite('./Results/Image Results/Dataset-' + str(s + 1) + 'Detection-Images-' + str(j + 1) + '.jpg', det)


if __name__ == '__main__':
    Image_Results()
