import cv2


# read the image
img = cv2.imread('Icons/1.png')

# shape prints the tuple (height,weight,channels)
print(img.shape)

# img will be a numpy array of the above shape
print(img)