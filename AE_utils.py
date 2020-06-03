import numpy as np
import cv2
import matplotlib.pyplot as plt

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def reshape (img, img_width, img_height):
    img = (cv2.resize(cv2.imread(img), (img_width, img_height)))
    img = img * 1./255
    img = img.reshape([-1, img_width, img_height,3])
    return img

def CheckAnomaly(model, val_img, test_img, img_width, img_height, threshold):
    val_img = reshape(val_img, img_width, img_height)
    test_img = reshape(test_img, img_width, img_height)
    pred_img = model.predict(val_img)
    val_mse = mse(pred_img[0], val_img[0])
    test_mse = mse(pred_img[0], test_img[0])
    plt.figure(figsize=(15,15))
    plt.subplot(1,3,1)
    plt.imshow(val_img[0])
    plt.axis('off')
    plt.title('Oroginal Image')
    plt.subplot(1,3,2)
    plt.imshow(pred_img[0])
    plt.axis('off')
    plt.title('Predicted Image')
    plt.subplot(1,3,3)
    plt.imshow(test_img[0])
    plt.axis('off')
    plt.title('Test Image')
    print('validate_mse: {}'.format(val_mse))
    print('Test_mse: {}'.format(test_mse))
    if test_mse  > threshold:
        print("Image is anomalous")
    else:
        print("Image is not anomalious")
    return test_mse  > threshold