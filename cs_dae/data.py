from matplotlib import image
import numpy as np
from keras.datasets import mnist
import imageio
from PIL import Image


def get_mnist_data():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    return (x_train, x_test)

def preprocess_celebA_data(data_dir='./img_align_celeba/'):
    for i in range(1, 202536):
        fname = str(i)
        fname = fname.zfill(6)
        im_orig = imageio.imread(data_dir + fname + '.jpg')
        h, w = im_orig.shape[:2]
        j = int(round((h - 108)/2.))
        k = int(round((w - 108)/2.))
        im = np.array(Image.fromarray(im_orig[j:j+108, k:k+108]).resize([64,64]))
        imageio.imwrite(data_dir  + fname + '.jpg', im)
    return

def get_celebA_data(train=False, test=True, train_data_dir='../img_align_celeba/train/data/', test_data_dir='testing_images/celebA/data/'):

    if train:
        x_train = np.zeros((160000, 64,64, 3),dtype=np.float32)

        for i in range(1,160001):
            fname = str(i)
            fname = fname.zfill(6)
            x_train[i-1] = (image.imread(train_data_dir + fname + '.jpg')).reshape((64,64,3))/255.

        return x_train

    else:
        x_test = np.zeros((64, 64,64, 3),dtype=np.float32)

        for i in range(182638,182702):
            fname = str(i)
            fname = fname.zfill(6)
            x_test[i-182638] = (image.imread(test_data_dir + fname + '.jpg')).reshape((64,64,3))/255.

        return x_test
