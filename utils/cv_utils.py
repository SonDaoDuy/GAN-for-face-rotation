import cv2
from matplotlib import pyplot as plt
import numpy as np

def read_cv2_img(path):
    '''
    Read color images
    :param path: Path to image
    :return: Only returns color images
    '''
    img = cv2.imread(path, -1)

    #print(img.shape)
    if img is not None:
        if len(img.shape) != 3:
            return None
        # #[0,255] -> [-1,1]
        # img = img *2 - 1
        # # RGB -> BGR
        # #img = img[:,:,[2,1,0]]
        # #show_cv2_img(img)
        # # B x H x W x C-> B x C x H x W
        # img = img.transpose(2, 0, 1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, (96,96))

    return img

def show_cv2_img(img, title='img'):
    '''
    Display cv2 image
    :param img: cv::mat
    :param title: title
    :return: None
    '''
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_images_row(imgs, titles, rows=1):
    '''
       Display grid of cv2 images image
       :param img: list [cv::mat]
       :param title: titles
       :return: None
    '''
    assert ((titles is None) or (len(imgs) == len(titles)))
    num_images = len(imgs)

    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, num_images + 1)]

    fig = plt.figure()
    for n, (image, title) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(rows, np.ceil(num_images / float(rows)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        ax.set_title(title)
        plt.axis('off')
    plt.show()

def convert_image(data):
    if len(data.shape)==4:
        img = data.transpose(0, 2, 3, 1)+1
        img = img / 2.0
        img = img * 255.
        img = img[:,:,:,[2,1,0]]

    else:
        img = data.transpose(1, 2, 0)+1
        img = img / 2.0
        img = img * 255.
        img = img[:,:,[2,1,0]]

    return img.astype(np.uint8)