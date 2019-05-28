import sys
import numpy as np
import matplotlib.pyplot as plt

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x=%.0f, y=%.0f, z=%f' % (x, y, z)

def display_img(img_path, size, slice):
    img = np.fromfile(img_path, dtype='float32').reshape(size)
    
    fig, ax = plt.subplots()
    im = plt.imshow(img[slice], interpolation='none', cmap='ocean')
    ax.format_coord = Formatter(im)
    plt.suptitle(img_path)

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("USE w h d slice img1 img2 imgN...")
        exit()
        
    size = [int(sys.argv[3]), int(sys.argv[2]), int(sys.argv[1])]
    slice = int(sys.argv[4])
    
    for i in range(5, len(sys.argv)):
        display_img(sys.argv[i], size, slice)
        
    plt.show()
