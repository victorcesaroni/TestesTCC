import sys
import numpy as np
import matplotlib.pyplot as plt

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x=%.0f, y=%.0f, z=%f' % (x, y, z)

slice = 0
img = []
size = [0,0,0]

def display_img(fig, ax):
    global slice
    global img
    global size
    
    ax.cla()
    im = plt.imshow(img[slice], interpolation='none', cmap='ocean')
    ax.format_coord = Formatter(im)
    plt.suptitle("slice %d" % slice)
    fig.canvas.draw()

def press(e):
    global slice
    print(e.key)
    if e.key == 'left':
        slice = (slice - 1)
        if slice < 0:
            slice = size[0]-1
        print("slice",slice)
        display_img(fig, ax)
    if e.key == 'right':
        slice = (slice + 1) % size[0]
        print("slice",slice)
        display_img(fig, ax)
    if e.key == 'ctrl+c':
        exit()

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("USE w h d img")
        exit()
        
    size = [int(sys.argv[3]), int(sys.argv[2]), int(sys.argv[1])]
    slice = 0
    img = (np.fromfile(sys.argv[4], dtype='float32').reshape(size))

    fig, ax = plt.subplots()
    display_img(fig, ax)
    cid = fig.canvas.mpl_connect('key_press_event', press)
        
    plt.show()
