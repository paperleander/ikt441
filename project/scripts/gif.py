#!/usr/bin/python

import os
import imageio
import glob
from IPython import display

image_folder = "../imgs"

def create_gif():
    folders = sorted(os.listdir("../{}/".format(image_folder)))
    newest = folders[-1]
    anim_file = "../gifs/{}.gif".format(newest)

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('../{}/{}/*.png'.format(image_folder, newest))
        filenames = sorted(filenames)
        last = -1
        for i,filename in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        #image = imageio.imread(filename)
        #writer.append_data(image)

    import IPython
    #if IPython.version_info > (6,2,0,''):
    display.Image(filename=anim_file)
    print("done")

if __name__ == '__main__':
    create_gif()
