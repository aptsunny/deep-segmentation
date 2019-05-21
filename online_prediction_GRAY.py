import argparse

import numpy as np
from PIL import Image
#from primesense import openni2
from skimage.transform import resize

from keras.preprocessing.image import array_to_img
from train_unet3_conv import get_conv
from train_unet import get_unet

import glob
import cv2
img_rows = 96
img_cols = 128

if __name__ == '__main__':
    #p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="")
    #p.add_argument('--v', dest='video_path', action='store', default='', help='path Video')
    #args = p.parse_args()

    #model = get_conv()
    model = get_unet()
    bit = 16
    i=0
    model.load_weights('weights_unet_16.h5')

    """
    dev = openni2.Device
    try:
        openni2.initialize()
        dev = openni2.Device.open_file(args.video_path.encode('utf-8'))
        print(dev.get_sensor_info(openni2.SENSOR_DEPTH))
    except (RuntimeError, TypeError, NameError):
        print(RuntimeError, TypeError, NameError)

    pbs = openni2.PlaybackSupport(dev)
    depth_stream = pbs.device.create_depth_stream()

    pbs.set_repeat_enabled(True)
    pbs.set_speed(-1.0)
    depth_stream.start()

    n_frames = pbs.get_number_of_frames(depth_stream)
    """
    """"""
    dirc="D:/Program Files/sunrelease/deep-segmentation/test"
    img_depth=glob.glob(dirc+"/*."+"png")

    n_frames=len(img_depth)

    #cla = np.ndarray((n_frames, img_rows, img_cols, 1),dtype=np.float32)
    for imgname in img_depth:
        midname = imgname[imgname.rindex("\\")+1:]
        

    #for i in range(0, n_frames - 1):
    #    frame_depth = depth_stream.read_frame()

        frame_depth=cv2.imread(dirc+"\\"+midname,cv2.IMREAD_GRAYSCALE)
        frame_depth=cv2.resize(frame_depth,(img_cols,img_rows),interpolation=cv2.INTER_LANCZOS4)

        print('imgs1.shape', frame_depth.shape)
        #img_rows = 96
        #img_cols = 128
        frame_depth.resize((img_rows, img_cols, 1))
        imgs = array_to_img(frame_depth)

        """"
        print("Depth {0} of {1} - {2}".format(i, n_frames, frame_depth.frameIndex))

        frame_depth_data = frame_depth.get_buffer_as_uint16()
        depth_array = np.ndarray((frame_depth.height, frame_depth.width),
                                 dtype=np.uint16,
                                 buffer=frame_depth_data)
        depth_array = resize(depth_array, (img_rows, img_cols), preserve_range=True)
        imgs = np.array([depth_array], dtype=np.uint16)
        imgs = imgs[..., np.newaxis]
        imgs = imgs.astype('float32')
        """
        mean = np.mean(imgs)
        std = np.std(imgs)
        imgs -= mean
        imgs /= std
        imgs= imgs.astype('float32')
        imgs /= 255.  # scale masks to [0, 1]
        print('imgs.shape', imgs.shape)

        #img=np.array([imgs],dtype=np.float32)
        #cla[i]=img
        i+=1

        imgs.resize((1, img_rows, img_cols, 1))

        predicted_image = model.predict(imgs, verbose=0)

        image = (predicted_image[0][:, :, 0] * 255.).astype(np.uint8)
        img = Image.fromarray(image)
        img.save("./predicted_images/" + str(i).zfill(4) + ".png")

    #print("predicted:", cla.shape)

    """
        np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint16)
        predicted_image = model.predict(imgs, verbose=0)
        image = (predicted_image[0][:, :, 0] * 255.).astype(np.uint8)
        img = Image.fromarray(image)
        img.save("./predicted_images/" + str(i).zfill(4) + ".png")

    #depth_stream.stop()
    #openni2.unload()
"""