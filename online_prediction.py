import argparse
#import cv2
import numpy as np
from PIL import Image
from primesense import openni2
from skimage.transform import resize

from train_unet import get_unet
img_rows = 96
img_cols = 128
i=0
from data import load_pre_data#
from skimage.io import imsave
import os
if __name__ == '__main__':
    #p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="")
    #p.add_argument('--v', dest='video_path', action='store', default='', help='path Video')
    #args = p.parse_args()

    model = get_unet()
    #model_select
    bit = 16
    model.load_weights('weights_unet_16.h5')

    imgs_bit_test, imgs_mask_test, imgs_bit_id_test = load_pre_data(bit)#
    imgs_bit_test = imgs_bit_test.astype('float32')
    mean = np.mean(imgs_bit_test)
    std = np.std(imgs_bit_test)
    imgs_bit_test -= mean
    imgs_bit_test /= std
    print("img.shape:", imgs_bit_test.shape)

    if bit == 8:
        print('-' * 30)
        print('Saving predicted masks to files...')
        print('-' * 30)
        pred_dir = 'preds_8'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        #n_frames = pbs.get_number_of_frames(depth_stream)
        #np.ndarray((imgs_bit_test.shape[0], img_rows, img_cols), dtype=np.uint16)#
        predicted_image = model.predict(imgs_bit_test, verbose=1)#progress bar

        for i in range(0, imgs_bit_test.shape[0]):
            image = (predicted_image[i][:, :, 0] * 255.).astype(np.uint8)
            img = Image.fromarray(image)
            #img.save("./predicted_images/" + str(i).zfill(4) + ".png")
            img.save("./"+pred_dir+"/" + str(i).zfill(4) + ".png")
    else:
        print('-' * 30)
        print('Saving predicted masks to files...')
        print('-' * 30)
        pred_dir = 'preds_16'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        #n_frames = pbs.get_number_of_frames(depth_stream)
        #np.ndarray((imgs_bit_test.shape[0], img_rows, img_cols), dtype=np.uint16)#
        predicted_image = model.predict(imgs_bit_test, verbose=1)#progress bar

        for i in range(0, imgs_bit_test.shape[0]):
            image = (predicted_image[i][:, :, 0] * 255.).astype(np.uint8)#grey
            #cv2.imwrite("./" + pred_dir + "/" + str(i).zfill(4) + "_2.png", img)
            """"""
            img = Image.fromarray(image)
            #img.save("./predicted_images/" + str(i).zfill(4) + ".png")
            img.save("./"+pred_dir+"/" + str(i).zfill(4) + ".png")





        #img.save("./predicted_images/" + ".png")
        #imsave(os.path.join("D:\Program Files\sunrelease\deep-segmentation\predicted_images\ "+'_pred2.png'), img)  # pred_dir,


    """
        for image, image_id in zip(imgs_mask_test, imgs_bit_id_test):
            image = (image[:, :, 0] * 255.).astype(np.uint8)
            img = Image.fromarray(image)#
            imsave(os.path.join(str(image_id).split('/')[-1] + '_pred2.png'), image)#pred_dir,

    

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds_16'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_bit_test, imgs_bit_id_test):
        #image = (image[:, :, 0] * 255.).astype(np.uint8)
        image = (image[:, :, 0] * 255.).astype('float32')
        #imgs_bit_mask_train = imgs_bit_mask_train.astype('float32')
        print("img.shape:",image.shape)
        print("img:",image)
        img = Image.fromarray(image)#
        imsave(os.path.join(str(image_id).split('/')[-1] + '_pred2.png'), image)#pred_dir,
        #img = Image.fromarray(image)
        #img.save("./predicted_images/" + str(i).zfill(4) + ".png")
    """
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
    for i in range(0, n_frames - 1):
        frame_depth = depth_stream.read_frame()
        print("Depth {0} of {1} - {2}".format(i, n_frames, frame_depth.frameIndex))
        frame_depth_data = frame_depth.get_buffer_as_uint16()
        depth_array = np.ndarray((frame_depth.height, frame_depth.width),
                                 dtype=np.uint16,
                                 buffer=frame_depth_data)
        depth_array = resize(depth_array, (img_rows, img_cols), preserve_range=True)
        imgs = np.array([depth_array], dtype=np.uint16)
        imgs = imgs[..., np.newaxis]

        imgs = imgs.astype('float32')

        mean = np.mean(imgs)
        std = np.std(imgs)

        imgs -= mean
        imgs /= std

        np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint16)
        predicted_image = model.predict(imgs, verbose=0)
        image = (predicted_image[0][:, :, 0] * 255.).astype(np.uint8)
        img = Image.fromarray(image)
        img.save("./predicted_images/" + str(i).zfill(4) + ".png")

    depth_stream.stop()
    openni2.unload()
    """
