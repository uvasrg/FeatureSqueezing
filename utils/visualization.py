import pickle
from PIL import Image
import numpy as np
import pdb
from squeeze import median_filter_np, binary_filter_np

IMAGE_SIZE = 28

filter_m = lambda x: median_filter_np(x, 3)
filter_b = lambda x: binary_filter_np(x)
filter_mb = lambda x: filter_b(filter_m(x))
filter_bm = lambda x: filter_m(filter_b(x))

def show_img(pixel_array, mode=None):
    img = Image.fromarray(pixel_array*255, mode=mode)
    img.show()


def show_imgs(imgs, width_num=10, height_num=10, x_margin=2, y_margin=2, fpath='/tmp/test.png'):
    total_width = width_num * IMAGE_SIZE + (width_num-1)*x_margin
    total_height = height_num * IMAGE_SIZE + (height_num-1)*y_margin

    new_im = Image.new('RGB', (total_width, total_height), (255,255,255))

    x_offset = 0
    y_offset = 0

    for img_array in imgs:
        # pdb.set_trace()
        # img_array = img_reshape(img_array)
        # img = Image.fromarray(img_array, 'RGB')
        img = Image.fromarray(np.squeeze(img_array)*255)
        img_width, img_height = img.size
        if x_offset + img_width <= total_width:
            pass
        else:
            
            if y_offset + img_height <= total_height:
                x_offset = 0
                y_offset += img_height + y_margin
            else:
                break
        new_im.paste(img, (x_offset,y_offset))
        x_offset += img_width + x_margin
    new_im.save(fpath)
    new_im.show()


def get_first_example_id_each_class(Y_test):
    Y_test_labels = np.argmax(Y_test, axis=1)
    selected_idx = [ np.where(Y_test_labels==i)[0][0] for i in range(10)]
    return selected_idx


def draw_fgsm_adv_examples(adv_x_dict, Y_test, fpath):
    eps_list = [0,0.1,0.2,0.3,0.4,0.5]
    width_num=10
    selected_example_idx = get_first_example_id_each_class(Y_test)
    imgs = []

    for eps in eps_list:
        adv_x_dict[eps] = adv_x_dict[eps][selected_example_idx,:]

    for eps in eps_list:
        imgs += list(adv_x_dict[eps])
        imgs += list(filter_b(adv_x_dict[eps]))

    show_imgs(imgs, width_num=width_num, height_num=len(imgs)/width_num, fpath=fpath)


def draw_jsma_adv_examples(X_adv, X_test, Y_test, fpath):
    width_num = 10
    selected_example_idx = get_first_example_id_each_class(Y_test)

    X_test = X_test[selected_example_idx,:]
    X_adv = X_adv[selected_example_idx,:]
    
    imgs = []
    imgs += list(X_test)
    imgs += list(filter_m(X_test))
    imgs += list(X_adv)
    imgs += list(filter_m(X_adv))

    show_imgs(imgs, width_num=width_num, height_num=len(imgs)/width_num, fpath=fpath)

def draw_orig_binary_examples():
    imgs = {}
    for j, img_group in enumerate(ret[0]):
        sid = j
        if sid not in imgs:
            imgs[sid] = []
        for i, img in enumerate(img_group):
            img = img.reshape(img.shape[1:])
            imgs[sid].append(img)

    imgs_list = []
    for row in range(2):
        imgs_list += imgs[row]

    preview_imgs_square(imgs_list, width_num = 10, height_num = 2)


