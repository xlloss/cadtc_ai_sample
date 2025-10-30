import cv2
import numpy as np
from rknnlite.api import RKNNLite
RKNN_MODEL = 'letnet_onnx_model.rknn'
TEST_PIC_NAME = './9786.png'

def show_top5(result):
    output = result[0].reshape(-1)
    output_sorted = sorted(output, reverse=True)
    top5_str = 'LeNet\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'

            top5_str += topi
    print(top5_str)

if __name__ == '__main__':
    rknn_model = RKNN_MODEL
    rknn_lite = RKNNLite()

    # load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)

    print('done')

    ori_img = cv2.imread(TEST_PIC_NAME)
    img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
    img = np.expand_dims(img,0)
    img = img.reshape(1, 1, 28, 28)

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)

    # Inference
    print('--> Running model')
    outputs = rknn_lite.inference(inputs=[img])
    show_top5(outputs)
    print('done')
    rknn_lite.release()
