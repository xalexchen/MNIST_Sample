import numpy as np
import struct
import matplotlib.pyplot as plt

def readfile():
    with open('train-images.idx3-ubyte','rb') as f1:
        buf1 = f1.read()
    with open('train-labels.idx1-ubyte','rb') as f2:
        buf2 = f2.read()
    return buf1, buf2


def get_image(buf1):
    image_index = 0
    image_index += struct.calcsize('>I')

    count = struct.unpack_from('>I', buf1, image_index)
    image_index += struct.calcsize('>I')

    size = struct.unpack_from('>2I', buf1, image_index)
    image_index += struct.calcsize('>II')

    size_fmt = ">" + str(size[0] * size[1]) + "B"
    im = []
    for i in range(count[0]):
        temp = struct.unpack_from(size_fmt, buf1, image_index) # '>784B'的意思就是用大端法读取784个unsigned byte
        im.append(np.reshape(temp,(size[0],size[1])))
        image_index += struct.calcsize(size_fmt)  # 每次增加784B
    return im


def get_label(buf2): # 得到标签数据
    label_index = 0
    label_index += struct.calcsize('>I')

    #读取大小
    count = struct.unpack_from(">I", buf2, label_index)
    size_fmt = ">" + str(count[0]) + "B"

    label_index += struct.calcsize('>I')
    return struct.unpack_from(size_fmt, buf2, label_index)

if __name__ == "__main__":
    image_data, label_data = readfile()
    im = get_image(image_data)
    label = get_label(label_data)

    index = 0
    for i in range(6000):
        if label[i] != 7:
            continue

        plt.subplot(3, 3, index + 1)
        title = u"标签对应为："+ str(label[i])
        plt.title(title, fontproperties='SimHei')
        plt.imshow(im[i], cmap='gray')

        index += 1
        if index >= 9:
            break

    plt.show()