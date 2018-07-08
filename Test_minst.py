# coding=utf-8
from skimage import filters
import scipy.misc
from tkinter import *
import os, glob
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import model

TEST_PATH = r'D:\Python\mnist\test_my\ok1/*'

def select(event):
    global img_dir
    img_dir = lst_demo.get(lst_demo.curselection())
    print(img_dir + "----select----")
    photo["file"] = img_dir


'''将任意格式图片转化为黑白28*28图片'''
def pre_pic(dir,a):

    name = dir.split('/')[-1].split('.')[0]
    SAVE_PATH = name + '.png'
    print(SAVE_PATH)
    img = Image.open(dir)
    pic = img.resize((28,28),Image.ANTIALIAS)  #设置压缩尺寸和选项，注意尺寸要用括号
    pic.save(SAVE_PATH)

    img = Image.open(SAVE_PATH)
    im_arr = np.array(img.convert('L'))  # 变成灰度图片
    thresh = filters.threshold_otsu(im_arr)  # 返回一个阈值
    threshold = thresh


    '''如果是普通图片需要进行这个步骤'''
    if a == 'no_black':
        for i in range(28):  # 反色 im_arr[][]是二维数组
            for j in range(28):
                # im_arr[i][j] = 255 - im_arr[i][j]
                # im_arr[i][j] =  im_arr[i][j]
                if (im_arr[i][j] > threshold):
                    im_arr[i][j] = 0
                else:
                    im_arr[i][j] = 255
        scipy.misc.imsave(SAVE_PATH, im_arr)
    '''如果是mnist数据集那样的就不用执行这一步'''


    # print(SAVE_PATH)
    # img_ready = im_arr.reshape([1, 784])
    # return的是0-1的数据 不是0-255
    return SAVE_PATH
# 从训练集中选取一张图片
# def get_one_image(train):
#     # n = len(train)
#     # #从训练集中随机抽取一张图片
#     # ind = np.random.randint(0, n)
#     files = os.listdir(train)
#     n = len(files)
#     ind = np.random.randint(0, n)
#     img_dir = os.path.join(train, files[ind])
#     # print(img_dir)
#     image_t = Image.open(img_dir)
#     plt.imshow(image_t)
#     # plt.show()
#     #image = image.resize([28, 28])
#     #image = np.array(image)
#     #返回的是目录
#     return img_dir

def get_one_image(train):
    # #从训练集中随机抽取一张图片
    image = Image.open(img_dir)
    image = image.resize([28, 28])
    image = np.array(image)
    return image

def evaluate_one_image():
    # train_dir = '/Users/yangyibo/GitWork/pythonLean/AI/猫狗识别/testImg/'
    # train = r'D:\PyCharm_code\Ai\Tensorflow_mooc_note\6\MinstNew\data/'
    # 获取图片路径集和标签集
    # train, train_label = input_data.get_files(train_dir)
    # image_dir = get_one_image(test)
    image_dir = pre_pic(img_dir,'no_black')
    with tf.Graph().as_default():
        BATCH_SIZE = 1  # 因为只读取一副图片 所以batch 设置为1
        N_CLASSES = 10  # 10个输出神经元
        # 转化图片格式

        image_name = tf.cast(image_dir, tf.string)
        # 读取图片的全部信息
        image_contents = tf.read_file(image_name)
        image_t=Image.open(image_dir)
        image_t = image_t.resize([28, 28])
        image_t = np.array(image_t)
        # 把图片解码，channels ＝3 为彩色图片, r，g ，b  黑白图片为 1 ，也可以理解为图片的厚度
        image = tf.image.decode_jpeg(image_contents, channels=1)
        # 将图片以图片中心进行裁剪或者扩充为 指定的image_W，image_H
        image = tf.image.resize_image_with_crop_or_pad(image, 28, 28)
        # 对数据进行标准化,标准化，就是减去它的均值，除以他的方差
        image = tf.image.per_image_standardization(image)
        # 转化图片
        image = tf.cast(image, tf.float32)
        # 图片原来是三维的 [208, 208, 3] 重新定义图片形状 改为一个4D  四维的 tensor
        image = tf.reshape(image, [-1, 28, 28, 1])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        # 因为 inference 的返回没有用激活函数，所以在这里对结果用softmax 激活
        logit = tf.nn.softmax(logit)

        # 用最原始的输入数据的方式向模型输入数据 placeholder
        x = tf.placeholder(tf.float32, shape=[28, 28])

        # 我门存放模型的路径
        logs_train_dir = r'D:\PyCharm_code\Ai\Tensorflow_mooc_note\6\MinstNew\logs\train/'
        # 定义saver
        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("从指定的路径中加载模型。。。。")
            # 将模型加载到sess 中
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('模型加载成功, 训练的步数为 %s' % global_step)
            else:
                print('模型加载失败，，，文件没有找到')
                # 将图片输入到模型计算
            prediction = sess.run(logit, feed_dict={x: image_t})
            # 获取输出结果中最大概率的索引
            max_index = np.argmax(prediction)
            print('\n\n------------------------------------------------------')
            print("|经过 1%s 次训练,预测该图片值为 %g 预计正确率为 %g %% |" % (global_step, max_index, prediction[:,max_index]))
            print('------------------------------------------------------\n\n')
                # 测试

    return max_index, prediction[:,max_index]




def def_test():
    # if(output["fg"]=="blue"):
    #     output["fg"] = "red"
    # else:
    #     output["fg"] ="blue"
    text_a, text_b = evaluate_one_image()
    txt.set(text_a)
    txt2.set(text_b)



def def_list_picture():
    txt.set("")
    path = TEST_PATH
    for i in glob.glob(path):
        ss = []
        ss = i.split('\\')
        # L.append(ss[-1])
        L.append(i)


def windows():
    window = Tk()
    window.title('人工智能识别数字演示')
    window.resizable(False, False)  # 固定窗口大小
    windowWidth = 880  # 获得当前窗口宽
    windowHeight = 550  # 获得当前窗口高
    # windowWidth = 1000               #获得当前窗口宽
    # windowHeight = 800
    screenWidth, screenHeight = window.maxsize()  # 获得屏幕宽和高
    geometryParam = '%dx%d+%d+%d' % (
    windowWidth, windowHeight, (screenWidth - windowWidth) / 2, (screenHeight - windowHeight) / 2)
    window.geometry(geometryParam)  # 设置窗口大小及偏移坐标
    # window.wm_attributes('-topmost',1)#窗口置顶
    return window

test = ""
window = windows()
# 测试按钮
button_test = Button(window, text="开始识别", width=15, fg="blue", command=def_test)
button_test.grid(row=0, column=2, columnspan=2, padx=30, pady=15)

# 测试结果标签
label_result = Label(window, text="识别结果为：")
label_result.grid(row=1, column=2, padx=5, pady=0)
# 测试结果标签
label_result = Label(window, text="准确率为：")
label_result.grid(row=1, column=3, padx=5, pady=0)

# 测试结果输出
txt = StringVar()
output = Entry(window, width=10, state="readonly", fg="blue", textvariable=txt)
output.grid(row=2, column=2, padx=5, pady=5)

txt2 = StringVar()
output2 = Entry(window, width=13, state="readonly", fg="blue", textvariable=txt2)
output2.grid(row=2, column=3, padx=5, pady=5)

# 选项框
yscroll = Scrollbar(window, orient=VERTICAL)
yscroll.grid(row=0, column=1, rowspan=20, pady=10, sticky=NS)
L = []
def_list_picture()
c = StringVar()
lst_demo = Listbox(window, width=60, height=30, listvariable=c, yscrollcommand=yscroll.set)
lst_demo.grid(row=0, column=0, rowspan=20, padx=0, pady=0)
lst_demo.bind("<<ListboxSelect>>", select)
c.set(tuple(L))

yscroll["command"] = lst_demo.yview

# 图片框
photo = PhotoImage(format='png')
photo_Label = Label(window, justify=RIGHT, image=photo, width=400, height=400)
photo_Label.grid(row=10, column=2, columnspan=2, padx=10, pady=10)
photo_Label.after(100)
window.mainloop()


'''完美实现 数字识别'''