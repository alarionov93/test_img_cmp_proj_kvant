import requests
import numpy as np
import cv2 as cv2
from glob import glob
from time import sleep
from matplotlib import pyplot as plt

EXAMPLE=""

Y0, X0, Y1, X1 = 100, 400, 400, 700
SIZE=8
e=2.78

def σ(x):
    return 1 / (1 + e**(-x))

def compare(img1, img2):
    D = []
    for x in range(SIZE):
        for y in range(SIZE):
            a = σ(img1[y,x] / 255)
            b = σ(img2[y,x] / 255)
            D += [(a-b)**2]
    return sum(D)**1/2

def imf(frame: 'np.array', imageExample: 'np.array') -> float:#функция сравнения кадра из видео с эталоном
    avg=imageExample.mean()#функция mean ищет среднее арифметическое масива изображения 
    avg1=frame.mean()
    # print(avg, avg1)
    
    s,threshold_image=cv2.threshold(imageExample,avg,255, 0)#Функция threshold возвращает изображение, в котором все пиксели, которые темнее 127 заменены на 0, а все, которые ярче 127, — на 255
    d,threshold_frame=cv2.threshold(frame,avg1,255, 0)#Функция threshold возвращает изображение, в котором все пиксели, которые темнее 127 заменены на 0, а все, которые ярче 127, — на 255
    
    resized_image = cv2.resize(threshold_image, (SIZE,SIZE), interpolation = cv2.INTER_AREA) #Уменьшим картинку,INTER_AREA это передискретизации с использованием отношения площади пикселя, понятия не имею что это, но оно работае
    resized_frame = cv2.resize(threshold_frame, (SIZE,SIZE), interpolation = cv2.INTER_AREA) #Уменьшим картинку,INTER_AREA это передискретизации с использованием отношения площади пикселя, понятия не имею что это, но оно работае

    # z = (resized_frame-resized_image)**2
    # similar_coeff = np.sum(z)
    # cv2.imshow('aaaa', threshold_image)
    cv2.imshow('aaaa', threshold_frame)
    similar_coeff = compare(resized_frame, resized_image)
    
    return similar_coeff


    #print(imf("/home/sophia/Изображения/IMG-1544.jpg"))#кадры из видео

def get_roi(img): # Фнукция для вывода определенной части изображения
    
    # img = cv2.imread(path, 0) 
    # print(img.shape)
    return img[Y0:Y1, X0:X1]

def get_painted_roi_on_frame(img):
    cv2.rectangle(img, (X0,Y0), (X1, Y1), (255,255,255), 4)

    return img


if __name__ == "__main__": #Если запустить файл, то код будет читаться отсюда

    last_cmd = ''
    cmd = ''    
    #1 - путь, а второе значение координат области изображения  
    files = glob('examples/img/*.png')
    print(files)
    exxs = [cv2.imread(exx, 0) for exx in files]
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, fframe = cap.read()
        img = cv2.cvtColor(fframe, cv2.COLOR_BGR2GRAY)
        img = np.flip(img, axis=1)
        img = img.copy() # error in opencv shell
        #print(img.shape)
        loli = get_roi(img)
        get_painted_roi_on_frame(img)
        # img = img.copy()
        #print(img.shape)
        # loli = cv2.imread('/Users/sanya/Work/test_img_cmp_proj_kvant/222/photo_1.jpeg', 0)#
        c = []
        c_val = []
        for ex in exxs:
            c += [imf(loli, ex)]
            # print(c)
        min_c = min(c)
        try:
            if min_c < 0.2:
                # print((min_c, files[c.index(min_c)].split('/')[-1]))
                cmd = files[c.index(min_c)].split('/')[-1].split('.')[0][:-1]
                if cmd != last_cmd:
                    requests.get('http://192.168.10.175:8000/input?cmd=%s' % cmd)
                    print((min_c, files[c.index(min_c)].split('/')[-1]))
                last_cmd = cmd
                # sleep(1)
        except ValueError:
            pass
        # print([ round(x,2) for x in c ])

        cv2.imshow('output', img)


        key = cv2.waitKey(10)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


