import numpy as np
import cv2 as cv2
from glob import glob
from matplotlib import pyplot as plt

EXAMPLE=""
ROI=[(100,400), (200,500)]
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

def imf(frame, example=EXAMPLE):#функция сравнения кадра из видео с эталоном
    imageExample = cv2.imread(example, 0)#
    
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

def get_roi(img, roi): # Фнукция для вывода определенной части изображения
    
    # img = cv2.imread(path, 0) 
    # print(img.shape)
    return img[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]

def get_painted_roi_on_frame(img, roi):
    cv2.rectangle(img, (roi[0][0], roi[1][0]), (roi[0][1], roi[1][1]), (255,255,255), 4)

    return img


if __name__ == "__main__": #Если запустить файл, то код будет читаться отсюда

    
    #1 - путь, а второе значение координат области изображения  
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, fframe = cap.read()
        img = cv2.cvtColor(fframe, cv2.COLOR_BGR2GRAY)
        img = np.flip(img, axis=1)
        img = img.copy() # error in opencv shell
        #print(img.shape)
        loli = get_roi(img, ROI)
        # painted_img = get_painted_roi_on_frame(img, ROI)
        cv2.rectangle(img, (ROI[0][1], ROI[1][1]), (ROI[0][0], ROI[1][0]), (255,255,255), 4)
        img = img.copy()
        #print(img.shape)
        # loli = cv2.imread('/Users/sanya/Work/test_img_cmp_proj_kvant/222/photo_1.jpeg', 0)#
        c = []
        c_val = []
        for ex in glob('/Users/sanya/Work/test_img_cmp_proj_kvant/222/*.jpeg'):
            c += [imf(loli, example=ex)]
            # c_val += [(imf(loli, example=ex), ex.split('/')[-1])]
            # print(c)
        min_c = min(c)
        print(min_c)

        cv2.imshow('output', img)


        key = cv2.waitKey(10)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

