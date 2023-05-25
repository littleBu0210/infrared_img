#识别视频的每一帧，利用之前的算法

import cv2
import os
import main
import numpy as np

if __name__ == '__main__':
    temp = 0
    #图片地址
    video_path = './video'
    #地址下面所有的文件名
    filename = os.listdir(video_path)
    #照片数量
    len_file = len(filename)
    #遍历所有视频
    for i in range(len_file):
        path = video_path+'/'+filename[i]
        cap = cv2.VideoCapture(path)
        while(cap.isOpened()):
            # ret返回布尔值
            ret, frame = cap.read()
            
            if ret == True:
                img = cv2.resize(frame, (480, 640))
                #灰度化
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #去水印
                img_logo = main.watermark(img_gray)
                #求梯度
                img_dft = main.dft_HP(img_logo)
                #二值化，此处需要设置一个阈值200
                img_bin = cv2.threshold(img_dft,200,255,cv2.THRESH_BINARY)[1]
                #轮廓检测
                binary,contours, hierarchy= cv2.findContours(img_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                #绘制轮廓
                draw_img = img.copy()
                #绘制轮廓
                res = cv2.drawContours(draw_img, contours, -1, (255, 0, 0), 3) 
                #合并图像 
                fin = np.hstack((res,img)) 
                #展示图片
                cv2.imshow("fin", fin)               
                if(temp==0):
                    cv2.waitKey(4000)
                    temp=1
                cv2.waitKey(40)
            else:
                break
        # 释放资源
        cap.release()
        # 关闭窗口
        cv2.destroyAllWindows()