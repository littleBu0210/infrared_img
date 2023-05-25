#采用opencv实现单张图片单个目标的识别
#读入图片
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def Gray_3d(img,n):
    img_gray_down = img
    #降采样n次
    for i in range(n):
        img_gray_down = cv2.pyrDown(img_gray_down)  
    #捕获像素
    (x_max,y_max) = img_gray_down.shape
    #切片
    x = np.linspace(0,x_max-1,x_max,dtype=np.uint64)
    y = np.linspace(0,y_max-1,y_max,dtype=np.uint64)        
    #绘制三维灰度图
    X, Y = np.meshgrid(y,x)
    Z = img_gray_down
    #曲面图
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    ax.set_xlabel("x/pixel")
    ax.set_ylabel("y/pixel")
    ax.set_zlabel("Gray Value")
    plt.show()

def watermark(img):
    #遮挡logo
    x,y,w,h = 20,12,95,34
    logo = img[y:y+h,x:x+w]
    #二值化
    logo_bin = cv2.threshold(logo,100,255,cv2.THRESH_BINARY)[1]
    #创建全空数组
    mask = np.zeros_like(img)
    #赋值
    mask[y:y+h,x:x+w] = logo_bin
    #开始修复
    #方法1
    dst1 = cv2.inpaint(img,mask, 3,cv2.INPAINT_TELEA) 
    # #方法2 
    # dst2 = cv2.inpaint(img,mask, 3,cv2.cv2.INPAINT_NS) 
    return dst1

#频域高通滤波法
def dft_HP(src):
    # 傅立叶变化
    src_dft=cv2.dft(np.float32(src),flags=cv2.DFT_COMPLEX_OUTPUT)
    # 将图片中心从左上角移到中心
    src_dft_shift=np.fft.fftshift(src_dft)
    
    # 制作掩膜令中心为0,后面才能过滤掉中心低频
    rows,cols=src.shape
    crow,ccol=int(rows/2),int(cols/2)
    mask=np.ones((rows,cols,2),np.uint8)
    size = 20
    mask[crow-size:crow+size,ccol-size:ccol+size]=0
    
    # 用掩膜对图像进行处理
    src_dft_shift_over=src_dft_shift*mask
    
    #将中心移回左上角
    src_dft_shift_over_ishift=np.fft.ifftshift(src_dft_shift_over)
    
    # 傅立叶逆变换
    src_dft_shift_over_ishift_idft=cv2.idft(src_dft_shift_over_ishift)
    
    #后续操作,将矢量转换成标量,并映射到合理范围之内
    src_dft_shift_over_ishift_idft=cv2.magnitude(src_dft_shift_over_ishift_idft[:,:,0],src_dft_shift_over_ishift_idft[:,:,1])
    src_dft_shift_over_ishift_idft=np.abs(src_dft_shift_over_ishift_idft)
    src_dft_shift_over_ishift_idft=(src_dft_shift_over_ishift_idft-np.amin(src_dft_shift_over_ishift_idft))/(np.amax(src_dft_shift_over_ishift_idft)-np.amin(src_dft_shift_over_ishift_idft))
    fin = np.zeros_like(src_dft_shift_over_ishift_idft,dtype=np.uint8)
    fin[:,:] = src_dft_shift_over_ishift_idft*255
    return fin


if __name__ == '__main__':
    #图片地址
    pic_path = './pic'
    #地址下面所有的文件名
    filename = os.listdir(pic_path)
    #照片数量
    len_file = len(filename)
    #遍历每一张照片
    for i in range(len_file):
        img = cv2.imread(pic_path + "/" + filename[i])
        #灰度化
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #去水印
        img_logo = watermark(img_gray)
        #求梯度
        img_dft = dft_HP(img_logo)
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
        while(True):
            cv2.imshow("fin", fin)
            cv2.waitKey(1000)
            break
 