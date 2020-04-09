import cv2
import numpy as np
import math
import os

PATH = 'F:/PythonProj/BitImgResample/sample'

# find images under specified path
def findImgFiles(path):
    files = os.listdir(path)
    imglist = []
    for file in files:
        filewithpath = os.path.join(path, file)
        if os.path.isfile(filewithpath):
            (filename, extension) = os.path.splitext(file)
            if extension == '.png':
                imglist.append(filewithpath)

    return imglist


#resize kernal
def ResizeKernel(s,a):
    if (abs(s) >=0) & (abs(s) <1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) >=1) & (abs(s) <2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

#Add Paddnig
def AddPadding(img,H,W,C):
    zimg = np.zeros((H+4,W+4,C))
    zimg[2:H+2,2:W+2,:C] = img
    zimg[2:H+2,0:2,:C]=img[:,0:1,:C]
    zimg[H+2:H+4,2:W+2,:]=img[H-1:H,:,:]
    zimg[2:H+2,W+2:W+4,:]=img[:,W-1:W,:]
    zimg[0:2,2:W+2,:C]=img[0:1,:,:C]
    zimg[0:2,0:2,:C]=img[0,0,:C]
    zimg[H+2:H+4,0:2,:C]=img[H-1,0,:C]
    zimg[H+2:H+4,W+2:W+4,:C]=img[H-1,W-1,:C]
    zimg[0:2,W+2:W+4,:C]=img[0,W-1,:C]
    return zimg

# Resize operatio
def ImageResize(img, ratio_h=0.5,ratio_w=0.5,a=-0.5):
    H,W,C = img.shape
    img = AddPadding(img,H,W,C)
    dH = math.floor(H*ratio_h)
    dW = math.floor(W*ratio_w)
    dst = np.zeros((dH, dW, C))

    h = 1/ratio_h
    w = 1/ratio_w

    inc = 0
    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                x, y = i * w + 2 , j * h + 2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                mat_l = np.matrix([[ResizeKernel(x1,a),ResizeKernel(x2,a),ResizeKernel(x3,a),ResizeKernel(x4,a)]])
                mat_m = np.matrix([[img[int(y-y1),int(x-x1),c],img[int(y-y2),int(x-x1),c],img[int(y+y3),int(x-x1),c],img[int(y+y4),int(x-x1),c]],
                                   [img[int(y-y1),int(x-x2),c],img[int(y-y2),int(x-x2),c],img[int(y+y3),int(x-x2),c],img[int(y+y4),int(x-x2),c]],
                                   [img[int(y-y1),int(x+x3),c],img[int(y-y2),int(x+x3),c],img[int(y+y3),int(x+x3),c],img[int(y+y4),int(x+x3),c]],
                                   [img[int(y-y1),int(x+x4),c],img[int(y-y2),int(x+x4),c],img[int(y+y3),int(x+x4),c],img[int(y+y4),int(x+x4),c]]])
                mat_r = np.matrix([[ResizeKernel(y1,a)],[ResizeKernel(y2,a)],[ResizeKernel(y3,a)],[ResizeKernel(y4,a)]])
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m),mat_r)
    return dst


# batch image resize
imagefiles = findImgFiles(PATH)
imageid = 0
nTotal = len(imagefiles)
index = 0
for file in imagefiles:
    index = index + 1
    print("正在处理 %d/%d:%s" % (index, nTotal, file))

    (filepath, tempfilename) = os.path.split(file)
    (filename, extension) = os.path.splitext(tempfilename)
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    dst = ImageResize(img, 0.5, 0.5, -0.5)

    result_file = filepath + "//" + filename + "_resize.png"
    cv2.imwrite(result_file, dst)
print('Completed!')