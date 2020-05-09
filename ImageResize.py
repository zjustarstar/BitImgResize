import cv2
import numpy as np
import math
import os
from sklearn import cluster


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


#bell kernal
def BellResizeKernel(s,a):
    if (abs(s) <0.5):
        return 0.75-abs(s)**2
    elif (abs(s) >0.5) & (abs(s) <1.5):
        return 0.5*((abs(s)-1.5)**2)
    return 0

#Hermite kernal
def HermiteResizeKernel(s,a):
    if (abs(s) <=1):
        return 2*abs(s)**3-3*abs(s)**2+1
    return 0

#Mitchell kernal
B=1.0/3.0
C=1.0/3.0
def MitchellResizeKernel(s,a):
    if (abs(s) <1):
        return 1.0/6.0*((12-9*B-6*C)*abs(s)**3+(-18+12*B+6*C)*abs(s)**2+(6-2*B))
    elif (abs(s) >=1) & (abs(s) <2):
        return 1.0/6.0*((-B-6*C)*abs(s)**3+(6*B+30*C)*abs(s)**2+(-12*B-48*C)*abs(s)+(8*B+24*C))
    return 0

#resize kernal
def ResizeKernel(s,a):
    if (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) >1) & (abs(s) <2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

#Add Paddnig
pad_t=4
pad_h=2
def AddPadding(img,H,W,C):
    zimg = np.zeros((H+pad_t,W+pad_t,C))
    zimg[pad_h:H+pad_h,pad_h:W+pad_h,:C] = img
    zimg[pad_h:H+pad_h,0:pad_h,:C]=img[:,0:1,:C]
    zimg[H+pad_h:H+pad_t,pad_h:W+pad_h,:]=img[H-1:H,:,:]
    zimg[pad_h:H+pad_h,W+pad_h:W+pad_t,:]=img[:,W-1:W,:]
    zimg[0:pad_h,pad_h:W+pad_h,:C]=img[0:1,:,:C]
    zimg[0:pad_h,0:pad_h,:C]=img[0,0,:C]
    zimg[H+pad_h:H+pad_t,0:pad_h,:C]=img[H-1,0,:C]
    zimg[H+pad_h:H+pad_t,W+pad_h:W+pad_t,:C]=img[H-1,W-1,:C]
    zimg[0:pad_h,W+pad_h:W+pad_t,:C]=img[0,W-1,:C]
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

                mat_l = np.array([[ResizeKernel(x1,a),ResizeKernel(x2,a),ResizeKernel(x3,a),ResizeKernel(x4,a)]])
                mat_m = np.array([[img[int(y-y1),int(x-x1),c],img[int(y-y2),int(x-x1),c],img[int(y+y3),int(x-x1),c],img[int(y+y4),int(x-x1),c]],
                                   [img[int(y-y1),int(x-x2),c],img[int(y-y2),int(x-x2),c],img[int(y+y3),int(x-x2),c],img[int(y+y4),int(x-x2),c]],
                                   [img[int(y-y1),int(x+x3),c],img[int(y-y2),int(x+x3),c],img[int(y+y3),int(x+x3),c],img[int(y+y4),int(x+x3),c]],
                                   [img[int(y-y1),int(x+x4),c],img[int(y-y2),int(x+x4),c],img[int(y+y3),int(x+x4),c],img[int(y+y4),int(x+x4),c]]])
                mat_r = np.array([[ResizeKernel(y1,a)],[ResizeKernel(y2,a)],[ResizeKernel(y3,a)],[ResizeKernel(y4,a)]])
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m),mat_r)
    return dst

def BellResize(img, ratio_h=0.5,ratio_w=0.5,a=-0.5):
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

                mat_l = np.array([[BellResizeKernel(x1,a),BellResizeKernel(x2,a),BellResizeKernel(x3,a),BellResizeKernel(x4,a)]])
                mat_m = np.array([[img[int(y-y1),int(x-x1),c],img[int(y-y2),int(x-x1),c],img[int(y+y3),int(x-x1),c],img[int(y+y4),int(x-x1),c]],
                                   [img[int(y-y1),int(x-x2),c],img[int(y-y2),int(x-x2),c],img[int(y+y3),int(x-x2),c],img[int(y+y4),int(x-x2),c]],
                                   [img[int(y-y1),int(x+x3),c],img[int(y-y2),int(x+x3),c],img[int(y+y3),int(x+x3),c],img[int(y+y4),int(x+x3),c]],
                                   [img[int(y-y1),int(x+x4),c],img[int(y-y2),int(x+x4),c],img[int(y+y3),int(x+x4),c],img[int(y+y4),int(x+x4),c]]])
                mat_r = np.array([[BellResizeKernel(y1,a)],[BellResizeKernel(y2,a)],[BellResizeKernel(y3,a)],[BellResizeKernel(y4,a)]])
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m),mat_r)
    return dst

def HermiteResize(img, ratio_h=0.5,ratio_w=0.5,a=-0.5):
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

                mat_l = np.array([[HermiteResizeKernel(x1,a),HermiteResizeKernel(x2,a),HermiteResizeKernel(x3,a),HermiteResizeKernel(x4,a)]])
                mat_m = np.array([[img[int(y-y1),int(x-x1),c],img[int(y-y2),int(x-x1),c],img[int(y+y3),int(x-x1),c],img[int(y+y4),int(x-x1),c]],
                                   [img[int(y-y1),int(x-x2),c],img[int(y-y2),int(x-x2),c],img[int(y+y3),int(x-x2),c],img[int(y+y4),int(x-x2),c]],
                                   [img[int(y-y1),int(x+x3),c],img[int(y-y2),int(x+x3),c],img[int(y+y3),int(x+x3),c],img[int(y+y4),int(x+x3),c]],
                                   [img[int(y-y1),int(x+x4),c],img[int(y-y2),int(x+x4),c],img[int(y+y3),int(x+x4),c],img[int(y+y4),int(x+x4),c]]])
                mat_r = np.array([[HermiteResizeKernel(y1,a)],[HermiteResizeKernel(y2,a)],[HermiteResizeKernel(y3,a)],[HermiteResizeKernel(y4,a)]])
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m),mat_r)
    return dst

def MitchellResize(img, ratio_h=0.5,ratio_w=0.5,a=-0.5):
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

                mat_l = np.array([[MitchellResizeKernel(x1,a),MitchellResizeKernel(x2,a),MitchellResizeKernel(x3,a),MitchellResizeKernel(x4,a)]])
                mat_m = np.array([[img[int(y-y1),int(x-x1),c],img[int(y-y2),int(x-x1),c],img[int(y+y3),int(x-x1),c],img[int(y+y4),int(x-x1),c]],
                                   [img[int(y-y1),int(x-x2),c],img[int(y-y2),int(x-x2),c],img[int(y+y3),int(x-x2),c],img[int(y+y4),int(x-x2),c]],
                                   [img[int(y-y1),int(x+x3),c],img[int(y-y2),int(x+x3),c],img[int(y+y3),int(x+x3),c],img[int(y+y4),int(x+x3),c]],
                                   [img[int(y-y1),int(x+x4),c],img[int(y-y2),int(x+x4),c],img[int(y+y3),int(x+x4),c],img[int(y+y4),int(x+x4),c]]])
                mat_r = np.array([[MitchellResizeKernel(y1,a)],[MitchellResizeKernel(y2,a)],[MitchellResizeKernel(y3,a)],[MitchellResizeKernel(y4,a)]])
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m),mat_r)
    return dst


#image color quantization
#主要采用kmeans聚类，然后像素填充
def colorquantize(raster, n_colors):
    width, height, depth = raster.shape
    reshaped_raster = np.reshape(raster, (width * height, depth))
    model = cluster.KMeans(n_clusters=n_colors)
    labels = model.fit_predict(reshaped_raster)
    palette = model.cluster_centers_
    quantized_raster = np.reshape(palette[labels], (width, height, palette.shape[1])).astype('ubyte')
    return quantized_raster


#test batch image resize
path = 'F:/PythonProj/BitImgResample/sample'
imagefiles = findImgFiles(path)
imageid = 0
nTotal = len(imagefiles)
index = 0
for file in imagefiles:
    index = index + 1
    print("正在处理 %d/%d:%s" % (index, nTotal, file))

    (filepath, tempfilename) = os.path.split(file)
    (filename, extension) = os.path.splitext(tempfilename)

    # image resize
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    dst = ImageResize(img, 0.5, 0.5, -0.5)
    result_file = filepath + "//" + filename + "_resize.png"
    cv2.imwrite(result_file, dst)

    # color quantize
    k = 8
    temp = colorquantize(dst, k)
    result_file2 = filepath + "//" + filename + "_" + str(k) + ".png"
    cv2.imwrite(result_file2, temp)


print('Completed!')