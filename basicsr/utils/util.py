from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import cv2
import Imath, OpenEXR


def delPadding(input_image, top_pad, bottom_pad, left_pad, right_pad):
    if bottom_pad == 0:
        bottom_pad = None
    if right_pad == 0:
        right_pad = None
    if isinstance(input_image, np.ndarray):
        img_delpad = input_image[:, top_pad:bottom_pad, left_pad:right_pad, :]
    else:
        img_delpad = input_image[:, :, top_pad:bottom_pad, left_pad:right_pad]
    
    return img_delpad


def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    return (batch - mean) / std


def tensor2hdr(input_image):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(np.float32)


def hdr2tonemapped(hdr_img):
    min_v, max_v = hdr_img.min(), hdr_img.max()
    norm_img = (hdr_img-min_v) / (max_v-min_v)
    tonemapped = np.log(1+5000.0*norm_img) / np.log(1+5000.0)
    return (tonemapped*255).astype(np.uint8)


def tensor_tonemap(image_tensor):
    # Input shape: [batch_size, channels, height, width]
    tonemapped = torch.log1p(5000.0*image_tensor) / torch.log1p(torch.tensor(5000.0).cuda())
    
    return tonemapped
    

def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array        
        if image_numpy.shape[0] == 1:  # attention map
            att_map = np.transpose(image_numpy, (1, 2, 0))
            return np.uint8(att_map*255)
        else: # no spikes
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
        # ----------- from [-1,1] to [0,1] ---------------
        image_numpy = (image_numpy + 1.0) / 2.0  # post-processing: tranpose and scaling
        # image_numpy = image_numpy**(1/2.2) * 255.0
        image_numpy = image_numpy * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image[0]
        image_numpy = image_numpy**(1/2.2) * 255.0

    return image_numpy.astype(imtype)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def rgbe2float(rgbe):
    res = np.zeros((rgbe.shape[0], rgbe.shape[1], 3))
    p = rgbe[:,:,3] > 0
    m = 2.0**(rgbe[:,:,3][p]-136.0)
    res[:,:,0][p] = rgbe[:,:,0][p] * m 
    res[:,:,1][p] = rgbe[:,:,1][p] * m 
    res[:,:,2][p] = rgbe[:,:,2][p] * m
    return res
    
    
def readHdr(fileName):
    fileinfo = {}
    with open(fileName, 'rb') as fd:
        tline = fd.readline().strip()
        # print((tline))

        if len(tline)<3 or tline[:2] != b'#?':
            # print ('invalid header')  
            return
        fileinfo['identifier'] = tline[2:]
 
        tline = fd.readline().strip()
        while tline:
            n = tline.find(b'=')
            if n>0:
                fileinfo[tline[:n].strip()] = tline[n+1:].strip()
            tline = fd.readline().strip()
 
        tline = fd.readline().strip().split(b' ')
        # print(tline)
        fileinfo['Ysign'] = tline[0][0]
        fileinfo['height'] = int(tline[1])
        fileinfo['Xsign'] = tline[2][0]
        fileinfo['width'] = int(tline[3])
        
        d = fd.read(1)
        data = []
        while d:
            data.append(ord(d))
            d = fd.read(1)
        # data = [ord(d) for d in fd.read()]
        height, width = fileinfo['height'], fileinfo['width']

        # print(len(data))

        img = np.zeros((height, width, 4))
        dp = 0
        for h in range(height):
            if data[dp] !=2 or data[dp+1]!=2:
                print ('this file is not run length encoded')
                print (data[dp:dp+4])
                return
            if data[dp+2]*256+ data[dp+3] != width:
                print ('wrong scanline width')
                return
            dp += 4
            for i in range(4):
                ptr = 0
                while(ptr < width):
                    if data[dp]>128:
                        count = data[dp]-128
                        if count==0 or count>width-ptr:
                            print ('bad scanline data')
                        img[h, ptr:ptr+count,i] = data[dp+1]
                        ptr += count
                        dp += 2
                    else:
                        count = data[dp]
                        dp += 1
                        if count==0 or count>width-ptr:
                            print ('bad scanline data')
                        img[h, ptr:ptr+count,i] = data[dp: dp+count]
                        ptr += count
                        dp +=count
    return img


def rbg2gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])

class IOException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def readEXR(hdrfile):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    hdr_t = OpenEXR.InputFile(hdrfile)
    dw = hdr_t.header()['dataWindow']
    size = (dw.max.x-dw.min.x+1, dw.max.y-dw.min.y+1)
    rstr = hdr_t.channel('R', pt)
    gstr = hdr_t.channel('G', pt)
    bstr = hdr_t.channel('B', pt)
    r = np.fromstring(rstr, dtype=np.float32)
    r.shape = (size[1], size[0])
    g = np.fromstring(gstr, dtype=np.float32)
    g.shape = (size[1], size[0])
    b = np.fromstring(bstr, dtype=np.float32)
    b.shape = (size[1], size[0])
    res = np.stack([r,g,b], axis=-1)
    imhdr = np.asarray(res)
    return imhdr

def writeEXR(img, file):
    try:
        img = np.squeeze(img)
        sz = img.shape
        header = OpenEXR.Header(sz[1], sz[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        out = OpenEXR.OutputFile(file, header)
        R = (img[:,:,0]).astype(np.float16).tostring()
        G = (img[:,:,1]).astype(np.float16).tostring()
        B = (img[:,:,2]).astype(np.float16).tostring()
        out.writePixels({'R' : R, 'G' : G, 'B' : B})
        out.close()
    except Exception as e:
        raise IOException("Failed writing EXR: %s"%e)
        
        
def whiteBalance(img):
    h, w = img.shape[:2]
    img = np.transpose(img, (2,0,1))
    img = np.reshape(img, (3,-1))
    r_max = img[0].max()
    g_max = img[1].max()
    b_max = img[2].max()
    mat = [[g_max/r_max,0,0], [0,1,0], [0,0,g_max/b_max]]
    img_wb = np.dot(mat, img)
    img_wb = np.reshape(img_wb, (3, h, w))
    img_wb = np.transpose(img_wb, (1,2,0))

    return img_wb 


def whiteBalance_mat(img, mat):
    h, w = img.shape[:2]
    img = np.transpose(img, (2,0,1))
    img = np.reshape(img, (3, -1))

    img_wb = np.dot(mat, img)
    img_wb = np.reshape(img_wb, (3, h, w))
    img_wb = np.transpose(img_wb, (1,2,0))
    
    return img_wb


def tensor_tonemap(image_tensor):
    # Input shape: [batch_size, channels, height, width]
    tonemapped = torch.log1p(5000.0 * image_tensor) / torch.log1p(torch.tensor(5000.0).cuda())

    return tonemapped

def tonemap(img, log_sum_prev=None):
    key_fac, epsilon, tm_gamma = 0.5, 1e-6, 1.4
    XYZ = BGR2XYZ(img)
    b, c, h, w = XYZ.shape
    if log_sum_prev is None:
        log_sum_prev = torch.log(epsilon + XYZ[:, 0, :, :]).sum((1, 2), keepdim=True)
        log_avg_cur = torch.exp(log_sum_prev / (h * w))
        key = key_fac
    else:
        log_sum_cur = torch.log(XYZ[:, 1, :, :] + epsilon).sum((1, 2), keepdim=True)
        log_avg_cur = torch.exp(log_sum_cur / (h * w))
        log_avg_temp = torch.exp((log_sum_cur + log_sum_prev) / (2.0 * h * w))
        key = key_fac * log_avg_cur / log_avg_temp
        log_sum_prev = log_sum_cur
    Y = XYZ[:, 1, :, :]
    Y = Y / log_avg_cur * key
    Lmax = torch.max(torch.max(Y, 1, keepdim=True)[0], 2, keepdim=True)[0]
    L_white2 = Lmax * Lmax
    L = Y * (1 + Y / L_white2) / (1 + Y)
    XYZ *= (L / XYZ[:, 1, :, :]).unsqueeze(1)
    image = XYZ2BGR(XYZ)
    image = torch.clamp(image, 0, 1) ** (1 / tm_gamma)
    return image, log_sum_prev


def BGR2XYZ(image):
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    b = image[..., 0, :, :]
    g = image[..., 1, :, :]
    r = image[..., 2, :, :]

    X = (0.4124 * r) + (0.3576 * g) + (0.1805 * b)
    Y = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)
    Z = (0.0193 * r) + (0.1192 * g) + (0.9505 * b)

    return torch.stack((X, Y, Z), -3)


def XYZ2BGR(image):
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    X = image[..., 0, :, :]
    Y = image[..., 1, :, :]
    Z = image[..., 2, :, :]

    r = (3.240625 * X) + (-1.537208 * Y) + (-0.498629 * Z)
    g = (-0.968931 * X) + (1.875756 * Y) + (0.041518 * Z)
    b = (0.055710 * X) + (-0.204021 * Y) + (1.056996 * Z)

    return torch.stack((b, g, r), -3)
        
        