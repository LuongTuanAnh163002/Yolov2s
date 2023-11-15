import torch
import cv2
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from random import randint, choice, shuffle, choices
import random
from pathlib import Path
import glob
from tqdm import tqdm


def read_data_yolo_format(trainlb_dir, vallb_dir):
    data = []
    list_file_txt_train = os.listdir(trainlb_dir)
    list_path_lb_train = trainlb_dir.split("/")
    list_path_lb_train[-2] = "images"
    trainimg_dir = "/".join(list_path_lb_train)
    data = []
    class_list_id = []
    for i in tqdm(list_file_txt_train, "Loading data..."):
        img_data = {}
        img_train_path = trainimg_dir+"/"+i[:-4] +".jpg"
        img = cv2.imread(img_train_path)
        H,W = img.shape[0:2]
        if os.path.exists(img_train_path)==True:
            img_data['file_path'] = i[:-4] +".jpg"
            with open(trainlb_dir+"/"+i, 'r') as f:
                input_text = [x.split() for x in f.read().strip().splitlines()]
                img_data['box_nb'] = len(input_text)
                new_input_text = []
                for box in input_text:
                    lb = int(box[0])
                    x1 = int((float(box[1])*W) - ((float(box[3])*W)/2))
                    y1 = int((float(box[2])*H) - ((float(box[4])*H)/2))
                    w = int(float(box[3])*W)
                    h = int(float(box[4])*H)
                    class_list_id.append(lb)
                    new_input_text.append([lb, x1, y1, w, h])
                img_data['boxes'] = new_input_text
        if len(img_data['boxes']) > 0:
            data.append(img_data)
    
    return data, class_list_id

def get_xywh_from_textline(text):
    coor = text.split(" ")
    result = []
    xywh = [int(coor[i]) for i in range(4)] if(len(coor) > 4) else None
    return xywh

def read_data_voc_format(file_path, nb_max=10):
    '''
    Read data in .txt file
    Return:
        list of image_data,
        each element is dict {file_path: , box_nb: , boxes:}
    '''
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, cur_line in enumerate(lines):
            if '.jpg' in cur_line:
                img_data = {
                    'file_path': cur_line.strip(),
                    'box_nb': int(lines[i+1]),
                    'boxes': [],
                }
                
                face_nb = img_data['box_nb']
                if(face_nb <= nb_max):
                    for j in range(face_nb):
                        rect = get_xywh_from_textline(lines[i+2+j].replace("\n", ""))
                        if rect is not None:
                            img_data['boxes'].append(rect)
                    if len(img_data['boxes']) > 0:
                        data.append(img_data)
    return data

class CustomDataset(Dataset):
    def __init__(self, data, number_class, S=13, BOX=5, img_size = 416, img_dir='', transforms=None):
        self.data = data
        self.img_dir = img_dir
        self.transforms = transforms
        self.number_class = number_class
        self.S = S
        self.BOX = BOX
        self.img_size = img_size
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, id):
        """"""
        img_data = self.data[id]
        img_fn = f"{self.img_dir}/{img_data['file_path']}"
        boxes = img_data["boxes"]
        box_nb = img_data["box_nb"]
        labels = torch.zeros((box_nb, self.number_class), dtype=torch.int64)
        for i in range(len(boxes)):
            labels[i][boxes[i][0]] = 1
        img = cv2.imread(img_fn).astype(np.float32)/255.0
        try:
            boxes = np.array(boxes)
            boxes = boxes[:, 1:]
            boxes = boxes.tolist()
        except:
            print(boxes)
        try:
            if self.transforms:
                sample = self.transforms(**{
                    "image":img,
                    "bboxes": boxes,
                    "labels": labels,
                })
                img = sample['image']
                boxes = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        except:
            return self.__getitem__(randint(0, len(self.data)-1))
        target_tensor = self.boxes_to_tensor(boxes.type(torch.float32), labels)
        return img, target_tensor
    
    def boxes_to_tensor(self, boxes, labels):
        """
        Convert list of boxes (and labels) to tensor format
        Output:
            boxes_tensor: shape = (Batchsize, S, S, Box_nb, (4+1+CLS))
        """
        boxes_tensor = torch.zeros((self.S, self.S, self.BOX, 5+self.number_class))
        cell_w, cell_h = self.img_size/self.S, self.img_size/self.S
        for i, box in enumerate(boxes):
            x,y,w,h = box
            # normalize xywh with cell_size
            x,y,w,h = x/cell_w, y/cell_h, w/cell_w, h/cell_h
            center_x, center_y = x+w/2, y+h/2
            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))
            
            if grid_x < self.S and grid_y < self.S:
                boxes_tensor[grid_y, grid_x, :, 0:4] = torch.tensor(self.BOX * [[center_x-grid_x,center_y-grid_y,w,h]])
                boxes_tensor[grid_y, grid_x, :, 4]  = torch.tensor(self.BOX * [1.])
                boxes_tensor[grid_y, grid_x, :, 5:]  = torch.tensor(self.BOX*[labels[i].numpy()])
        return boxes_tensor

def collate_fn(batch):
    return tuple(zip(*batch))

def target_tensor_to_boxes(boxes_tensor, confidence_score, S=13, BOX=5, img_size=416):
    '''
    Recover target tensor (tensor output of dataset) to bboxes
    Input:
        boxes_tensor: bboxes in tensor format - output of dataset.__getitem__
    Output:
        boxes: list of box, each box is [x,y,w,h]
    '''
    cell_w, cell_h = img_size/S, img_size/S
    boxes = []
    for i in range(S):
        for j in range(S):
            for b in range(BOX):
                data = boxes_tensor[i,j,b]
                x_center,y_center, w, h, obj_prob, cls_prob = data[0], data[1], data[2], data[3], data[4], data[5:]
                prob = obj_prob*max(cls_prob)
                if prob > confidence_score:
                    x, y = x_center+j-w/2, y_center+i-h/2
                    x,y,w,h = x*cell_w, y*cell_h, w*cell_w, h*cell_h
                    box = [x,y,w,h]
                    boxes.append(box)
    return boxes

def output_tensor_to_boxes(boxes_tensor, confidence_score, anchor_boxs,
                          S=13, BOX=5, img_size=416):
    cell_w, cell_h = img_size/S, img_size/S
    boxes = []
    probs = []
    class_index = []
    for i in range(S):
        for j in range(S):
            for b in range(BOX):
                anchor_wh = torch.tensor(anchor_boxs[b])
                data = boxes_tensor[i,j,b]
                xy = torch.sigmoid(data[:2])
                wh = torch.exp(data[2:4])*anchor_wh
                obj_prob = torch.sigmoid(data[4:5])
                cls_prob = torch.softmax(data[5:], dim=-1)
                combine_prob = obj_prob*max(cls_prob)
                
                if combine_prob > confidence_score:
                    x_center, y_center, w, h = xy[0], xy[1], wh[0], wh[1]
                    x, y = x_center+j-w/2, y_center+i-h/2
                    x,y,w,h = x*cell_w, y*cell_h, w*cell_w, h*cell_h
                    box = [x,y,w,h, combine_prob]
                    boxes.append(box)
                    class_index.append(torch.argmax(cls_prob))
    return boxes, class_index

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
class LoadImages:  # for inference
    def __init__(self, path, img_size=416):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            img = img0.copy()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()
                    img = img0.copy()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            img = cv2.imread(path)
            assert img0 is not None, 'Image Not Found ' + path
            #print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Convert


        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files