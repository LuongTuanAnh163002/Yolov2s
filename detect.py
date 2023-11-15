import argparse
import torch
import torch.nn as nn
import cv2
from pathlib import Path
import time
import numpy as np

from models.yolo import YOLO
from utils.general import increment_path
from utils.datasets import LoadImages, output_tensor_to_boxes
from utils.metrics import nonmax_suppression
from utils.plots import visualize_bbox

def detect(opt, save_img=True):
  source, weight, confident_score = opt.source, opt.weight, opt.conf_thres
  img_size, device_number = opt.img_size, opt.device
  save_dir = opt.save_dir
  Path(save_dir).mkdir(parents=True, exist_ok=True)

  checkpoint = torch.load(weight)
  device = torch.device("cuda:"+str(device_number)) if torch.cuda.is_available() else torch.device("cpu")
  model = YOLO(S=13, BOX=5, CLS=checkpoint["number_class"])
  model.to(device)
  model.load_state_dict(checkpoint["state_dict"])
  model.eval()

  nc = checkpoint["number_class"]
  class_name = checkpoint["class_name"]
  anchor_box=[[1.08,1.19],
              [3.42,4.41],
              [6.63,11.38],
              [9.42,5.11],
              [16.62,10.52]]
  
  vid_path, vid_writer = None, None
  dataset = LoadImages(source, img_size=img_size)
  t0 = time.time()
  for path, im, im0s, vid_cap in dataset:
    img = im.astype(np.float32)/255.0
    img = cv2.resize(img, (img_size, img_size))
    img_tensor = torch.from_numpy(img).unsqueeze(0)
    img_tensor = img_tensor.permute(0, 3, 1, 2)
  
    output = model(img_tensor.to(device))
    boxes, class_id = output_tensor_to_boxes(output[0].detach().cpu(), confident_score, anchor_box)
    boxes = nonmax_suppression(boxes)
    labels = [class_name[i] for i in class_id]
    img = img_tensor[0].permute(1,2,0).cpu().numpy()
    im0s = cv2.resize(im0s, (img_size, img_size))
    img0 = visualize_bbox(im0s, boxes=boxes, label=labels)
    p = Path(path)
    save_path = str(Path(save_dir) / p.name)
    if save_img:
        if dataset.mode == 'image':
            cv2.imwrite(save_path, img0)
            print(f" The image with the result is saved in: {save_path}")
        else:  # 'video' or 'stream'
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, img.shape[1], img.shape[0]
                    save_path += '.mp4'
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(img)

    print(f'Done. ({time.time() - t0:.3f}s)')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='image, video', help='data.yaml path')
    parser.add_argument('--weight', type=str, default='weight of yolov1', help='data.yaml path')
    parser.add_argument('--conf_thres', type=float, default=0.1, help='confident threshold')
    parser.add_argument('--img_size', type=int, default=416, help='[train, test] image sizes')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/detect', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    opt = parser.parse_args()
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
    detect(opt)