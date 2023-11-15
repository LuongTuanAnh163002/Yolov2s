import matplotlib.pyplot as plt
import random
import cv2
import torch

def plot_img(img, size=(7,7)):
    plt.figure(figsize=size)
    plt.imshow(img[:,:,::-1])
    plt.show()
    
# vẽ bounding box lên ảnh
def visualize_bbox(img, boxes, thickness=2, color=(0, 0, 255), label=None, draw_center=True):
    img_copy = img.cpu().permute(1,2,0).numpy() if isinstance(img, torch.Tensor) else img.copy()
    tl = thickness or round(0.002 * (img_copy.shape[0] + img_copy.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    for i, box in enumerate(boxes):
        x,y,w,h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img_copy = cv2.rectangle(
            img_copy,
            (x,y),(x+w, y+h),
            color, thickness)
        if draw_center:
            center = (x+w//2, y+h//2)
            img_copy = cv2.circle(img_copy, center=center, radius=3, color=(0,255,0), thickness=2)
            
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label[i], 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x + t_size[0], y - t_size[1] - 3
            cv2.rectangle(img_copy, (x,y),c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img_copy, label[i], (x, y - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img_copy

def plot_loss(train_loss, save_dir):
    plt.figure(figsize=(8, 8))
    plt.plot(train_loss)
    plt.title("Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    save_path = save_dir + "/" + "results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')