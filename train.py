import torch
import torchvision
import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from torchsummary import summary
import time
import yaml
from tqdm import tqdm
import warnings
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

from utils.datasets import read_data_yolo_format, read_data_voc_format, CustomDataset
from utils.datasets import collate_fn
from utils.general import select_device, increment_path
from utils.loss import custom_loss 
from utils.plots import plot_loss
from models.yolo import YOLO


warnings.filterwarnings("ignore") #remove warning

def train(opt, tb_writer):
  epochs, batch_size, device_number, workers = opt.epochs, \
    opt.batch_size, opt.device, opt.workers
  
  data_format, weight_ckpt = opt.data_format, opt.weight_ckpt

  save_dir = opt.save_dir
  wdir = Path(save_dir) / 'weights'
  wdir.mkdir(parents=True, exist_ok=True)
  last = wdir / 'last.pth'
  init = wdir / 'init.pth'

  with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
  train_path = data_dict['train']
  val_path = data_dict['val']
  nc = data_dict['nc']
  class_name = data_dict['names']
  voc_path_txt = data_dict['voc_path_txt']


  list_train_img_path = train_path.split("/")
  list_train_img_path[-2] = "labels"
  train_label_path = "/".join(list_train_img_path)

  list_val_img_path = val_path.split("/")
  list_val_img_path[-2] = "labels"
  val_label_path = "/".join(list_val_img_path)

  train_transforms = A.Compose([
        A.Resize(height=416, width=416),
        A.RandomSizedCrop(min_max_height=(350, 416), height=416, width=416, p=0.4),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(p=1.0)
    ],bbox_params={"format":"coco",'label_fields': ['labels']})

  if data_format == "yolo":
        train_data, class_id_list = read_data_yolo_format(train_label_path, val_label_path)
        train_dataset = CustomDataset(train_data, transforms=train_transforms,
                                     number_class=nc, img_dir=train_path)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            collate_fn=collate_fn, drop_last=True)
  
  elif data_format == "voc":
      if voc_path_txt == None:
          print("File not found")
          return
      else:
          train_data = read_data_voc_format(voc_path_txt)
          train_dataset = CustomDataset(train_data, transforms=train_transforms, 
                                        number_class=nc, img_dir=train_path)
          train_loader = DataLoader(
              train_dataset,
              batch_size=batch_size,
              shuffle=True,
              num_workers=workers,
              collate_fn=collate_fn, drop_last=True)

  device = select_device(device_number, batch_size)
  torch.autograd.set_detect_anomaly(True)
  model = YOLO(S=13, BOX=5, CLS=nc).to(device)
  if opt.resume:
    print("Resume model")
    checkpoint = torch.load(weight_ckpt)
    model = YOLO(S=13, BOX=5, CLS=checkpoint["number_class"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
  print(summary(model, (3, 416, 416)))
  ANCHOR_BOXS = [[1.08,1.19],
               [3.42,4.41],
               [6.63,11.38],
               [9.42,5.11],
               [16.62,10.52]]
  
  checkpoint_init = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "number_class": nc,
            "class_name": class_name,
    }
  torch.save(checkpoint_init, init)

  print(f"Starting training for {epochs} epochs")
  t0 = time.time()
  total_train_loss = []
  for epoch in range(epochs):
    pbar = tqdm(train_loader, leave=True)
    model.zero_grad()
    print(('\n' + '%10s' + '%12s' * 3) % ('Epoch', 'gpu_mem', 'train_loss', 'img_size'))
    epoch_train_loss = []
    for batch_idx, (imgs, targets) in enumerate(pbar):
        tensor_imgs, tensor_targets = torch.stack(imgs), torch.stack(targets)
        output = model(tensor_imgs.to(device))
        loss = custom_loss(output_tensor=output, target_tensor=tensor_targets.to(device), 
                            anchor_boxs=ANCHOR_BOXS, device= device)
        loss_value = loss.item()
        epoch_train_loss.append(loss_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        s = ('%10s' + '%12s' + '%12.4g' * 2) % (
                    '%g/%g' % (epoch, epochs - 1), mem, loss_value, 416)
        pbar.set_description(s)
    if epoch % 15 == 0:
      subdir = 'epoch0' + str(epoch) + '.pth'
      epoch_ckpt = wdir / subdir
      checkpoint_epoch = {
          "state_dict": model.state_dict(),
          "optimizer": optimizer.state_dict(),
          "number_class": nc,
          "class_name": class_name,
      }
      torch.save(checkpoint_epoch, epoch_ckpt)
    total_train_loss.append(sum(epoch_train_loss)/len(epoch_train_loss))
    #--------------------------------Add tensorboard scalar---------------------------------------
    tags = ['train/loss']  # train loss per epoch
    list_result = [sum(epoch_train_loss)/len(epoch_train_loss)]
    
    for i, tag in enumerate(tags):
        if tb_writer:
            tb_writer.add_scalar(tag, list_result[i], epoch)
    #--------------------------------Add tensorboard saclar---------------------------------------
  #--------------------------------Add tensorboard histogram---------------------------------------
  if tb_writer:
    tb_writer.add_histogram('classes', torch.tensor(class_id_list), 0)
  #--------------------------------Add tensorboard histogram---------------------------------------
  print('%g epochs completed in %.3f hours.\n' % (epochs, (time.time() - t0) / 3600))
  plot_loss(total_train_loss, save_dir)
  checkpoint_last = {
          "state_dict": model.state_dict(),
          "optimizer": optimizer.state_dict(),
          "number_class": nc,
          "class_name": class_name,
  }
  torch.save(checkpoint_last, last)
  print("All result save in:", save_dir)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/1class_flag.yaml', help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--weight_ckpt', type=str, default='', help='weight checkpoint to resume')
    parser.add_argument('--batch_size', type=int, default=32, help='total batch size for all GPUs')
    parser.add_argument('--data_format', type=str, default="yolo", help='data format')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=2, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    opt = parser.parse_args()
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
    print(opt)
    print(f"Start with 'tensorboard --logdir {opt.save_dir}, view at http://localhost:6006/")
    tb_writer = SummaryWriter(opt.save_dir)
    train(opt,tb_writer)