import torch

def post_process_output(output, anchor_boxs, device):
    """Convert output of model to pred_xywh"""
    # xy
    xy = torch.sigmoid(output[:,:,:,:,:2]+1e-6)

    # wh
    wh = output[:,:,:,:,2:4]
    anchors_wh = torch.Tensor(anchor_boxs).view(1,1,1,5,2).to(device)
    wh = torch.exp(wh)*anchors_wh
    
    # objectness confidence
    obj_prob = torch.sigmoid(output[:,:,:,:,4:5]+1e-6)
    
    # class distribution
    cls_dist = torch.softmax(output[:,:,:,:,5:], dim=-1)
    return xy, wh, obj_prob, cls_dist

def post_process_target(target_tensor):
    """
    Tách target tensor thành từng thành phần riêng biệt: xy, wh, object_probility, class_distribution
    """
    xy = target_tensor[:,:,:,:,:2]
    wh = target_tensor[:,:,:,:,2:4]
    obj_prob = target_tensor[:,:,:,:,4:5]
    cls_dist = target_tensor[:,:,:,:,5:]
    return xy, wh, obj_prob, cls_dist

def square_error(output, target):
    return (output-target)**2

def custom_loss(output_tensor, target_tensor, anchor_boxs, device,
               S=13, BOX=5, img_size=416):
    """
    Luồng xử lí:
        1. Tính diện tích các pred_bbox
        2. Tính diện tích các true_bbox
        3. Tính iou giữa từng pred_bbox với true_bbox tương ứng (nằm trong cùng 1 cell)
        4. Trong mỗi cell, xác định best_box - box có iou với true_bbox đạt giá trị max so với 4 pred_bbox còn lại
        5. Tính các loss thành phần theo công thức trong ảnh
        6. Tính Total_loss
    """
    cell_size = (img_size/S, img_size/S)
    NOOB_W, CONF_W, XY_W, PROB_W, WH_W = 2.0, 10.0, 0.5, 1.0, 0.1

    pred_xy, pred_wh, pred_obj_prob, pred_cls_dist = post_process_output(output_tensor, anchor_boxs,
                                                                        device)
    true_xy, true_wh, true_obj_prob, true_cls_dist = post_process_target(target_tensor)
    
    # tính diện tích các pred_bbox
    pred_ul = pred_xy - 0.5*pred_wh
    pred_br = pred_xy + 0.5*pred_wh
    pred_area = pred_wh[:,:,:,:,0]*pred_wh[:,:,:,:,1]
    
    # Tính diện tích các true_bbox
    true_ul = true_xy - 0.5*true_wh
    true_br = true_xy + 0.5*true_wh
    true_area = true_wh[:,:,:,:,0]*true_wh[:,:,:,:,1]

    # Tính iou giữa từng pred_bbox với true_bbox tương ứng (nằm trong cùng 1 cell)
    intersect_ul = torch.max(pred_ul, true_ul)
    intersect_br = torch.min(pred_br, true_br)
    intersect_wh = intersect_br - intersect_ul
    intersect_area = intersect_wh[:,:,:,:,0]*intersect_wh[:,:,:,:,1]
    
    # Trong mỗi cell, xác định best_box - box có iou với true_bbox đạt giá trị max so với 4 pred_bbox còn lại
    iou = intersect_area/(pred_area + true_area - intersect_area)
    max_iou = torch.max(iou, dim=3, keepdim=True)[0]
    best_box_index =  torch.unsqueeze(torch.eq(iou, max_iou).float(), dim=-1)
    true_box_conf = best_box_index*true_obj_prob
    
    # Tính các loss thành phần theo công thức trong ảnh
    xy_loss =  (square_error(pred_xy, true_xy)*true_box_conf*XY_W).sum()
    wh_loss =  (square_error(pred_wh, true_wh)*true_box_conf*WH_W).sum()
    obj_loss = (square_error(pred_obj_prob, true_obj_prob)*(CONF_W*true_box_conf + NOOB_W*(1-true_box_conf))).sum()
    cls_loss = (square_error(pred_cls_dist, true_cls_dist)*true_box_conf*PROB_W).sum()

    # Loss kết hợp
    total_loss = xy_loss + wh_loss + obj_loss + cls_loss
    return total_loss