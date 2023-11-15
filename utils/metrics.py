def overlap(interval_1, interval_2):
    x1, x2 = interval_1
    x3, x4 = interval_2
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

def compute_iou(box1, box2):
    """Compute IOU between box1 and box2"""
    x1,y1,w1,h1 = box1[0], box1[1], box1[2], box1[3]
    x2,y2,w2,h2 = box2[0], box2[1], box2[2], box2[3]
    
    ## if box2 is inside box1
    if (x1 < x2) and (y1<y2) and (w1>w2) and (h1>h2):
        return 1
    
    area1, area2 = w1*h1, w2*h2
    intersect_w = overlap((x1,x1+w1), (x2,x2+w2))
    intersect_h = overlap((y1,y1+h1), (y2,y2+w2))
    intersect_area = intersect_w*intersect_h
    iou = intersect_area/(area1 + area2 - intersect_area)
    return iou

def nonmax_suppression(boxes, IOU_THRESH = 0.4):
    """remove ovelap bboxes"""
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    for i, current_box in enumerate(boxes):
        if current_box[4] <= 0:
            continue
        for j in range(i+1, len(boxes)):
            iou = compute_iou(current_box, boxes[j])
            if iou > IOU_THRESH:
                boxes[j][4] = 0
    boxes = [box for box in boxes if box[4] > 0]
    return boxes