import numpy as np
d = [[-1,0],[1,0],[0,1],[0,-1]]
def search_region(G,pos):
    x1,y1,x2,y2 = pos[1],pos[0],pos[1],pos[0]
    Q = set()
    Q.add(pos)
    h,w = G.shape
    visited = np.zeros((h,w))
    visited[pos] = 1
    while Q:
        u = Q.pop()
        for move in d :
            row = u[0]+move[0]
            col = u[1]+move[1]
            if(row>=0 and row<h and col>=0 and col<w and G[row,col] == 1 and visited[row,col]==0):
                visited[row,col]=1
                Q.add((row,col))
                x1 = min(x1,col)
                x2 = max(x2,col)
                y1 = min(y1,row)
                y2 = max(y2,row)
    return [int(x1),int(y1),int(x2),int(y2)]

# a=np.array([[0,1,1,1,0,0,0,1,1],[0,1,1,0,0,0,0,1,0],[0,1,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,1,1]])
# search_region(a,(0,2))

def compute_iou(region,dets):
    if dets.shape[0] == 0:
        return 0

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order = scores.argsort()[::-1]
    # scores = scores[order]

    xx1 = np.maximum(region[0], x1)
    yy1 = np.maximum(region[1], y1)
    xx2 = np.minimum(region[2], x2)
    yy2 = np.minimum(region[3], y2)

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (areas + (region[2] - region[0] + 1) * (region[3] - region[1] + 1) - inter)
    max_iou = np.max(ovr)
    return max_iou

def compute_brightness(im):
    brightness = np.mean(0.299*im[:,:,2] + 0.587*im[:,:,1] + 0.114*im[:,:,0])/255.0
    return brightness

def anomely_nms(all_results,iou_thred = 0.8):
    anomalies = []
    for anomely in all_results:
        info = anomely['region']
        info.append(anomely['score'])
        info.append(anomely['start_time'])
        info.append(anomely['end_time'])
        anomalies.append(info)
    dets = np.array(anomalies)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    start_time = dets[:, 5]
    end_time = dets[:, 6]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)   
    order = scores.argsort()[::-1]  
    keep = []  
    while order.size > 0:   
        i = order[0]  
        keep.append(i)  
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        ovr = inter / (areas[i] + areas[order[1:]] - inter)    
        inds = np.where(ovr > iou_thred)[0] 
        tmp_order = order[inds + 1]
        if len(tmp_order)>0:
            dets[i,5] = np.min(start_time[tmp_order])
            dets[i,6] = np.max(end_time[tmp_order])
        inds = np.where(ovr <= iou_thred)[0]  
        order = order[inds + 1] 
    dets = dets[keep,:]
    return dets