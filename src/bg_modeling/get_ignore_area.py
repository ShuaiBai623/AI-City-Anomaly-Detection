import skimage
from skimage.measure import label 
from scipy.ndimage.filters import gaussian_filter
import cv2

video_id = sys.argv[1]
count_thred = 0.08
min_area = 2000
gass_sigma = 3
score_thred = 0.1
with open("detection_results/test_framebyframe/video%s.txt"%(video_id),'r') as f:
for line in f:
    line = line.rstrip()
    word = line.split(',')
    frame = int(word[0])
    x1 = int(word[2])
    y1 = int(word[3])
    tmp_w = int(word[4])
    tmp_h = int(word[5])
    score = float(word[6])
    if frame not in dt_results_fbf:
       dt_results_fbf[frame]=[]
    if score > score_thred :
       dt_results_fbf[frame].append([x1,y1,x1+tmp_w,y1+tmp_h,

im = cv2.imread("data/AIC_Track3/ori_images/%s/1.jpg"%video_id)
h,w,c = im.shape
mat = np.zeros((h,w))
for frame in dt_results_fbf:
if frame <18000:
   tmp_score = np.zeros((h,w))
   for box in dt_results_fbf[frame]:
       score = box[4]
       tmp_score[int(float(box[1])):int(float(box[3])),int(float(box[0])):int(float(box[2]))] = np.maximum(score,tmp_score[int(float(box[1])):int(float(box[3])),int(float(box[0])):int(float(box[2]))])

   mat = mat+tmp_score
mat = mat-np.min(mat)
mat = mat/np.max(mat)
mask= mat>count_thred
mask = label(mask, connectivity = 1)
num = np.max(mask)
for i in range(1,int(num+1)):
    if np.sum(mask==i)<min_area:
        mask[mask==i]=0     
mask = mask>0
mask = mask.astype(float)
k = gaussian_filter(mask,gass_sigma)
mask = k>count_thred
np.save("detection_results/seg_masks/%s.npy"%str(video_id),mask)
