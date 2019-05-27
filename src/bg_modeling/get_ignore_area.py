import skimage
from skimage.measure import label 
from scipy.ndimage.filters import gaussian_filter
import cv2

video_id = sys.argv[1]
count_thred = 0.08
min_area = 2000
gass_sigma = 3
for video_id in range(77,78):
    # print(video_id)
    mat = np.load("AICity/data_process/segs/%s.npy"%str(video_id))
    sns.heatmap(mat,cmap='jet')
    plt.show()
    mat = mat-np.min(mat)
    mat = mat/np.max(mat)
    mask= mat>count_thred
#     mask = mask.astype(float)
#     k = gaussian_filter(mask,4)
#     mask = k>0.08
    sns.heatmap(mask,cmap='jet')
    plt.show()
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
