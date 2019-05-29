import cv2
import os
import sys
import glob
import tqdm
# video_name = sys.argv[1]
root = "data/AIC_Track3/videos/"
dest_dir = "data/AIC_Track3/ori_images/"
video_names = [str(i)+'.mp4' for i in range(1,101)]
print("caputure videos")
for video_name in tqdm.tqdm(video_names):
    file_name = video_name
    folder_name = dest_dir+file_name.split('.')[0]
    os.makedirs(folder_name,exist_ok=True)
    vc = cv2.VideoCapture(root+video_name)
    c = 1
    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False

    timeF =1   # 视频帧计数间隔频率

    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        pic_path = folder_name+'/'
        if (c % timeF == 0):  # 每隔timeF帧进行存储操作
            # print(pic_path + str(c) + '.jpg')
            cv2.imwrite(pic_path + str(c) + '.jpg', frame)  # 存储为图像,保存名为 >文件夹名_数字（第几个文件）.jpg
        c = c + 1
        cv2.waitKey(1)
    vc.release()

dest_dir_processed = "data/AIC_Track3/processed_images/"
print("average images")
for i in tqdm.tqdm(range(1,101)):
    video_name = str(i)
    path_file_number=glob.glob(os.path.join(dest_dir,video_name,'*.jpg')) #获取当前文件夹下个数
    internal_frame = 4
    start_frame = 1
    video_name = str(i)
    nums_frames = len(path_file_number)
    alpha=0.1
    os.mkdir(dest_dir_processed+video_name,exist_ok=True)

    for j in range(4,5):
        internal_frame = 4+j*4
        num_pic = int(nums_frames/internal_frame)
        former_im = cv2.imread(dest_dir_processed+"%d/1.jpg"%i)
        img = cv2.imread(os.path.join(root,video_name,str(start_frame)+'.jpg'))
        for i in range(num_pic):
            now_im = cv2.imread(os.path.join(root,video_name,str(i*internal_frame+start_frame)+'.jpg'))
            if np.mean(np.abs(now_im-former_im))>5:
                img = img*(1-alpha)+now_im*alpha
                cv2.imwrite(dest_dir_processed+video_name+'/'+str(i*internal_frame+start_frame)
                        +'_'+str(j)+'.jpg',img)
            else:
                cv2.imwrite(dest_dir_processed+video_name+'/'+str(i*internal_frame+start_frame)
                        +'_'+str(j)+'.jpg',img*0)
            former_im = now_im
