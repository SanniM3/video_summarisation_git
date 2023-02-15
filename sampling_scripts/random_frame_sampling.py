import cv2
import argparse
import pathlib
import os
import random
from tqdm import tqdm

args = argparse.ArgumentParser()
args.add_argument("-data_path", type=str,
                  default='/Users/ilakyaprabhakar/Documents/Edin MSc/MLP/test_data/videos',
                  help="Path to directory containing videos")
args.add_argument("-save_path", type=str,
                  default='/Users/ilakyaprabhakar/Documents/Edin MSc/MLP/test_data/',
                  help="Path to save folder containing frame representation of videos")
args = args.parse_args()

frames_file = os.path.join(args.save_path, 'random_frames')
pathlib.Path(frames_file).mkdir(parents=True, exist_ok=True)

for video in tqdm(os.listdir(args.data_path)):
    video_name, extension = os.path.splitext(video)
    # make folder for each video
    
    video_file = os.path.join(frames_file, video_name)
    pathlib.Path(video_file).mkdir(parents=True, exist_ok=True)
    
    # filters out DS Store files when running locally
    if extension == ".mp4":
        # load vid
        vidcap = cv2.VideoCapture(os.path.join(args.data_path, video))
        # find tot number of frames
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        # sample 6 random frames
        random_frame_numbers = random.sample(range(0, total_frames), 6)
        
        n = 0
        for frame_number in random_frame_numbers:
    
            # set frame position
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, image = vidcap.read()
            
            if success:
                cv2.imwrite(os.path.join(video_file, video_name+'_frame{}.jpg'.format(n)), image)
                n+=1