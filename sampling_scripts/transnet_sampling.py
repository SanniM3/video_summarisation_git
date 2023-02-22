from TransNetV2.transnetv2 import TransNetV2
from argparse import ArgumentParser
from tqdm import tqdm
import pathlib
import os
import cv2
import numpy as  np
import pandas as pd 


par = ArgumentParser()
par.add_argument("-model_dir", type=str, default="/Users/ilakyaprabhakar/Documents/Edin MSc/MLP/video_summarisation_git/sampling_scripts/TransNetV2/transnetv2-weights",
                    help="Path to directory of model weights")
par.add_argument("-data_dir", type=str, default="/Users/ilakyaprabhakar/Documents/Edin MSc/MLP/video_summarisation_git/data_subset/videos",
                    help="Path to directory containing videos")
args = par.parse_args()

# create directory to save frames
save_path = os.path.split(args.data_dir)[0]
frames_file = os.path.join(save_path, 'transnet_frames')
pathlib.Path(frames_file).mkdir(parents=True, exist_ok=True)

# load model
model = TransNetV2(model_dir=args.model_dir)

stats = {'tot_frames': [], 'num_scenes': []}

# inference for each video
for video in tqdm(os.listdir(args.data_dir)):
        
    video_name, extension = os.path.splitext(video)
    
    # filters out DS Store files when running locally
    if extension == ".mp4":
    
        # make folder for each video
        video_file = os.path.join(frames_file, video_name)
        pathlib.Path(video_file).mkdir(parents=True, exist_ok=True)
        
        # load vid
        vidcap = cv2.VideoCapture(os.path.join(args.data_dir, video))
        # find tot number of frames
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        stats['tot_frames'].append(total_frames)
        
        # predict transitions
        video_frames, single_frame_predictions, all_frame_predictions = \
            model.predict_video(os.path.join(args.data_dir, video))
        
        frame_numbers, num_scenes = model.predictions_to_N_frames(video_frames, single_frame_predictions)
        
        stats['num_scenes'].append(num_scenes)
        
        n = 0
        for frame_number in frame_numbers:
    
            # set frame position
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, image = vidcap.read()
            if success:
                cv2.imwrite(os.path.join(video_file, video_name+'_frame{}.jpg'.format(n)), image)
                n+=1
                
df = pd.DataFrame(stats)
df.to_csv(os.path.join(frames_file, 'stats.csv') , index=False)