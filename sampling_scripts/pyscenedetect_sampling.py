from scenedetect import detect, AdaptiveDetector, ContentDetector
import argparse
import pathlib
import os
from tqdm import tqdm
import json
import cv2
import numpy as  np
import pandas as pd

args = argparse.ArgumentParser()
args.add_argument("-data_dir", type=str,
                  default='/Users/ilakyaprabhakar/Documents/Edin MSc/MLP/video_summarisation_git/data_subset/videos',
                  help="Path to directory containing videos")
args.add_argument("-data_json", type=str, default="/Users/ilakyaprabhakar/Documents/Edin MSc/MLP/video_summarisation_git/data_subset/video_annotation.json",
                    help="Path to json containing video annotations")
args = args.parse_args()

# create directory to save frames
save_path = pathlib.Path(args.data_dir).parent
frames_file = os.path.join(save_path, 'pyscenedetect_frames')
pathlib.Path(frames_file).mkdir(parents=True, exist_ok=True)

# open annotation json and get video metadata
with open(args.data_json, 'r') as file:
    json_data = json.load(file)
video_meta_data = pd.DataFrame(json_data['videos'])

stats = {'video_id': [], 'category': [],
         'tot_frames': [], 'num_scenes': []}

# scene detection for each video
for video in tqdm(os.listdir(args.data_dir)):
        
    video_name, extension = os.path.splitext(video)
    
    # filters out DS Store files when running locally
    if extension == ".mp4":
        
        # fill in stats df
        stats['category'].append(
            video_meta_data.loc[video_meta_data['video_id'] == video_name]['category'].iloc[0])
        stats['video_id'].append(video_name)
    
        try:
    
            # make folder for each video
            video_file = os.path.join(frames_file, video_name)
            pathlib.Path(video_file).mkdir(parents=True, exist_ok=True)
            
            # load vid
            vidcap = cv2.VideoCapture(os.path.join(args.data_dir, video))
            # find tot number of frames
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            stats['tot_frames'].append(total_frames)
            
            # predict transitions
            scenes = detect(os.path.join(args.data_dir, video), AdaptiveDetector())
            if len(scenes) <= 1:
                scenes = detect(os.path.join(args.data_dir, video), ContentDetector())
            
            if len(scenes) <= 1:
                print('File {} skipped as no scenes found'.format(video_name))
                pass
            
            scene_list = []

            for i, scene in enumerate(scenes):
                scene_list.append([scene[0].get_frames()+1, scene[1].get_frames()])

            stats['num_scenes'].append(len(scene_list))
            
            frame_numbers = np.zeros(6)
            
            # sample 1 frame randomly from each scene until 6 frames are found
            n = 0
            while n < 6:
                for s in scenes:
                    frame_numbers[n] = np.random.randint(s[0], s[1]+1)
                    n+=1 
                    if n == 6:
                        break

            # save frames
            n = 0
            for frame_number in frame_numbers:
        
                # set frame position
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, image = vidcap.read()
                if success:
                    cv2.imwrite(os.path.join(video_file, video_name+'_frame{}.jpg'.format(n)), image)
                    n+=1
        except:
            print('File {} skipped'.format(video_name))
            pass
                
df = pd.DataFrame(stats)
df.to_csv(os.path.join(frames_file, 'stats.csv') , index=False)