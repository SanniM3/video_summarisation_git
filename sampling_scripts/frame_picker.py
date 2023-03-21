import random
import numpy as np

random.seed(42)
np.random.seed(42)

def _select_n_scenes(scenes, n):
    """given a list of scenes, randomly pick n scenes from it.
    
    If there are fewer than n scenes,
    first pick all scenes, and then fill the difference with randomly 
    selected scenes (with replacement)
    """
    if len(scenes) == n:
        return scenes
    elif len(scenes) > n:
        return random.sample(scenes, n)
    elif len(scenes) < n:
        diff = n - len(scenes)
        return scenes + random.choices(scenes, k=diff)

def _select_frames_from_scenes(scenes):
    # print(scenes)
    ret = []
    for start, end in scenes:
        #print(f"selecting between {start} and {end} (exclusive)")
        if start == end:
            frame = start # work around b/c randint hates start == end
        else:
            frame = np.random.randint(start, end)
        ret.append(frame)
    return sorted(ret)

def pick_n_frames(scenes, n):
    scene_subset = _select_n_scenes(scenes, n)
    return _select_frames_from_scenes(scene_subset)