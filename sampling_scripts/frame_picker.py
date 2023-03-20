import random
import numpy as np

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
    return sorted([np.random.randint(start, end+1) for start, end in scenes])

def pick_n_frames(scenes, n):
    scene_subset = _select_n_scenes(scenes, n)
    return _select_frames_from_scenes(scene_subset)