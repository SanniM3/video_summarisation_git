source .venv/bin/activate
python sampling_scripts/random_frame_sampling.py -data_dir data_subset/videos
python command_builder/build_training_command.py data_subset/random_frames/ data_subset/train_val_videodatainfo.json > train.sh