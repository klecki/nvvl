#!/bin/bash
python ./tools/transcode_scenes.py --master_data data --resolution 4K
python ./tools/transcode_scenes.py --master_data data --resolution 1080p
python ./tools/transcode_scenes.py --master_data data --resolution 720p
python ./tools/transcode_scenes.py --master_data data --resolution 540p

python ./tools/extract_frames.py --master_data data --resolution 540p
python ./tools/extract_frames.py --master_data data --resolution 720p
python ./tools/extract_frames.py --master_data data --resolution 1080p
python ./tools/extract_frames.py --master_data data --resolution 4K #not sure if ok
