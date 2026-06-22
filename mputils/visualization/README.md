
# Fingerspelling Video Visualization!!

### Basic Steps

- First run `parquet_by_participant.py` to output a parquet file (in `/data`) with just a single participant's data
- Next run `categorize_seqs.py` to split the participant data into the correct and incorrect video (pass the HResults file and a path to the original data folder for this).
- Now run `visualize.py` to visualize participant data. It will output videos in `videos/`
- Finally run `categorize_videos.py` to categorize the videos into correct/incorrect/remaining. (There will usually be some videos in remaining since `visualize` will look at all pt vids (not just the train split)

