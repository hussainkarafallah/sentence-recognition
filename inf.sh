cd ~/ondemand/src

apptainer exec ../my_container.sif python3 inference.py \
               --cpt ~/ondemand/checkpoints/cp_10_30070.pth \
               --val-data ~/ondemand/data/ocr_data/data/val_data.csv \
               --data ~/ondemand/data/ocr_data/data \
               --vocab ~/ondemand/data/ocr_data/data/char_dict.json \
               --ngpus 2

