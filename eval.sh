cd ~/ondemand/src

apptainer exec ../my_container.sif python3 train.py --epoch 30 --save-to ~/ondemand/checkpoints \
                 --train-data ~/ondemand/data/ocr_data/data/train_data.csv \
                 --val-data ~/ondemand/data/ocr_data/data/val_data.csv \
                 --data ~/ondemand/data/ocr_data/data \
                 --vocab ~/ondemand/data/ocr_data/data/char_dict.json \
                 --save-freq 5000 \
                 --lr 1e-3 \
		         --cpt ~/ondemand/checkpoints/cp_10_30070.pth \
                 --eval True \
                 --ngpus 2
