train-voc:
	python main.py --data VOC2007 --save_dir checkpoint/VOC2007 --display

evaluate-voc:
	python main.py --data VOC2007 --resume checkpoint/VOC2007/MLGCN/SGD_COCO_lr_001_lrp_01_bs16_5/checkpoint_best.pth --evaluate --display
