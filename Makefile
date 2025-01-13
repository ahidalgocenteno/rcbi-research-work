train-voc:
	python main.py --data VOC2007 --save_dir checkpoint/VOC2007 --display

train-voc-2012:
	python main.py --data VOC2012 --save_dir checkpoint/VOC2012/Normal --model_name MLGCN --gpu 1 --display

evaluate-voc:
	python main.py --data VOC2007 --resume checkpoint/VOC2007/MLGCN/SGD_COCO_lr_001_lrp_01_bs16_5/checkpoint_best.pth --evaluate --display

train-coco:
	python main.py --data COCO2014 --save_dir checkpoint/COCO2014 --display

train-improvement-voc:
	python main.py --data VOC2007 --model_name MLGCNImproved --save_dir checkpoint/VOC2007/MLGCNImproved --display

train-effi:
	python main.py --data VOC2007 --model_name MLGCNEfficientNet --batch-size 8 --save_dir checkpoint/VOC2007/MLGCNEfficientNet --display

train-effi-coco:
	python main.py --data COCO2014 --model_name MLGCNEfficientNet --batch-size 8 --save_dir checkpoint/COCO2014/MLGCNEfficientNet --display

train-mobilenetv3:
	python main.py --data VOC2007 --model_name MLGCNRealTime --batch-size 16 --save_dir checkpoint/VOC2007/MLGCNRealTime --display

evaluate-efficientnet-coco:
	python main.py --data COCO2014 --model_name MLGCNEfficientNet --resume checkpoint/COCO2014/MLGCNEfficientNet/segundo_entreno/checkpoint_best.pth --evaluate --display

