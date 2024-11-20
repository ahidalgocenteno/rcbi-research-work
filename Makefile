


train:
	python demo_voc2007_gcn.py --data data -j 4

evaluate:
	python demo_voc2007_gcn.py --data data --evaluate --resume checkpoint/voc2007/model_best_14.8557.pth.tar