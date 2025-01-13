from models import get_model
from argparse import Namespace
import torch
import cv2
import numpy as np

coco_labels = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

VOC_LABELS = [
    'background',  # class 0
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]



def resize_to_fixed_size(img, target_size=448):
    # Resize the image to target_size x target_size without preserving aspect ratio
    resized_img = cv2.resize(img, (target_size, target_size))
    return resized_img

def center_crop_to_size(img, target_size=448):
    h, w, _ = img.shape

    # Ensure the image is larger than or equal to the target size
    if h < target_size or w < target_size:
        raise ValueError(f"Image dimensions ({h}, {w}) are smaller than the target crop size {target_size}.")

    # Calculate the crop region
    top = (h - target_size) // 2
    left = (w - target_size) // 2
    bottom = top + target_size
    right = left + target_size

    # Perform center crop
    cropped_img = img[top:bottom, left:right]

    return cropped_img


def resize_with_padding(img, target_size=448):
    h, w, _ = img.shape
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image while keeping aspect ratio
    resized_img = cv2.resize(img, (new_w, new_h))
    
    # Calculate padding
    top_pad = (target_size - new_h) // 2
    bottom_pad = target_size - new_h - top_pad
    left_pad = (target_size - new_w) // 2
    right_pad = target_size - new_w - left_pad

    # Add padding
    padded_img = cv2.copyMakeBorder(
        resized_img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    return padded_img

def numpy_to_tensor(img_np):
    # Convert numpy array to tensor
    img_tensor = torch.from_numpy(img_np)  # Shape: (H, W, C)
    img_tensor = img_tensor.permute(2, 0, 1)  # Shape: (C, H, W)
    img_tensor = img_tensor.unsqueeze(0)  # Shape: (1, C, H, W) for batch size of 1
    img_tensor = img_tensor.float()
    return img_tensor

model_path = "checkpoint/VOC2007/MLGCN/SGD_COCO_lr_001_lrp_01_bs16_5/checkpoint_best.pth"

model_architecture = "MLGCN"
embeddings_path = "wordfeature"
num_classes = 20
args = Namespace(model_name=model_architecture, word_feature_path=embeddings_path)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = get_model(num_classes, args)
checkpoint = torch.load(model_path, map_location=device)

model_dict = model.state_dict()

for k, v in checkpoint['state_dict'].items():
    if k in model_dict and v.shape == model_dict[k].shape:
        model_dict[k] = v
    else:
        print('\tMismatched layers: {}'.format(k))
model.load_state_dict(model_dict)
del checkpoint
del model_dict

model.to(device)

model.eval()


# ImageNet mean and standard deviation values
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

img_path = "examples/Pierre-Person.jpg"
img = cv2.imread(img_path)


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


img_resized_padded = resize_to_fixed_size(img_rgb, target_size=448)

# Normalize pixel values to [0, 1]
img_rgb = img_resized_padded.astype(np.float32) / 255.0

# Apply ImageNet normalization
img_normalized = (img_rgb - imagenet_mean) / imagenet_std

img_tensor = numpy_to_tensor(img_normalized)

img_tensor = img_tensor.to(device)

outputs = model(img_tensor)

# Apply the sigmoid function to the output
sigmoid_output = torch.sigmoid(outputs)

positions = torch.where(sigmoid_output > 0.5)[1]  # Get class indices

# Get the labels corresponding to the indices
predicted_labels = [VOC_LABELS[i+1] for i in positions.tolist()]

print(f"Predicted labels: {predicted_labels}")


print('hola')