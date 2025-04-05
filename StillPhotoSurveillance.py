import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import random

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()
model.to(device)

# COCO class names (subset relevant to surveillance)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Custom categories for people and objects
PERSON_CATEGORIES = {
    "thief/unwanted": ["knife", "backpack", "suitcase"],  # Suspicious items
    "worker/garbage": ["bottle", "broom", "truck"],       # Work-related (broom not in COCO, simulated)
    "friendly": ["cell phone", "book", "dog"]             # Casual/friendly items
}
WEAPON_CLASSES = ["knife", "baseball bat"]  # Expandable to "gun" with custom training
TOOL_CLASSES = ["scissors", "spoon", "bottle"]  # Work tools

def get_prediction(img_path, threshold):
    """Perform instance segmentation and return masks, boxes, and predicted classes."""
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])

    masks = prediction[0]['masks'].cpu().numpy()
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    pred_cls = [COCO_INSTANCE_CATEGORY_NAMES[labels[i]] for i in range(len(scores)) if scores[i] > threshold]
    masks = masks[scores > threshold]
    boxes = boxes[scores > threshold]
    scores = scores[scores > threshold]

    return masks, boxes, pred_cls, scores

def classify_person(objects_nearby):
    """Classify a person based on nearby detected objects."""
    for obj in objects_nearby:
        for category, items in PERSON_CATEGORIES.items():
            if obj in items:
                return category
    return "unknown"  # Default if no clear match

def classify_object(obj_name):
    """Distinguish between weapons and tools."""
    if obj_name in WEAPON_CLASSES:
        return "weapon"
    elif obj_name in TOOL_CLASSES:
        return "tool"
    return "other"

def analyze_surveillance_image(img_path, threshold=0.5, save_output=True):
    """Process surveillance image and classify people/objects."""
    masks, boxes, pred_cls, scores = get_prediction(img_path, threshold)

    # Analyze detected objects and people
    people = []
    objects = []
    for i, cls in enumerate(pred_cls):
        if cls == "person":
            people.append({"box": boxes[i], "mask": masks[i], "nearby_objects": []})
        else:
            objects.append({"class": cls, "box": boxes[i], "type": classify_object(cls)})

    # Associate objects with people based on proximity (simple overlap check)
    for person in people:
        p_box = person["box"]
        for obj in objects:
            o_box = obj["box"]
            if (o_box[0] < p_box[2] and o_box[2] > p_box[0] and  # X overlap
                o_box[1] < p_box[3] and o_box[3] > p_box[1]):    # Y overlap
                person["nearby_objects"].append(obj["class"])
        person["category"] = classify_person(person["nearby_objects"])

    # Visualize
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for person in people:
        mask = person["mask"][0] > 0.5
        box = person["box"]
        category = person["category"]
        color = {"thief/unwanted": (255, 0, 0), "worker/garbage": (0, 255, 0), 
                 "friendly": (0, 0, 255), "unknown": (128, 128, 128)}.get(category, (255, 255, 255))

        for c in range(3):
            img[:, :, c] = np.where(mask, img[:, :, c] * 0.5 + color[c] * 0.5, img[:, :, c])
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(img, category, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    for obj in objects:
        box = obj["box"]
        label = f"{obj['class']} ({obj['type']})"
        color = (255, 0, 255) if obj["type"] == "weapon" else (255, 255, 0)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(img, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Output results
    print(f"\nAnalysis for {os.path.basename(img_path)}:")
    print(f"People detected: {len(people)}")
    for i, p in enumerate(people):
        print(f"Person {i+1}: Category={p['category']}, Nearby Objects={p['nearby_objects']}")
    print(f"Objects detected: {len(objects)}")
    for o in objects:
        print(f"Object: {o['class']} ({o['type']})")

    if save_output:
        output_path = os.path.join(os.path.dirname(img_path), f"surveillance_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved output to: {output_path}")
    else:
        cv2.imshow("Surveillance Analysis", img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

def main():
    image_dir = "/path/to/your/surveillance/images"  # Update this path
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} does not exist. Update the path.")
        return

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images: {image_files}")

    for file_name in image_files:
        img_path = os.path.join(image_dir, file_name)
        print(f"\nProcessing: {img_path}")
        try:
            analyze_surveillance_image(img_path, threshold=0.75, save_output=True)
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

if __name__ == "__main__":
    main()
