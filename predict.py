from ultralytics import YOLO

# Load your newly trained best weights
model = YOLO("runs/detect/train/weights/best.pt")

# Run prediction on an image, video, or your webcam
# (Set source=0 to use your Mac's webcam live!)
results = model.predict(source="path_to_your_image.jpg", show=True, conf=0.5)