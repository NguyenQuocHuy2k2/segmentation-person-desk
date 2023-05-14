from ultralytics import YOLO

model = YOLO("best.pt")
results = model("53.jpg", show=True, hide_labels=True)
x =0
y = 0
for result in results:
    boxes = result.boxes.numpy()
    for box in boxes:
        if box.cls == 1:
            x += 1
        else:
            y += 1
print("persons:", x)
print("desks:", y)

