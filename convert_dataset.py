import os
import shutil
import xml.etree.ElementTree as ET

# CHANGE THESE PATHS
images_path = "Datacluster Fire and Smoke Sample/Datacluster Fire and Smoke Sample"
annotations_path = "Annotations/Annotations"
output_path = "fire_dataset"

fire_path = os.path.join(output_path, "fire")
nofire_path = os.path.join(output_path, "nofire")

os.makedirs(fire_path, exist_ok=True)
os.makedirs(nofire_path, exist_ok=True)

image_files = os.listdir(images_path)

for img_file in image_files:
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    xml_file = os.path.splitext(img_file)[0] + ".xml"
    xml_path = os.path.join(annotations_path, xml_file)
    img_path = os.path.join(images_path, img_file)

    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        fire_found = False
        for obj in root.findall("object"):
            name = obj.find("name").text.lower()
            if name == "fire":
                fire_found = True
                break

        if fire_found:
            shutil.copy(img_path, os.path.join(fire_path, img_file))
        else:
            shutil.copy(img_path, os.path.join(nofire_path, img_file))
    else:
        shutil.copy(img_path, os.path.join(nofire_path, img_file))

print("Dataset conversion complete!")