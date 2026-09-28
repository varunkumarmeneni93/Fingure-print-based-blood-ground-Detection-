import os
import cv2
import numpy as np

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 32):
        params = {'ksize': (ksize, ksize), 'sigma': 1.0, 'theta': theta, 
                  'lambd': 15.0, 'gamma': 0.02, 'psi': 0, 'ktype': cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5 * kern.sum()
        filters.append((kern, params))
    return filters

def getGabor(img, filters):
    results = []
    for kern, params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        results.append(fimg)
    return np.asarray(results[31])

# Path to dataset
DATASET_PATH = "Dataset"
OUTPUT_DIR = "model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

filters = build_filters()
X, Y = [], []
labels = sorted(os.listdir(DATASET_PATH))

for label_index, label in enumerate(labels):
    label_folder = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_folder):
        continue

    for img_name in os.listdir(label_folder):
        img_path = os.path.join(label_folder, img_name)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = getGabor(img, filters)
            img = cv2.resize(img, (32, 32))
            X.append(img)
            Y.append(label_index)
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

X = np.asarray(X)
Y = np.asarray(Y)

np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "Y.npy"), Y)

print(f"✅ Dataset generation complete.")
print(f"✅ Total images: {len(X)} | Classes: {len(labels)}")
