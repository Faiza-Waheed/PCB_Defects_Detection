import os, re
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import h5py
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
#from skimage.util import random_noise

#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

working_path = os.path.dirname(os.path.abspath(__file__)) + '/'
main_path = os.path.join(working_path, os.pardir, os.pardir)

home_path = os.path.expanduser('~')
downloads_path = os.path.join(home_path, 'Downloads')

# Set up folder paths
# Define the source paths for images and annotations
image_pool_path = os.path.join(main_path, os.pardir, 'VOC_PCB', 'JPEGImages')
annot_pool_path = os.path.join(main_path, os.pardir, 'VOC_PCB', 'Annotations')

# Define the destination path for images and annotations
image_dest_path = os.path.join(main_path, 'data', 'Images')
annot_dest_path = os.path.join(main_path, 'data', 'Annotations')

# Define the destination path for bboxes and masks
bb_path = os.path.join(main_path, 'data', 'Images_bb')
mask_path = os.path.join(main_path, 'data', 'Pixel_masks')

# Define the destination path for csv file
csv_path = os.path.join(main_path, 'data', 'csv')

image_dataset_path = image_dest_path


def plot_images(images, masks, labels, num_samples=6):
    _ , axes = plt.subplots(num_samples//5+max(0,min(1,num_samples%5)), 5, figsize=(10, 6))
    axes = axes.ravel()
    sample_indices = np.random.choice(len(images), num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        axes[i].imshow(images[idx], cmap='gray')
        axes[i].imshow(masks[idx], alpha=0.5, cmap='jet')
        axes[i].set_title(f'Label: {labels[idx]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def count_labels(y_labels):
    unique_labels, label_counts = np.unique(y_labels, return_counts=True)
    for label, count in zip(unique_labels, label_counts):
        print(f"Label: {label}, Count: {count}")
    print("\n")    
    return label_counts


def init_dataset():
    # read and group csv file
    csv_bounding_boxes_path = os.path.join(csv_path, 'PCB_annotations_dataset.csv')
    bounding_boxes_data = pd.read_csv(csv_bounding_boxes_path, sep=";")
    grouped_bbox = bounding_boxes_data.groupby('filename')

    images = os.listdir(image_dataset_path)

    image_data = []
    mask_data = []
    class_labels = []

    for filename in images:
    
        image_path = os.path.join(image_dataset_path, filename)
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (600, 600))  
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_array = np.asarray(gray_image) / 255.0 
        image_data.append(image_array)

        filename = os.path.splitext(filename)[0]
            
        mask = np.zeros((600, 600), dtype=np.uint8)

        if filename in grouped_bbox.groups:
            image_bbox_df = grouped_bbox.get_group(filename)

            for index, row in image_bbox_df.iterrows():
                class_label = row['defect']
            
                xmin, ymin, xmax, ymax = [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])] 
                #mask_color = class_to_color[class_label]
                mask[ymin:ymax, xmin:xmax] = 1 #mask_color 
        
            
        mask_data.append(mask)
        
        class_labels.append(class_label)

    return np.array(image_data), np.array(mask_data), np.array(class_labels)


def crop_images(X_images, y_bounding_boxes, y_class_labels, new_size, none_threshold=5):
    if not all(new < orig and orig % new == 0 for orig, new in zip((600, 600), new_size)):
        print("New size must be smaller and a divisor of the original size.")
        return False
    
    cropped_images = []
    cropped_masks = []
    cropped_y_class_labels = []

    for image, mask, labels in zip(X_images, y_bounding_boxes, y_class_labels):
        for y in range(0, 600, new_size[0]):
            for x in range(0, 600, new_size[0]):
                y0, x0 = y, x
                y_new_size, x_new_size = y+new_size[0], x+new_size[0]
                # if a bounding box crosses the cropping border, shift the corresponding border, so defects are not cut off
                while( np.any(mask[y0, x0:x_new_size])) and (y0 != 0): 
                    y0 -= 1
                while (np.any(mask[min(y_new_size, 599), x0:x_new_size])) and (y_new_size != 600):
                    y_new_size += 1
                while( np.any(mask[y0:y_new_size, x0])) and (x0 != 0):
                    x0 -= 1
                while( np.any(mask[y0:y_new_size, min(x_new_size, 599)])) and (x_new_size != 600):
                    x_new_size += 1
                patch_image = cv2.resize(image[y0:y_new_size, x0:x_new_size], new_size)
                patch_mask = cv2.resize(mask[y0:y_new_size, x0:x_new_size], new_size)
                
                cropped_images.append(patch_image)
                cropped_masks.append(patch_mask)
                if np.sum(patch_mask) > none_threshold:
                    cropped_y_class_labels.append(labels)
                else:
                    cropped_y_class_labels.append("none")

    return np.array(cropped_images), np.array(cropped_masks), np.array(cropped_y_class_labels)


def separate_defects(original_images, mask_images, class_labels):
    separated_images = []
    separated_masks = []
    separated_labels = []

    for idx, (original_image, mask_image, class_label) in enumerate(zip(original_images, mask_images, class_labels)):
        contours, _ = cv2.findContours(mask_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            for _, contour in enumerate(contours):
                all_contours_mask = np.zeros_like(mask_image)
                cv2.drawContours(all_contours_mask, [contour], -1, 255, -1)

                exclude_other_contours_mask = np.zeros_like(mask_image)
                cv2.drawContours(exclude_other_contours_mask, contours, -1, 255, -1)
                exclude_other_contours_mask = cv2.bitwise_xor(all_contours_mask, exclude_other_contours_mask)

                new_image = original_image.copy()
                new_mask = mask_image.copy()

                new_image[exclude_other_contours_mask == 255] = 0
                new_mask[exclude_other_contours_mask == 255] = 0

                separated_images.append(new_image)
                separated_masks.append(new_mask)
                separated_labels.append(class_label)
        else:
            separated_images.append(original_image.copy())
            separated_masks.append(mask_image.copy())
            separated_labels.append(class_label)
    return np.asarray(separated_images), np.asarray(separated_masks), np.asarray(separated_labels)


def balance_dataset(images, masks, labels):
    # Filter out extra samples for each class label
    filtered_images = []
    filtered_masks = []
    filtered_labels = []

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    max_samples_per_class = min(label_counts)

    label_count_dict = defaultdict(int)

    for label, count in zip(unique_labels, label_counts):
        label_count_dict[label] = count

    for label in unique_labels:
        label_indices = [i for i, l in enumerate(labels) if l == label]
        if label_count_dict[label] > max_samples_per_class:
            selected_indices = np.random.choice(label_indices, max_samples_per_class, replace=False)
        else:
            selected_indices = label_indices
        filtered_images.extend(images[selected_indices])
        filtered_masks.extend(masks[selected_indices])
        filtered_labels.extend(labels[selected_indices])

    # Convert the filtered lists to arrays
    return np.array(filtered_images), np.array(filtered_masks), np.array(filtered_labels)


def clean_none_labels(masks, labels):
    for i, label in enumerate(labels):
        # Check if the label is "none"
        if label == "none":
            # Replace the corresponding mask image with a blank image
            masks[i] = np.zeros_like(masks[i])
    return masks



def one_hot_encode_labels(labels):
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    unique_labels = encoder.classes_
    labels_categorical = to_categorical(labels_encoded)
    return labels_categorical, unique_labels

def decode_labels(labels_cat, label_list):
    indices = list(map(lambda cat: np.argmax(cat), labels_cat))
    return np.asarray([label_list[i] for i in indices])

def augment_data(images, masks, labels):
    augmented_images = []
    augmented_masks = []
    augmented_labels = []
    none_label = np.array([0, 0, 1, 0, 0, 0, 0])
    threshold = 20

    for image, mask, labels in zip(images, masks, labels):
        augmented_images.append(image)
        augmented_masks.append(mask)
        augmented_labels.append(labels)

        # shear
        shear_image, shear_mask = shear(image, mask)        
        augmented_images.append(shear_image)
        augmented_masks.append(shear_mask)
        if np.sum(shear_mask) > threshold:
            augmented_labels.append(labels)
        else:
            augmented_labels.append(none_label)

        # flip
        flip_image, flip_mask = flip(image, mask)
        augmented_images.append(flip_image)
        augmented_masks.append(flip_mask)
        if np.sum(flip_mask) > threshold:
            augmented_labels.append(labels)
        else:
            augmented_labels.append(none_label)

        # zoom
        zoom_image, zoom_mask = zoom(image, mask)
        augmented_images.append(zoom_image)
        augmented_masks.append(zoom_mask)
        if np.sum(zoom_mask) > threshold:
            augmented_labels.append(labels)
        else:
            augmented_labels.append(none_label)

        # width shift
        width_shift_image, width_shift_mask = width_shift(image, mask)
        augmented_images.append(width_shift_image)
        augmented_masks.append(width_shift_mask)
        if np.sum(width_shift_mask) > threshold:
            augmented_labels.append(labels)
        else:
            augmented_labels.append(none_label)

        # height shift
        height_shift_image, height_shift_mask = height_shift(image, mask)
        augmented_images.append(height_shift_image)
        augmented_masks.append(height_shift_mask)
        if np.sum(height_shift_mask) > threshold:
            augmented_labels.append(labels)
        else:
            augmented_labels.append(none_label)
        
        # rotate
        rotate_image, rotate_mask = rotate(image, mask)
        augmented_images.append(rotate_image)
        augmented_masks.append(rotate_mask)
        if np.sum(rotate_mask) > threshold:
            augmented_labels.append(labels)
        else:
            augmented_labels.append(none_label)
        
        # noise
        #noise_image = gaussian_noise(image)
        #augmented_images.append(noise_image)
        #augmented_masks.append(mask) 
        #augmented_labels.append(labels)

    augmented_images = np.array(augmented_images)
    augmented_masks = np.array(augmented_masks)
    augmented_labels = np.array(augmented_labels)

    return augmented_images, augmented_masks, augmented_labels

def shear(image, mask):
    shear_factor = np.random.uniform(-0.2, 0.2)
    rows, cols = image.shape[:2]
    shear_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    shear_image = cv2.warpAffine(image, shear_matrix, (cols, rows))
    shear_mask = cv2.warpAffine(mask, shear_matrix, (cols, rows))

    padded_image = np.zeros((new_size[0], new_size[0]), dtype=np.uint8)
    padded_mask = np.zeros((new_size[0], new_size[0]), dtype=np.uint8)
    padded_image[:rows, :cols] = shear_image
    padded_mask[:rows, :cols] = mask

    return shear_image, shear_mask

def flip(image, mask):
    flip_image = cv2.flip(image, 1)
    flip_mask = cv2.flip(mask, 1)
    return flip_image, flip_mask

def zoom(image, mask):
    zoom_factor = np.random.uniform(0.8, 1.2)
    zoomed_image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor)
    zoomed_mask = cv2.resize(mask, None, fx=zoom_factor, fy=zoom_factor)

    rows, cols = zoomed_image.shape[:2]
    if rows < new_size[0] or cols < new_size[0]:
        pad_rows = max(0, (new_size[0] - rows) // 2)
        pad_cols = max(0, (new_size[0] - cols) // 2)
        zoomed_image = cv2.copyMakeBorder(zoomed_image, pad_rows, new_size[0] - rows - pad_rows, pad_cols, new_size[0] - cols - pad_cols, cv2.BORDER_CONSTANT, value=0)
        zoomed_mask = cv2.copyMakeBorder(zoomed_mask, pad_rows, new_size[0] - rows - pad_rows, pad_cols, new_size[0] - cols - pad_cols, cv2.BORDER_CONSTANT, value=0)
    elif rows > new_size[0] or cols > new_size[0]:
        zoomed_image = zoomed_image[:new_size[0], :new_size[0]]
        zoomed_mask = zoomed_mask[:new_size[0], :new_size[0]]

    return zoomed_image, zoomed_mask

def width_shift(image, mask):
    shift_x = np.random.uniform(-0.2, 0.2) * image.shape[1]
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, 0]])
    width_shift_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    width_shift_mask = cv2.warpAffine(mask, translation_matrix, (mask.shape[1], mask.shape[0]))

    if width_shift_image.shape[1] > new_size[0]:
        width_shift_image = width_shift_image[:, :new_size[0]]
        width_shift_mask = width_shift_mask[:, :new_size[0]]
    elif width_shift_image.shape[1] < new_size[0]:
        pad_width = new_size[0] - width_shift_image.shape[1]
        width_shift_image = np.pad(width_shift_image, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
        width_shift_mask = np.pad(width_shift_mask, ((0, 0), (0, pad_width), (0, 0)), mode='constant')

    return width_shift_image, width_shift_mask

def height_shift(image, mask):
    shift_y = np.random.uniform(-0.2, 0.2) * image.shape[0]
    translation_matrix = np.float32([[1, 0, 0], [0, 1, shift_y]])
    height_shift_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    height_shift_mask = cv2.warpAffine(mask, translation_matrix, (mask.shape[1], mask.shape[0]))

    if height_shift_image.shape[0] > new_size[0]:
        height_shift_image = height_shift_image[:new_size[0], :]
        height_shift_mask = height_shift_mask[:new_size[0], :]
    elif height_shift_image.shape[0] < new_size[0]:
        pad_height = new_size[0] - height_shift_image.shape[0]
        height_shift_image = np.pad(height_shift_image, ((0, pad_height), (0, 0), (0, 0)), mode='constant')
        height_shift_mask = np.pad(height_shift_mask, ((0, pad_height), (0, 0), (0, 0)), mode='constant')

    return height_shift_image, height_shift_mask

def rotate(image, mask):
    angle = np.random.uniform(-20, 20)
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotate_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    rotate_mask = cv2.warpAffine(mask, rotation_matrix, (cols, rows))

    if rotate_image.shape[0] > new_size[0] or rotate_image.shape[1] > new_size[0]:
        rotate_image = cv2.resize(rotate_image, (new_size[0], new_size[0]))
        rotate_mask = cv2.resize(rotate_mask, (new_size[0], new_size[0]))
    elif rotate_image.shape[0] < new_size[0] or rotate_image.shape[1] < new_size[0]:
        pad_height = max(0, new_size[0] - rotate_image.shape[0])
        pad_width = max(0, new_size[0] - rotate_image.shape[1])
        rotate_image = np.pad(rotate_image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
        rotate_mask = np.pad(rotate_mask, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')

    return rotate_image, rotate_mask

def save_data_to_disk(X, y_mask, y_class_cat, filepath):
    with h5py.File(filepath, 'w') as hf:
        hf.create_dataset('X', data=X)
        hf.create_dataset('y_mask', data=y_mask)
        hf.create_dataset('y_class_cat', data=y_class_cat)

#def gaussian_noise(image):
#    noisy_image = random_noise(image, mode='gaussian', mean=0, var=0.01, clip=True)
#    noisy_image = (255 * noisy_image).astype(np.uint8)
#    noisy_image = torch.tensor(noisy_image, dtype=torch.float32)
#    
#    return noisy_image

num_plots = int(input("Number of image test plots after preprocessing: "))

X_images, y_bounding_boxes, y_class_labels = init_dataset()

print('Original imagery:')
print(f'Images shape: {X_images.shape}')
print(f'Masks shape: {y_bounding_boxes.shape}')
print(f'Labels shape: {y_class_labels.shape}')

count_labels(y_class_labels)

new_size = (100, 100)
X_cropped_images, y_cropped_masks, y_cropped_y_class_labels = crop_images(X_images, y_bounding_boxes, y_class_labels, new_size)

print('Cropped imagery:')
print(f'Images shape: {X_cropped_images.shape}')
print(f'Masks shape: {y_cropped_masks.shape}')
print(f'Labels shape: {y_cropped_y_class_labels.shape}')

count_labels(y_cropped_y_class_labels)

X_separated_images, y_separated_masks, y_separated_labels = separate_defects(X_cropped_images, y_cropped_masks, y_cropped_y_class_labels)

print('Separated imagery:')
print(f'Images shape: {X_separated_images.shape}')
print(f'Masks shape: {y_separated_masks.shape}')
print(f'Labels shape: {y_separated_labels.shape}')

count_labels(y_separated_labels)

X_balanced_images, y_balanced_masks, y_balanced_labels = balance_dataset(X_separated_images, y_separated_masks, y_separated_labels)

y_balanced_masks = clean_none_labels(y_balanced_masks, y_balanced_labels)

y_balanced_labels_categorical, unique_label_list = one_hot_encode_labels(y_balanced_labels)

print('Balanced imagery:')
print(f"Images shape: {X_balanced_images.shape}")
print(f"Masks shape: {y_balanced_masks.shape}")
print(f"Labels shape: {y_balanced_labels.shape}")
print(f"Encoded labels shape: {y_balanced_labels_categorical.shape}")

count_labels(y_balanced_labels)

# Split data into train and validation sets
X_train, X_test, y_train_mask, y_test_mask, y_train_class_cat, y_test_class_cat = train_test_split(X_balanced_images, y_balanced_masks, y_balanced_labels_categorical, test_size=0.2, random_state=666)

y_train_class = decode_labels(y_train_class_cat, unique_label_list)
y_test_class = decode_labels(y_test_class_cat, unique_label_list)

print("Training data:")
print("X_train shape:", X_train.shape)
print("y_train_mask shape:", y_train_mask.shape)
print("y_train_class_cat shape:", y_train_class_cat.shape)
count_labels(y_train_class)

print("Test data:")
print("X_test shape:", X_test.shape)
print("y_test_mask shape:", y_test_mask.shape)
print("y_test_class_cat shape:", y_test_class_cat.shape)
count_labels(y_test_class)

# Data Augmentation:
# rotation ,width_shift, height_shift, shear, zoom, horizontal_flip, noise
X_train_augmented, y_train_mask_augmented, y_train_class_augmented_cat = augment_data(X_train, y_train_mask, y_train_class_cat)

y_train_class_augmented = decode_labels(y_train_class_augmented_cat, unique_label_list)

y_train_mask_augmented = clean_none_labels(y_train_mask_augmented, y_train_class_augmented)

print('Augmented training imagery:')
print(f"Images shape: {X_train_augmented.shape}")
print(f"Masks shape: {y_train_mask_augmented.shape}")
print(f"Labels shape: {y_train_class_augmented.shape}")
print(f"Encoded labels shape: {y_train_class_augmented_cat.shape}")

plot_images(X_train_augmented, y_train_mask_augmented, y_train_class_augmented, num_samples=num_plots)

# save the augmented dataset to disk
file_name_train = "train_data.h5"
file_name_test = "test_data.h5"

file_path_train = os.path.join(downloads_path, file_name_train)
file_path_test = os.path.join(downloads_path, file_name_test)

save_data_to_disk(X_train_augmented, y_train_mask_augmented, y_train_class_augmented_cat, file_path_train)
save_data_to_disk(X_test, y_test_mask, y_test_class_cat, file_path_test)

print(f"Training data saved to {file_path_train}.")
print(f"Validation data saved to {file_path_test}.")