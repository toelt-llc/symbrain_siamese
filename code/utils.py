import cv2, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from datasets import load_dataset
from keras import ops
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold
import itertools

hf_dataset = load_dataset("agucci/mri-sym2")

def non_zeros(img, plot = True):
    """ Detects the coordinates of all non-zero (non-black) pixels along the four edges of the image. 
        These points are returned into a single list of (x, y).

        img : PIL Image
    """

    # img = example['half1_noise']
    img_array = np.asarray(img.convert("L"))
    height, width = img_array.shape

    # Find the edge points from the top line
    edge_points = []
    min_x, min_y = width, height
    max_x, max_y = 0, 0
    for y in range(height):
        for x in range(width):
            if img_array[y, x]:
                edge_points.append((x, y))
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    if plot: 
        # Plot the half1_noise and edge points
        plt.figure(figsize=(4, 4))
        plt.imshow(img_array, cmap='gray')
        if edge_points:
            x_points, y_points = zip(*edge_points)
            plt.scatter(x_points, y_points, color='red', s=10)  # Plot edge points as red dots
        plt.title('Brain segmentation - non zeros pixels')
        plt.axis('off')
        plt.show()
        # Print the number of intersection points for reference
        print("Number of Edge Points:", len(edge_points))
        print(f"Min coordinates: ({min_x}, {min_y})")
        print(f"Max coordinates: ({max_x}, {max_y})")

    return img_array, edge_points

def segment(img, edge_points):
    """ Segment the the brain part from the black background.
        Returns an Image without background. 

        img : PIL Image
        edge_points : (x,y) list of iamge edge points
    """

    img_array = np.array(img.convert('L'))
    height, width = img_array.shape

    # Initialize a binary mask
    mask = np.zeros_like(img_array, dtype=np.uint8)

    for x, y in edge_points[:width]:
        mask[y, x] = 255

    for x, y in edge_points[width:2*width]:
        mask[y, x] = 255

    for x, y in edge_points[2*width:2*width+height]:
        mask[y, x] = 255

    for x, y in edge_points[2*width+height:]:
        mask[y, x] = 255

    rgba_img = np.zeros((height, width, 4), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            rgba_img[y, x, :3] = img_array[y, x]  # Copy intensity from original half1_noise
            rgba_img[y, x, 3] = mask[y, x]       # Set transparency from mask

    transparent_img = Image.fromarray(rgba_img, 'RGBA')

    return transparent_img

def rotate(img, line, show_line = False):
    """
        Rotates a PIL image so that the specified line becomes horizontal and centered on the image. 
        Applies the rotation of the image pixels to follow the line and returns the adjusted PIL image.
        Option (show_line) to draw the line on the rotated image for debug.
        Returns a rotated Image without background.

        img :   PIL Image
        line :  Json string defining the line coordinates
    """
    img_array = np.array(img)
    height, width = img_array.shape[:2]

    # Draw the (long) line
    draw = ImageDraw.Draw(img.copy())
    line_coords = eval(line.replace('} {', '}, {'))

    for i in range(len(line_coords) - 1):
        start_point = (line_coords[i]['x'], line_coords[i]['y'])
        end_point = (line_coords[i+1]['x'], line_coords[i+1]['y'])

        # line slope & intercept + border points
        a = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
        b = start_point[1] - a * start_point[0]
        left_point = (0, int(b))                # (0, b)
        right_point = (width, int(width*a+b))   # (290, 290*a + b)

        angle = np.arctan2(right_point[1] - left_point[1],
                right_point[0] - left_point[0]) * 180 / np.pi
        center_x = (left_point[0] + right_point[0]) / 2
        center_y = (left_point[1] + right_point[1]) / 2
        # 
        center_new_x = img_array.shape[1] / 2
        center_new_y = img_array.shape[0] / 2

        # Translation
        offset_x = center_new_x - center_x
        offset_y = center_new_y - center_y
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        translated_img = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))
        rotation_matrix = cv2.getRotationMatrix2D((center_new_x, center_new_y), angle, 1)
        rotated_img = cv2.warpAffine(translated_img, rotation_matrix, (img_array.shape[1], img_array.shape[0]))

        rotated_img_pil = Image.fromarray(rotated_img, 'RGBA')

        # (Might not be necessary)
        # Calculate the offset caused by the rotation
        rotated_center = np.dot(rotation_matrix, np.array([center_x, center_y, 1]))
        offset_x_rotated = center_new_x - rotated_center[0]
        offset_y_rotated = center_new_y - rotated_center[1]
        # Adjust the position of the rotated half1_noise
        M_adjusted = np.float32([[1, 0, offset_x_rotated], [0, 1, offset_y_rotated]])
        final_img = cv2.warpAffine(rotated_img, M_adjusted, (img_array.shape[1], img_array.shape[0]))
        final_rotated_img_pil = Image.fromarray(final_img, 'RGBA')

        # Draw the (long) line
        if show_line:
            draw1, draw2 = ImageDraw.Draw(rotated_img_pil), ImageDraw.Draw(final_rotated_img_pil)
            draw1.line([left_point, right_point], fill=(255, 0, 0, 255), width=1)
            draw2.line([(0, 145.0), (290, 145.0)], fill=(255, 0, 0, 255), width=1)

    return final_rotated_img_pil

def slice_aligned(transparent_img):
    """ Slice an Image aligned to the ((0, 145) (290, 145)) axis line.
        To run after the previous pre_processing functions.
        Returns the two separated Images.

        transparent_img : PIL Image
    """

    # intercept = 145
    _, height = transparent_img.size
    intercept = int(height / 2)

    img_array = np.array(transparent_img)#.convert('L'))
    reshaped_1 = Image.fromarray(img_array[intercept:, :, :], 'RGBA')
    reshaped_2 = Image.fromarray(img_array[:intercept, :, :], 'RGBA')

    return(reshaped_1.convert("L"), reshaped_2.convert("L"))

# def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
#     """Creates a plot of pairs and labels, and prediction if it's test dataset.

#     Arguments:
#         pairs: Numpy Array, of pairs to visualize, having shape
#                (Number of pairs, 2, 28, 28).
#         to_show: Int, number of examples to visualize (default is 6)
#                 `to_show` must be an integral multiple of `num_col`.
#                  Otherwise it will be trimmed if it is greater than num_col,
#                  and incremented if if it is less then num_col.
#         num_col: Int, number of images in one row - (default is 3)
#                  For test and train respectively, it should not exceed 3 and 7.
#         predictions: Numpy Array of predictions with shape (to_show, 1) -
#                      (default is None)
#                      Must be passed when test=True.
#         test: Boolean telling whether the dataset being visualized is
#               train dataset or test dataset - (default False).

#     Returns:
#         None.
#     """

#     # Define num_row
#     # If to_show % num_col != 0
#     #    trim to_show,
#     #       to trim to_show limit num_row to the point where
#     #       to_show % num_col == 0
#     #
#     # If to_show//num_col == 0
#     #    then it means num_col is greater then to_show
#     #    increment to_show
#     #       to increment to_show set num_row to 1
#     num_row = to_show // num_col if to_show // num_col != 0 else 1

#     # `to_show` must be an integral multiple of `num_col`
#     #  we found num_row and we have num_col
#     #  to increment or decrement to_show
#     #  to make it integral multiple of `num_col`
#     #  simply set it equal to num_row * num_col
#     to_show = num_row * num_col

#     # Plot the images
#     fig, axes = plt.subplots(num_row, num_col, figsize=(10, 10))
#     for i in range(to_show):
#         # If the number of rows is 1, the axes array is one-dimensional
#         if num_row == 1:
#             ax = axes[i % num_col]
#         else:
#             ax = axes[i // num_col, i % num_col]

#         ax.imshow(ops.concatenate([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
#         ax.set_axis_off()
#         if test:
#             ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
#         else:
#             ax.set_title("Label: {}".format(labels[i]))
#     if test:
#         plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
#     else:
#         plt.tight_layout(rect=(0, 0, 1.5, 1.5))
#     plt.show()

def add_noise(image, size='big', noise_type='noisy', rgb=False, force_shape=0):
    """
    Add a random noise area to the upper or lower half of the image, close to the middle.

    Parameters:
    - image: PIL.Image object.
    - shape: Shape of the noise area ('circle', 'square', or 'polygon').
    - size: string indicating the size of the noise area ('big', 'mid', 'lil').

    Returns:
    - image with added noise area.
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)
    shapes = ['circle', 'square']#, 'polygon']
    if force_shape == 0 or force_shape == 1:
        shape = shapes[force_shape]
    else:
        shape = shapes[random.randint(0, len(shapes) -1 )]

    #big 40x40 medium 20x20 small 10x10
    if size == 'big':
        x1, y1 = random.randint(66, 129), 105
        x2, y2 = x1+40, y1+40
    elif size == 'mid':
        x1, y1 = random.randint(66, 149), random.randint(106, 124)
        x2, y2 = x1+20, y1+20
    elif size == 'lil':
        x1, y1 = random.randint(66, 159), random.randint(106, 134)
        x2, y2 = x1+10, y1+10

    bbox = (x1, y1, x2, y2)

    if noise_type == 'plain':
        fill_color = 255        # Plain white
        if rgb: fill_color = (255, 255, 255)
    elif noise_type == 'noisy':
        # fill_color = 255
        if rgb: fill_color = tuple(random.randint(200, 215) for _ in range(3))  # Noisy white
        else: fill_color = random.randint(140, 215)

    # if bbox:
    if shape == 'circle':
        draw.ellipse(bbox, fill=fill_color)
    elif shape == 'square':
        draw.rectangle(bbox, fill=fill_color)
        # print('square')

    return image


## siamese torch 2
# TODO FIX calls

def cut_align(image, line, resize=True, reshape = (224,224), show=False):
    """
    Given an image and line annotation from the dataset, 
    remove background, segment, rotate, cut and re-align the two halves.
    Returns halves with the same view angle (second half is 180 rotated and mirrored).

    image : PIL input
    line :  Json string defining the line coordinates

    """
    _, edge_points = non_zeros(image, plot=False)
    transparent_img = segment(image, edge_points)
    rotated_img = rotate(transparent_img, line, show_line=False)
    slice1, slice2 = slice_aligned(rotated_img)
    slice1 = slice1.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    square1, square2 = Image.new('L', (290, 290), 0), Image.new('L', (290, 290), 0) # center slice in black square
    square1.paste(slice1, (0, 50)), square2.paste(slice2, (0, 50))

    if resize:
        square1, square2 = square1.resize(reshape), square2.resize(reshape)

    return square1, square2

def transforms(examples):
    """
    Convert to grayscale, and map cut_align to hf dataset, add column for each half
    """
    # careful with the conversion here, resize migth be needed on other datasets
    list_slices = [cut_align(im.convert("L"), line, resize=True) for im, line in zip(examples["image"], examples["line"])]
    examples["slice1"], examples["slice2"] = [i[0] for i in list_slices], [i[1] for i in list_slices]
    # examples["image_convert"] = [image.convert("L").resize((290, 290)) for image in examples["image"]]

    return examples

def transforms_noresize(examples):
    """
    transforms without resizing, temptative
    """
    # careful with the conversion here, resize migth be needed on other datasets
    list_slices = [cut_align(im.convert("L"), line, resize=False) for im, line in zip(examples["image"], examples["line"])]
    examples["slice1"], examples["slice2"] = [i[0] for i in list_slices], [i[1] for i in list_slices]
    # examples["image_convert"] = [image.convert("L").resize((290, 290)) for image in examples["image"]]

    return examples

def paired_stratified_split(dataset, test_size, stratify_by, random_state):
    """
    Used for the skfold dataset, used to replace hf train_test which didnt have stratify
    """
    # each consecutive pair has to be kept together
    pair_count = len(dataset) // 2
    pair_indices = np.arange(pair_count)
    pair_types = np.array(dataset[stratify_by][::2])  # type of first half of each pair

    # Split pair indices
    skf = StratifiedKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_state)
    train_pair_indices, test_pair_indices = next(skf.split(pair_indices, pair_types))

    # Convert pair indices to image indices
    train_indices = np.concatenate([2*train_pair_indices, 2*train_pair_indices+1])
    test_indices = np.concatenate([2*test_pair_indices, 2*test_pair_indices+1])

    # Sort indices to maintain original order
    train_indices.sort()
    test_indices.sort()

    return dataset.select(train_indices), dataset.select(test_indices)

def add_noise_range(image, size=0, noise_type='noisy', rgb=False, force_shape=None):
    """
    Add a random noise area to the upper or lower half of the image, close to the middle.

    Parameters:
    - image: PIL.Image object.
    - shape: Shape of the noise area ('circle', 'square', or 'polygon').
    - size: range indicating the size of the noise area (0 (2,2) to 9 maximum (40,40))

    Returns:
    - image with added noise area.
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)
    shapes = ['circle', 'square']#, 'polygon']
    if force_shape == 0 or force_shape == 1:
        shape = shapes[force_shape]
    else:
        shape = shapes[random.randint(0, len(shapes) -1 )]

    sizes = [int(2 + (40 - 2) * i / 9) for i in range(10)]  # 10 sizes from 2 to 40
    size_drawn = sizes[size]
    #big 40x40 medium 20x20 small 10x10
    #Xx (66)
    # Calculate valid x range based on size
    x1 = random.randint(61, 169 - size_drawn)
    y1 = random.randint(105, 145 - size_drawn)
    bbox = (x1, y1, x1 + size_drawn, y1 + size_drawn)

    if noise_type == 'plain':
        fill_color = 255        # Plain white
        if rgb: fill_color = (255, 255, 255)
    elif noise_type == 'noisy':
        # fill_color = 255
        if rgb: fill_color = tuple(random.randint(200, 215) for _ in range(3))  # Noisy white
        else: fill_color = random.randint(140, 215)

    # if bbox:
    if shape == 'circle':
        draw.ellipse(bbox, fill=fill_color)
    elif shape == 'square':
        draw.rectangle(bbox, fill=fill_color)
        # print('square')

    return image

def siamese_noise_dataset_fold_range(test_size=0.2, n_splits=5, noise_size=0, resize=True, t1=True, t2=True, random_state=4):
    """
    Creates n folds from the data without duplicating samples and alternates between T1 and T2 image types
    AND range of noise size, with modified add_noise.
    Noise_size : 0 (min) to 9 (max)
    """
    slice1, slice2, labels, image_types = [], [], [], []

    # T1 and T2
    transform_fn = transforms if resize else transforms_noresize
    dst1 = hf_dataset['train'].map(transform_fn, batched=True)
    dst2 = hf_dataset['test'].map(transform_fn, batched=True)

    # iterators for T1 and T2 data, used below
    t1_iter = enumerate(zip(dst1['slice1'], dst1['slice2']))
    t2_iter = enumerate(zip(dst2['slice1'], dst2['slice2']))

    # First part : loop the dataset to add noise and labels
    for (i1, (t1_slice1, t1_slice2)), (i2, (t2_slice1, t2_slice2)) in itertools.zip_longest(t1_iter, t2_iter, fillvalue=(None, (None, None))):
        # T1 data
        if t1 and i1 is not None:
            if i1 % 2 == 0:
                slice1.append(t1_slice1)
                slice2.append(t1_slice2)
                labels.append(0)
            else:
                if random.choice([True, False]):
                    slice1.append(add_noise_range(t1_slice1, size=noise_size))
                    slice2.append(t1_slice2)
                else:
                    slice1.append(t1_slice1)
                    slice2.append(add_noise_range(t1_slice2, size=noise_size))
                labels.append(1)
            image_types.append(0)  # T1 type

        # T2 data
        if t2 and i2 is not None:
            if i2 % 2 == 0:
                slice1.append(t2_slice1)
                slice2.append(t2_slice2)
                labels.append(0)
            else:
                if random.choice([True, False]):
                    slice1.append(add_noise_range(t2_slice1, size=noise_size))
                    slice2.append(t2_slice2)
                else:
                    slice1.append(t2_slice1)
                    slice2.append(add_noise_range(t2_slice2, size=noise_size))
                labels.append(1)
            image_types.append(1)  # T2 type

    ds_ = Dataset.from_dict({
        'slice1': slice1,
        'slice2': slice2,
        'label': labels,
        'image_type': image_types
    })

    print(f"Total dataset size: {len(ds_)}")
    print(f"Number of T1 images: {sum(1 for t in image_types if t == 0)}")
    print(f"Number of T2 images: {sum(1 for t in image_types if t == 1)}")

    # Second part : splits with val, stratified, every 2 images kept together
    ds_trainval, ds_test = paired_stratified_split(ds_, test_size=test_size, stratify_by='image_type', random_state=random_state)

    # stratified k-fold split
    indices = np.arange(len(ds_trainval) // 2)
    image_types = np.array(ds_trainval['image_type'][::2]) # type of first half of each pair
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_indices = list(skf.split(indices, image_types))

    ds_folds = []
    for fold, (train_pair_idx, val_pair_idx) in enumerate(fold_indices):
        # Convert pair indices to image indices
        train_idx = np.concatenate([2*train_pair_idx, 2*train_pair_idx+1])
        val_idx = np.concatenate([2*val_pair_idx, 2*val_pair_idx+1])
        train_idx.sort()
        val_idx.sort()

        ds_train = ds_trainval.select(train_idx)
        ds_val = ds_trainval.select(val_idx)

        ds_folds.append(DatasetDict({
            'train': ds_train,
            'valid': ds_val,
            'test': ds_test
        }))

    return ds_folds