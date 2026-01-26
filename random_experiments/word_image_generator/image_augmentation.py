import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


def to_grayscale(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def normalize_contrast(img):
    return cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


def add_gaussian_noise(img, sigma_range=(5, 25)):
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img, amount=0.003):
    noisy = img.copy()
    num_pixels = int(amount * img.size)

    # Salt
    coords = [np.random.randint(0, i, num_pixels) for i in img.shape]
    noisy[tuple(coords)] = 255

    # Pepper
    coords = [np.random.randint(0, i, num_pixels) for i in img.shape]
    noisy[tuple(coords)] = 0

    return noisy


def gaussian_blur(img, k_range=(3, 5)):
    k = random.choice(range(k_range[0], k_range[1] + 1, 2))
    return cv2.GaussianBlur(img, (k, k), 0)


def motion_blur(img, k_range=(3, 7)):
    k = random.choice(range(k_range[0], k_range[1] + 1, 2))
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = np.ones(k)
    kernel /= k
    return cv2.filter2D(img, -1, kernel)

def add_illumination_gradient(img, strength=0.4):
    h, w = img.shape
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xv, yv = np.meshgrid(x, y)

    gradient = (xv + yv) / 2
    gradient = 1 + strength * (gradient - 0.5)

    shaded = img.astype(np.float32) * gradient
    return np.clip(shaded, 0, 255).astype(np.uint8)


def jpeg_compression(img, quality_range=(30, 90)):
    quality = random.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)


def random_threshold(img):
    thresh_type = random.choice([
        cv2.THRESH_BINARY,
        cv2.THRESH_BINARY_INV
    ])
    thresh_val = random.randint(100, 160)
    _, th = cv2.threshold(img, thresh_val, 255, thresh_type)
    return th


def morphology_noise(img):
    k = random.choice([1, 2])
    kernel = np.ones((k, k), np.uint8)

    if random.random() < 0.5:
        return cv2.erode(img, kernel, iterations=1)
    else:
        return cv2.dilate(img, kernel, iterations=1)


def small_affine_transform(img, max_shift=2):
    h, w = img.shape
    tx = random.randint(-max_shift, max_shift)
    ty = random.randint(-max_shift, max_shift)

    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderValue=255)




def scanned_word_augmentation(img):
    img = to_grayscale(img)
    img = normalize_contrast(img)

    if random.random() < 0.5:
        img = add_gaussian_noise(img)

    if random.random() < 0.3:
        img = add_salt_pepper_noise(img)

    if random.random() < 0.4:
        img = gaussian_blur(img)

    if random.random() < 0.2:
        img = motion_blur(img)

    if random.random() < 0.5:
        img = add_illumination_gradient(img)

    if random.random() < 0.4:
        img = jpeg_compression(img)

    if random.random() < 0.3:
        img = morphology_noise(img)

    if random.random() < 0.3:
        img = small_affine_transform(img)

    return img





img = cv2.imread("/Users/xai/Personal/Projects/TeluguOCR/correction_model/test_data/words/images/image_46.png")
img = to_grayscale(img)
img = normalize_contrast(img)

funcs = [add_gaussian_noise, add_salt_pepper_noise, gaussian_blur, motion_blur, add_illumination_gradient, jpeg_compression, random_threshold, morphology_noise, small_affine_transform]

# r, c = 3, 3
# fig, ax = plt.subplots(r, c)
#
# for i in range(r):
#     for j in range(c):
#         func = funcs[c*i+j]
#         ax[i][j].imshow(func(img), cmap='gray')
#         ax[i][j].set_title(func.__name__)
#
# plt.show()



for i in range(5):
    img = cv2.imread("/Users/xai/Personal/Projects/TeluguOCR/correction_model/test_data/words/images/image_46.png")

    img = scanned_word_augmentation(img)

    plt.imshow(img, cmap='gray')
    plt.show()




# import cv2
# import matplotlib.pyplot as plt
#
# # fol_name = '/Users/xai/Personal/Projects/TeluguOCR/correction_model/test_data/words/images'
# fol_name = '/Users/xai/Downloads/train/images'
# files = os.listdir(fol_name)
#
# for file in files:
#     img = cv2.imread(f"{fol_name}/{file}")
#     plt.imshow(img)
#     plt.show()
#
#     img = remove_padding(img)
#     plt.imshow(img, cmap='gray')
#     plt.show()
#
#     scaled_img = cv2.resize(img, (224, 224))
#     plt.imshow(scaled_img, cmap='gray')
#     plt.show()