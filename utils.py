import os
import random
import string
import cv2
import numpy as np 
from PIL import Image
import math

# ====================
def read_file(path):
    with open(path, 'r') as f:
        return f.readlines()

# ======== CONSTANTS ====================
ALPHA_RANGE = (0.35, 0.7)
FONT_THICKNESS_RANGE = (3, 7)
IMAGE_EXT = set(['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'])

FIXED_TEXTS = [text.rstrip('\n') for text in read_file('data/watermark_texts.txt')]
# FIXED_TEXTS = read_file('data/watermark_texts.txt')

print(FIXED_TEXTS[:10])

CLEAN_DIR = 'data/images_'
WATERMARK_DIR = 'data/watermark'

# ======== Utils for utils :D ==============

CV2_FONTS = [
    #cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_ITALIC,
    cv2.QT_FONT_BLACK,
    cv2.QT_FONT_NORMAL
]


def random_float(x, y):
    return random.random()*(y-x)+x


def get_text_size(text, font, font_scale, thickness):
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    return w, h+baseline


def get_font_scale(needed_height, text, font, thickness):
    w, h = get_text_size(text, font, 1, thickness)
    return needed_height/h


def place_text(image, text, color=(255,255,255), alpha=1, position=(0, 0), angle=0,
               font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, thickness=3):
    image = np.array(image)
    overlay = np.zeros_like(image)
    output = image.copy()
    
    cv2.putText(overlay, text, position, font, font_scale, color, thickness)
    
    if angle != 0:
        text_w, text_h = get_text_size(text, font, font_scale, thickness)
        rotate_M = cv2.getRotationMatrix2D((position[0]+text_w//2, position[1]-text_h//2), angle, 1)
        overlay = cv2.warpAffine(overlay, rotate_M, (overlay.shape[1], overlay.shape[0]))
    
    overlay[overlay==0] = image[overlay==0]
    cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)
    
    return Image.fromarray(output)


def get_random_font_params(text, text_height, fonts, font_thickness_range):
    font = random.choice(fonts)
    font_thickness_range_scaled = [int(font_thickness_range[0]*(text_height/35)),
                                   int(font_thickness_range[1]*(text_height/85))]
    try:
        font_thickness = min(random.randint(*font_thickness_range_scaled), 2)
    except ValueError:
        font_thickness = 2
    font_scale = get_font_scale(text_height, text, font, font_thickness)
    return font, font_scale, font_thickness


def place_random_centered_watermark(
        pil_image, 
        text,
        center_point_range_shift=(-0.025, 0.025),
        random_angle=(0,0),
        text_height_in_percent_range=(0.15, 0.18),
        text_alpha_range=(0.23, 0.5),
        fonts=CV2_FONTS,
        font_thickness_range=(2, 7),
        colors=[(255,255,255)]
    ):
    w, h = pil_image.size
    
    position_shift_x = random_float(*center_point_range_shift)
    offset_x = int(w*position_shift_x)
    position_shift_y = random_float(*center_point_range_shift)
    offset_y = int(w*position_shift_y)
    
    text_height = int(h*random_float(*text_height_in_percent_range))
    
    font, font_scale, font_thickness = get_random_font_params(text, text_height, fonts, font_thickness_range)

    text_width, _ = get_text_size(text, font, font_scale, font_thickness)
    
    position_x = int((w/2)-text_width/2+offset_x)
    position_y = int((h/2)+text_height/2+offset_y)
    
    return place_text(
        pil_image, 
        text,
        color=random.choice(colors),
        alpha=random_float(*text_alpha_range),
        position=(position_x, position_y), 
        angle=random.randint(*random_angle),
        thickness=font_thickness,
        font=font, 
        font_scale=font_scale
    )


def place_random_watermark(
        pil_image, 
        text,
        random_angle=(0,0),
        text_height_in_percent_range=(0.10, 0.18),
        text_alpha_range=(0.18, 0.4),
        fonts=CV2_FONTS,
        font_thickness_range=(2, 6),
        colors=[(255,255,255)]
    ):
    w, h = pil_image.size
    
    text_height = int(h*random_float(*text_height_in_percent_range))
    
    font, font_scale, font_thickness = get_random_font_params(text, text_height, fonts, font_thickness_range)

    text_width, _ = get_text_size(text, font, font_scale, font_thickness)
    
    position_x = random.randint(0, max(w-text_width, 10))
    position_y = random.randint(text_height, h)
    
    return place_text(
            pil_image, 
            text,
            color=random.choice(colors),
            alpha=random_float(*text_alpha_range),
            position=(position_x, position_y), 
            angle=random.randint(*random_angle),
            thickness=font_thickness,
            font=font, 
            font_scale=font_scale
        )


def center_crop(image, w, h):
    center = image.shape
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2
    return image[int(y):int(y+h), int(x):int(x+w)]



def place_text_checkerboard(image, text, color=(255,255,255), alpha=1, step_x=0.1, step_y=0.1, angle=0,
                            font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, thickness=3):
    image_size = image.size
    
    image = np.array(image.convert('RGB'))
    if angle != 0:
        border_scale = 0.4
        overlay_size = [int(i*(1+border_scale)) for i in list(image_size)]
    else:
        overlay_size = image_size
        
    w, h = overlay_size
    overlay = np.zeros((overlay_size[1], overlay_size[0], 3))
    output = image.copy()
    
    text_w, text_h = get_text_size(text, font, font_scale, thickness)
    
    c = 0
    for rel_pos_x in np.arange(0, 1, step_x):
        c += 1
        for rel_pos_y in np.arange(text_h/h+(c%2)*step_y/2, 1, step_y):
            position = (int(w*rel_pos_x), int(h*rel_pos_y))
            cv2.putText(overlay, text, position, font, font_scale, color, thickness)
    
    if angle != 0:
        rotate_M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        overlay = cv2.warpAffine(overlay, rotate_M, (overlay.shape[1], overlay.shape[0]))
    
    overlay = center_crop(overlay, image_size[0], image_size[1])
    overlay[overlay==0] = image[overlay==0]
    overlay = overlay.astype(np.uint8)
    cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)
    
    return Image.fromarray(output)


def place_random_diagonal_watermark(
        pil_image, 
        text,
        random_step_x=(0.25, 0.4),
        random_step_y=(0.25, 0.4),
        random_angle=(-60,60),
        text_height_in_percent_range=(0.10, 0.18),
        text_alpha_range=(0.18, 0.4),
        fonts=CV2_FONTS,
        font_thickness_range=(2, 6),
        colors=[(255,255,255)]
    ):
    w, h = pil_image.size
    
    text_height = int(h*random_float(*text_height_in_percent_range))
    
    font, font_scale, font_thickness = get_random_font_params(text, text_height, fonts, font_thickness_range)

    text_width, _ = get_text_size(text, font, font_scale, font_thickness)
    
    return place_text_checkerboard(
            pil_image, 
            text,
            color=random.choice(colors),
            alpha=random_float(*text_alpha_range),
            step_x=random_float(*random_step_x),
            step_y=random_float(*random_step_y),
            angle=random.randint(*random_angle),
            thickness=font_thickness,
            font=font, 
            font_scale=font_scale
        )


def get_extension(filepath):
    return os.path.splitext(filepath)[-1]


def listdir_rec(folder_path):
    filepaths = []
    for root, dirname, files in os.walk(folder_path):
        for file in files:
            filepaths.append(os.path.join(root, file))
    return filepaths


def list_images(folder_path):
    files = listdir_rec(folder_path)
    return [f for f in files if get_extension(f) in IMAGE_EXT]


def read_image_rgb(path):
    pil_img = Image.open(path)
    pil_img.load()
    if pil_img.format is 'PNG' and pil_img.mode is not 'RGBA':
        pil_img = pil_img.convert('RGBA')
    pil_img = pil_img.convert('RGB')
    return pil_img



def save_iterable(path, _iter):
    with open(path, 'w') as f:
        for el in _iter:
            f.write(f'{el}\n')
    print(f'Saved successfully to {path}')


def get_filepaths_recursive(dir):
    import itertools
    return list(itertools.chain.from_iterable(
        [os.path.join(root, file) for file in files]
        for root, _, files in os.walk(dir)
    ))


def get_filenames_recursive(dir):
    import itertools
    return list(itertools.chain.from_iterable(
        files
        for root, _, files in os.walk(dir)
    ))


def filter_by_extension(filenames, ext):
    return [f for f in filenames if f.endswith(ext)]


def parallelize(func, inputs, n_workers=4):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(func, inputs))
    
    return results



# ====================================================================

def generate_random_string(length):
    return ''.join(random.choices(string.ascii_lowercase + string.ascii_uppercase + string.digits + ' ', k=length))

def generate_random_url(length, www_prob=0.5):
    suffixes = ('ru', 'en', 'com', 'su', 'net')
    random_s = ''.join(random.choices(string.ascii_lowercase*3 + string.digits + '-', k=length))
    url = random_s+'.'+random.choice(suffixes)
    if random.random() < www_prob:
        url = 'www.'+url
    return url


def diagonal_watermark(pil_image, text):
    return place_random_diagonal_watermark(
        pil_image, 
        text,
        random_step_x=(0.25, 0.4),
        random_step_y=(0.25, 0.4),
        random_angle=(-60,60),
        text_height_in_percent_range=get_text_height_in_percent_range(text),
        text_alpha_range=ALPHA_RANGE,
        fonts=CV2_FONTS,
        font_thickness_range=FONT_THICKNESS_RANGE,
        colors=COLORS,
    )

def centered_watermark(pil_image, text):
    return place_random_centered_watermark(
        pil_image, 
        text,
        center_point_range_shift=(-0.025, 0.025),
        random_angle=(0,0),
        text_height_in_percent_range=get_text_height_in_percent_range(text),
        text_alpha_range=ALPHA_RANGE,
        fonts=CV2_FONTS,
        font_thickness_range=FONT_THICKNESS_RANGE,
        colors=COLORS,
    )

def centered_angled_watermark(pil_image, text):
    return place_random_centered_watermark(
        pil_image, 
        text,
        center_point_range_shift=(-0.005, 0.005),
        random_angle=(-45,45),
        text_height_in_percent_range=get_text_height_in_percent_range(text),
        text_alpha_range=ALPHA_RANGE,
        fonts=CV2_FONTS,
        font_thickness_range=FONT_THICKNESS_RANGE,
        colors=COLORS,
    )

def random_watermark(pil_image, text):
    return place_random_watermark(
        pil_image, 
        text,
        random_angle=(0,0),
        text_height_in_percent_range=get_text_height_in_percent_range(text),
        text_alpha_range=ALPHA_RANGE,
        fonts=CV2_FONTS,
        font_thickness_range=FONT_THICKNESS_RANGE,
        colors=COLORS,
    )

def random_angled_watermark(pil_image, text):
    return place_random_watermark(
        pil_image, 
        text,
        random_angle=(-15,15),
        text_height_in_percent_range=get_text_height_in_percent_range(text),
        text_alpha_range=ALPHA_RANGE,
        fonts=CV2_FONTS,
        font_thickness_range=FONT_THICKNESS_RANGE,
        colors=COLORS,
    )


def create_color_palette():
    colors = []
    colors.extend([(i,i,i) for i in range(245, 255)]) # white shades
    colors.extend([(i,i,i) for i in range(245, 255)]) # white shades
    colors.extend([(i,i,i) for i in range(245, 255)]) # white shades
    colors.extend([(i,i,i) for i in range(240, 255)]) # white shades
    colors.extend([(i,i,i) for i in range(240, 255)]) # white shades
    colors.extend([(i,i,i) for i in range(240, 255)]) # white shades
    #
    colors.extend([(i,0,0) for i in range(245, 255)]) # red shades
    colors.extend([(0,i,0) for i in range(245, 255)]) # green shades
    colors.extend([[random.randint(210,255) for i in range(3)] for j in range(15)])
    return colors

def get_text_height_in_percent_range(text):
    if len(text) >= 7:
        k = len(text)/7
        text_height_start = max(0.06, 0.095/math.sqrt(k))
        text_height_end = max(0.10, 0.165/math.sqrt(k))
        res = (text_height_start, text_height_end)
        return res
    else:
        return (0.07, 0.135)

# ==================================
COLORS = create_color_palette()
FUNCTIONS = [
    random_watermark, random_angled_watermark,
    centered_watermark, centered_angled_watermark,
    diagonal_watermark
]

F_PROBABILITIES = [0.2, 0.05, 0.3, 0.15, 0.3]
assert sum(F_PROBABILITIES) == 1


def resize_by_max(img, size=600):
    maxind = 0 if img.size[0]>img.size[1] else 1
    rescale = size/img.size[maxind]
    return img.resize([int(i*rescale) for i in img.size])


def get_random_watermark_text():
    return random.choice(FIXED_TEXTS)

    
def generate_watermark(img) -> Image:
    watermark_func = np.random.choice(FUNCTIONS, 1, p=F_PROBABILITIES)[0]
    img_resized = resize_by_max(img, size=600)
    return watermark_func(img_resized, get_random_watermark_text())


def generate_watermark_dataset(clean_dir, out_dir, n_workers=4):
    from functools import partial
    clean_images_path = get_filepaths_recursive(clean_dir)
    
    def add_watermark_and_save(inputs_, out_dir):
        try:
            index, clean_image_path = inputs_
            pil_image = Image.open(clean_image_path).convert('RGB')
            watermarked = generate_watermark(pil_image)
            save_path = os.path.join(out_dir, f'image_{index}.{get_extension(clean_image_path)}')
            watermarked.save(save_path)
        except Exception as e:
            print(e)
    
    func_ = partial(add_watermark_and_save, out_dir=out_dir)
    inputs = enumerate(clean_images_path)
    parallelize(func_, inputs, n_workers=n_workers)


generate_watermark_dataset(CLEAN_DIR, WATERMARK_DIR, n_workers=4)





