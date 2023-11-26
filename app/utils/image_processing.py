from PIL import ImageColor
import numpy as np


'''
Original label description

3,sky
5,tree
10,grass
14,earth;rock
17,mountain;mount
18,plant;flora;plant;life
22,water
27,sea
61,river
'''

def rgb2hex(rgb):
    r, g, b = rgb
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

ELEMENTS = {
    0: 'sky',
    1: 'tree',
    2: 'grass',
    3: 'earth/rock',
    4: 'mountain',
    5: 'plants/flora',
    6: 'water',
    7: 'sea',
    8: 'river'
}

ELEMENTS_2_COLOR = {
    'sky': '#467bc7',
    'tree': '#044005',
    'grass': '#3ec240',
    'earth/rock': '#704103',
    'mountain': '#140f06',
    'plants/flora': '#007d5e',
    'water': '#09818f',
    'sea': '#16098f',
    'river': '#74bedb'
}

ELEMENTS_ONE_HOT_ENCODING = {}
i = 0
for key, element in ELEMENTS_2_COLOR.items():
    color = np.array(ImageColor.getcolor(element, 'RGB'))
    l = np.zeros(12)
    l[i] = 1.0
    ELEMENTS_ONE_HOT_ENCODING[element] = l
    i += 1


def find_the_most_similar_color_encoding(color):
    min_distance = None
    best_encoding = None
    for hex_color, encoding in ELEMENTS_ONE_HOT_ENCODING.items():
        color_arr = np.array(ImageColor.getcolor(hex_color, 'RGB'))
        distance = np.linalg.norm(color-color_arr)
        if not min_distance:
            min_distance = distance
            best_encoding = encoding
        elif distance < min_distance:
            min_distance = min_distance
            best_encoding = encoding
    return best_encoding
         

def find_color_encoding(color):
    hex_color = rgb2hex(color)
    color_encoding = ELEMENTS_ONE_HOT_ENCODING.get(hex_color, None)
    if color_encoding is not None:
        return color_encoding
    return find_the_most_similar_color_encoding(color)


def convert_img_for_gaugan(input_image):
    image = np.zeros((256,256,12))
    for x in range(250):
        for y in range(250):
            image[x][y] = find_color_encoding(input_image[x][y])
    return image