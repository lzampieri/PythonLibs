from PIL import Image
import numpy as np

def read_jpg(filepath):
    img = np.asarray( Image.open(filepath) )

    return {
        'img': img,
        'filename': filepath,
    }