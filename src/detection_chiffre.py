import struct

LABELS_FILE_MAGIC = 2049
IMAGES_FILE_MAGIC = 2051

def read_labels(filename, limit=None):
    with open(filename, 'rb') as f:
        magic = struct.unpack('>i', f.read(4))[0]
        if magic != LABELS_FILE_MAGIC:
            raise RuntimeError(
            "Invalid file type for labels: %s, expected %s"
            % (magic, LABELS_FILE_MAGIC))
        nb_labels = struct.unpack('>i', f.read(4))[0]
        if limit:
            nb_labels = min(limit, nb_labels)
        result = [struct.unpack('B', f.read(1))[0]
            for i in range(0, nb_labels)]
    return result

def read_images(filename, limit=None):
 with open(filename, 'rb') as f:
    magic = struct.unpack('>i', f.read(4))[0]
        if magic != IMAGES_FILE_MAGIC:
            raise RuntimeError(
            "Invalid file type for images: %s, expected %s"
            % (magic, IMAGES_FILE_MAGIC))
        nb_images = struct.unpack('>i', f.read(4))[0]
        if limit:
            nb_images = min(limit, nb_images)
            nb_rows = struct.unpack('>i', f.read(4))[0]
            nb_columns = struct.unpack('>i', f.read(4))[0]
        result = []
        for i in range(0, nb_images):
            pixels = [struct.unpack('B', f.read(1))[0]/255.0
            for k in range(0, nb_rows*nb_columns)]
        result.append(pixels)
    return result

def show_image(image):
    result = ""
    for i in range(0, 28):
        line = ""
        for j in range(0, 28):
            if self.pixels[i*self.width+j] < 0.01:
            line += " "
            elif self.pixels[i*self.width+j] < 0.5:
            line += "."
            else:
            line += "#"
        result += line + "\n"