from abc import ABC, abstractmethod
import os
import utils.str_tools
import xml.etree.ElementTree as ET
from PIL import Image
from multiprocessing.dummy import Pool
from shutil import copyfile

class DataSetReader(ABC):
    def __init__(self, dire, name, image_types, label_type):
        self.name = name
        self.dirs = {'root': dire}
        self.data = []
        self.class_type = set()
        images, labels = self.read_from_dir(self.dirs['root'], image_types, label_type)
        total_iamge = len(images)
        count = self.pair(images, labels)
        print('%d / %d images paired' % (count, total_iamge))

    @abstractmethod
    def resolve_labels(self, label_url) -> list:
        pass

    @abstractmethod
    def identify(self, image_url):
        pass

    def read_from_dir(self, search_path, image_types, label_type):
        data = utils.str_tools.get_file_list(search_path, image_types + [label_type], True)
        label_data_list = []
        for i in data[-1]:
            label_data_list.append(self.resolve_labels(i))

        if len(label_data_list) == 1:
            label_data_list = label_data_list[0]
        return [i for item in data[0:-1] for i in item], label_data_list

    def pair(self, data, labels):
        label_ids = [i[0] for i in labels]
        count = 0

        for i in data:
            try:
                pos = label_ids.index(self.identify(i))
                count += 1
                self.data.append([i] + labels[pos][1:])
            except ValueError as e:
                data.remove(i)
        return count


# class ClassificationDataSetReader(DataSetReader):
#     def __init__(self, dire, name):
#         super().__init__(dire, name)
#         self.class_types = set()


class ROCODataSetReader(DataSetReader):
    def __init__(self, dire, name='ROCO'):

        super().__init__(dire, name, ['jpg'], 'xml')

    def identify(self, image_url):
        _, file_name = os.path.split(image_url)
        return os.path.splitext(file_name)[0]

    def resolve_labels(self, label_url) -> list:
        path, file_name = os.path.split(label_url)
        file = ET.parse(label_url)
        root = file.getroot()

        objs = root.findall('object')
        obj_list = []
        for obj in objs:
            name = obj.find('name').text
            if name == 'armor':
                name = name + obj.find('armor_color').text + obj.find('armor_class').text
            if name == 'watcher':
                continue
            self.class_type.add(name)
            obj = obj.find('bndbox')
            x_min = float(obj.find('xmin').text)
            y_min = float(obj.find('ymin').text)
            x_max = float(obj.find('xmax').text)
            y_max = float(obj.find('ymax').text)
            obj_list.append([name, x_min, y_min, x_max, y_max])

        return [os.path.splitext(file_name)[0]] + obj_list


class DataSetConvertor(ABC):
    def __init__(self, dataset_reader: DataSetReader, dire=None, name=None):
        self.name = name if name is not None else dataset_reader.name
        self.dirs = {'root': os.path.join(dire, self.name) if dire is not None else dataset_reader.dirs['root']}
        self.data = dataset_reader.data

        self.class_type = dataset_reader.class_type

    @abstractmethod
    def build(self, percent):
        pass

    def append(self, dataset_reader: DataSetReader):
        self.data = self.data + dataset_reader.data
        self.class_type = set(list(self.class_type) + list(dataset_reader.class_type))


class YOLOConvertor(DataSetConvertor):
    def __init__(self, dataset_reader: DataSetReader, dire=None, name=None):
        super().__init__(dataset_reader, dire, name)

        self.dirs['train'] = os.path.join(self.dirs['root'], 'train')
        self.dirs['val'] = os.path.join(self.dirs['root'], 'val')
        self.dirs['train_labels'] = os.path.join(self.dirs['root'], 'train/labels')
        self.dirs['val_labels'] = os.path.join(self.dirs['root'], 'val/labels')
        self.dirs['train_images'] = os.path.join(self.dirs['root'], 'train/images')
        self.dirs['val_images'] = os.path.join(self.dirs['root'], 'val/images')

    def build(self, percent, copy_img):
        for i in self.dirs.values():
            try:
                print(i)
                os.mkdir(i)
            except:
                pass


        total = len(self.data)
        divide = int(total * percent)

        with Pool() as pool:
            pool.map(lambda x:self.write_yolo_label(x, 'train', copy_img), self.data[:divide])
        with Pool() as pool:
            pool.map(lambda x:self.write_yolo_label(x, 'val', copy_img), self.data[divide:total])
        self.write_config()
    def write_config(self):
        with open(self.dirs['root'] + '/' + self.name + '.yaml', 'w') as f:
            f.write('train: ' + self.dirs['train_images'] + '\n')
            f.write('val: ' + self.dirs['val_images'] + '\n')
            f.write('\n\n')
            f.write('nc: ' + str(len(self.class_type)) + '\n\n\n')
            f.write('names: ' + str(list(self.class_type)))

    def write_yolo_label(self, item, dtype, copy_img):
        image_url = item[0]
        img = Image.open(image_url)
        img_size = img.size
        _, name = os.path.split(image_url)
        if copy_img :
            try:
                copyfile(image_url, self.dirs['root'] + '\\' + dtype + '\\images\\' + name)
                print(dtype)
            except IOError as e:
                print("Unable to copy file. %s" % e)
            except:
                print("Unexpected error:", sys.exc_info())
        name = os.path.splitext(name)[0] + '.txt'
        label_url = self.dirs['root'] + '\\' + dtype + '\\labels\\' + name

        with open(label_url, 'w') as f:
            for label in item[1:]:
                class_name = label[0]
                index = list(self.class_type).index(class_name)

                buf = str(index) + ' ' + self.data2yolo(label[1:], img_size) + '\n'
                f.write(buf)

    def data2yolo(self, label_data: list, size):
        width, height = size
        x_min, y_min, x_max, y_max = label_data

        if y_max > height:
            y_max = height
        if x_max > width:
            x_max = width
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        return '%f %f %f %f' % (
            (x_max + x_min) / 2 / width,
            (y_max + y_min) / 2 / height,
            (x_max - x_min) / width,
            (y_max - y_min) / height
        )


if __name__ == '__main__':
    roco = ROCODataSetReader(
        'E:\\Documents\\project\\ubuntuProjects\\datasets\\DJI ROCO', 'ROCO')
    dataset = YOLOConvertor(roco, 'E:\Documents\project\\ubuntuProjects\\datasets')

    dataset.build(0.8, True)
