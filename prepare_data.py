import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
import os
import shutil
import csv
import random

data_dir = '/content/Labours Memory'

xml_folders = []
for f1 in os.listdir(data_dir):
    f1_path = data_dir + os.path.sep + f1
    if os.path.isdir(f1_path):
        for f2 in os.listdir(f1_path):
            f2_path = f1_path + os.path.sep + f2
            if os.path.isdir(f2_path):
                for f3 in os.listdir(f2_path):
                    f3_path = f2_path + os.path.sep + f3
                    if os.path.isdir(f3_path):
                        xml_folders.append(f3_path)

print(f'xml_folders:\n{xml_folders}')

#xml_folders = [xml_folders[4]]
def process_points(points):
  points = points.split()
  for i in range(len(points)):
      points[i] = [int(x) for x in points[i].split(',')]
  points = np.array(points, dtype=np.int32)
  points = points.reshape((-1, 1, 2))
  return points

lines_folder = '/content/data/lines'
if os.path.exists(lines_folder):
    shutil.rmtree(lines_folder)
os.mkdir(lines_folder)
img_id = 1

# we will store the values in this list later in csv
# each value is a tuple represent the path to img,
# and its label
train_csv_list = []
val_csv_list = []
p = 0.8
for folder in xml_folders:

    xml_folder = folder + os.path.sep + 'page'
    csv_list = []
    for xml in os.listdir(xml_folder):
        xml_path = xml_folder + os.path.sep + xml
        # parse the xml
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # extract points from the xml
        # for c in root[1][1]:
        #   print(c.attrib)
        points = root[1][1][0].attrib['points']
        #print(f'points before parsing: {points}')
        # encode points in ndarray
        points = process_points(points)
        #print(f'points after parsing: {points}')



        # read the image
        img_dir = folder
        img_name = root[1].attrib['imageFilename']
        img_path = img_dir + os.path.sep + img_name
        # some bad data exists
        if not os.path.isfile(img_path):
            print(f'Image not exist: {img_path}')
            continue
        #print(f'img_path: {img_path}')
        img = cv.imread(img_path, 0)

        # extract all lines
        for i in range(1, len(root[1][1]) - 1):
            elem = root[1][1][i]
            if len(elem) < 3:
                print(xml_path)
                continue
            coords = elem[0]
            points = coords.attrib['points']
            points = process_points(points)

            # save cropped images
            rect = cv.boundingRect(points)
            x,y,w,h = rect
            cropped = img[y:y+h, x:x+w].copy()

            line_image_path = f'{lines_folder + os.path.sep}im{img_id}.png'
            try:
                cv.imwrite(line_image_path, cropped)
            except:
                print(f'Something wrong with image: {img_path}')
                continue
            img_id += 1

            # get the label
            
            label = elem[2][0].text

            # add to csv_list
            csv_list.append([f'lines{os.path.sep}im{img_id-1}', label])
    r = random.random()
    if r < p:
        train_csv_list.extend(csv_list)
    else:
        val_csv_list.extend(csv_list)
print(f'number of train lines: {len(train_csv_list)}')
print(f'number of val lines: {len(val_csv_list)}')
#print(f'csv_list[:10]:\n{csv_list[:10]}')

print('write to csv...')
train_csv_file_path = '/content/data/train_data.csv'

with open(train_csv_file_path, 'w', newline='') as csv_file:
    data_writer = csv.writer(csv_file)
    data_writer.writerows(train_csv_list)

val_csv_file_path = '/content/data/val_data.csv'

with open(val_csv_file_path, 'w', newline='') as csv_file:
    data_writer = csv.writer(csv_file)
    data_writer.writerows(val_csv_list)
