#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import argparse
import colorsys
import imghdr
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .yad2k import ObjectDetector

parser = argparse.ArgumentParser(
    description='Run a YOLO_v2 style detection model on test images..')
parser.add_argument(
    'model_path',
    help='path to h5 model file containing body'
    'of a YOLO_v2 model')
parser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default='model_data/yolo_anchors.txt')
parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to coco_classes.txt',
    default='model_data/coco_classes.txt')
parser.add_argument(
    '-t',
    '--test_path',
    help='path to directory of test images, defaults to images/',
    default='images')
parser.add_argument(
    '-o',
    '--output_path',
    help='path to output test images, defaults to images/out',
    default='images/out')
parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .3',
    default=.3)
parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .5',
    default=.5)


def _generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    colors = {class_names[i]: v for i, v in enumerate(colors)}
    return colors


def _main(args):
    model_path = os.path.expanduser(args.model_path)
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)
    test_path = os.path.expanduser(args.test_path)
    output_path = os.path.expanduser(args.output_path)

    if not os.path.exists(output_path):
        print('Creating output path {}'.format(output_path))
        os.mkdir(output_path)

    model = YoloModel(model_path, anchors_path, classes_path, args.score_threshold, args.iou_threshold)
    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Generate colors for drawing bounding boxes.
    colors = _generate_colors(model.class_names)

    for image_file in os.listdir(test_path):
        image_filepath = os.path.join(test_path, image_file)
        try:
            image_type = imghdr.what(image_filepath)
            if not image_type:
                continue
        except IsADirectoryError:
            continue

        objects = model.detect(image_filepath)
        print('Found {} objects for {}'.format(len(objects), image_file))

        image = Image.open(image_filepath)

        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, object in reversed(list(enumerate(objects))):
            label = '{} {:.2f}'.format(object.class_name, object.score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            print(label, (object.left, object.top), (object.right, object.bottom))

            if object.top - label_size[1] >= 0:
                text_origin = np.array([object.left, object.top - label_size[1]])
            else:
                text_origin = np.array([object.left, object.top + 1])

            color = colors[object.class_name]

            # My kingdom for a good redistributable image drawing library.
            for j in range(thickness):
                draw.rectangle(
                    [object.left + j, object.top + j, object.right - j, object.bottom - j],
                    outline=color)

            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=color)
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        image.save(os.path.join(output_path, image_file), quality=90)

    model.close()


if __name__ == '__main__':
    _main(parser.parse_args())
