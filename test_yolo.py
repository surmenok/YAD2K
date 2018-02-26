#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import argparse
import colorsys
import imghdr
import os
import random

from collections import namedtuple

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from yad2k.models.keras_yolo import yolo_eval, yolo_head, yolo_loss

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


Object = namedtuple('Object', ['class_name', 'top', 'left', 'bottom', 'right', 'score'])

class YoloModel:
    def __init__(self, model_path, anchors_path, classes_path, score_threshold, iou_threshold):
        self.class_names = self._load_class_names(classes_path)
        anchors = self._load_anchors(anchors_path)
        self.yolo_model = load_model(model_path)

        # Verify model, anchors, and classes are compatible
        num_classes = len(self.class_names)
        num_anchors = len(anchors)

        # TODO: Assumes dim ordering is channel last
        model_output_channels = self.yolo_model.layers[-1].output_shape[-1]
        assert model_output_channels == num_anchors * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes. ' \
            'Specify matching anchors and classes with --anchors_path and ' \
            '--classes_path flags.'

        # Check if model is fully convolutional, assuming channel last order.
        self.model_image_size = self.yolo_model.layers[0].input_shape[1:3]
        self.is_fixed_size = self.model_image_size != (None, None)

        # Generate output tensor targets for filtered bounding boxes.
        # TODO: Wrap these backend operations with Keras layers.
        yolo_outputs = yolo_head(self.yolo_model.output, anchors, len(self.class_names))
        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(
            yolo_outputs,
            self.input_image_shape,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold)

        self.session = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    def detect(self, image_filepath):
        image = Image.open(image_filepath)
        image_data = self._transform_image(image)

        out_boxes, out_scores, out_classes = self.session.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        objects = []

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            object = Object(predicted_class, top, left, bottom, right, score)
            objects.append(object)

        return objects

    def close(self):
        self.session.close()

    def _load_anchors(self, anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def _load_class_names(self, classes_path):
        with open(classes_path) as f:
            class_names = f.readlines()
        return [c.strip() for c in class_names]

    def _transform_image(self, image):
        if self.is_fixed_size:  # TODO: When resizing we can use minibatch input.
            resized_image = image.resize(
                tuple(reversed(self.model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        return image_data


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
