from collections import namedtuple

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image

from .models.keras_yolo import yolo_eval, yolo_head, yolo_loss

Object = namedtuple('Object', ['class_name', 'top', 'left', 'bottom', 'right', 'score'])

class ObjectDetector:
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

    def detect(self, image):
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
