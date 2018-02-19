"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
import argparse

import os

import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)

# Args
argparser = argparse.ArgumentParser(
    description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

argparser.add_argument(
    '-i',
    '--images_path',
    help="path to a folder containing images for training and validation",
    default=os.path.join('..', 'data', 'udacity-object-dataset'))

argparser.add_argument(
    '-t',
    '--train_path',
    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images' for the training set",
    default=os.path.join('..', 'data', 'udacity-object-dataset', 'train.npz'))

argparser.add_argument(
    '-v',
    '--val_path',
    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images' for the validation set",
    default=os.path.join('..', 'data', 'udacity-object-dataset', 'val.npz'))

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default=os.path.join('model_data', 'yolo_anchors.txt'))

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default=os.path.join('..', 'DATA', 'underwater_classes.txt'))

# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

def _main(args):
    images_path = os.path.expanduser(args.images_path)
    train_path = os.path.expanduser(args.train_path)
    val_path = os.path.expanduser(args.val_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)

    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)

    model_body, model = create_model(anchors, class_names)

    train(
        model,
        class_names,
        anchors,
        train_path,
        val_path,
        images_path
    )
    
    weights_name='trained_stage_3_best.h5'
    model_name='trained_model.h5'
    model_body.load_weights(weights_name)
    model_body.save(model_name)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def process_data(images, boxes=None):
    '''processes the data'''
    images = [PIL.Image.fromarray(i) for i in images]
    orig_size = np.array([images[0].width, images[0].height])
    orig_size = np.expand_dims(orig_size, axis=0)

    # Image preprocessing.
    processed_images = [i.resize((416, 416), PIL.Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        boxes = [box.reshape((-1, 5)) for box in boxes]
        # Get extents as y_min, x_min, y_max, x_max, class for comparision with
        # model output.
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
        boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0]  < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))

        return np.array(processed_images), np.array(boxes)
    else:
        return np.array(processed_images)

def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)

def create_model(anchors, class_names, load_pretrained=True, freeze_body=True):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model


def batch_generator(image_filenames, boxes, images_path, anchors, batch_size):
    index = 0
    
    while True:
        batch_images = []
        batch_boxes = []
        
        for i in range(batch_size):
            filename = image_filenames[index]
            image = Image.open(os.path.join(images_path, filename))
            image = np.array(image, dtype=np.uint8)
            batch_images.append(image)
            
            box = boxes[index]
            batch_boxes.append(box)
            
            index = (index + 1) % len(image_filenames)
            
        batch_images, batch_boxes = process_data(batch_images, batch_boxes)
        detectors_mask, matching_true_boxes = get_detector_mask(batch_boxes, anchors)
        
        yield [batch_images, batch_boxes, detectors_mask, matching_true_boxes], np.zeros(len(batch_images))


def train(model, class_names, anchors, train_path, val_path, images_path):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''
    
    workers = 10
    
    train_data = np.load(train_path)
    train_image_filenames, train_boxes = train_data['images'], train_data['boxes']
    
    val_data = np.load(val_path)
    val_image_filenames, val_boxes = val_data['images'], val_data['boxes']

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    logging = TensorBoard()
    checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    batch_size = 32
    
    train_batches = batch_generator(train_image_filenames, train_boxes, images_path, anchors, batch_size)
    train_steps_per_epoch = len(train_image_filenames) // batch_size
    
    val_batches = batch_generator(val_image_filenames, val_boxes, images_path, anchors, batch_size)
    val_steps_per_epoch = len(val_image_filenames) // batch_size
    
    model.fit_generator(train_batches,
              validation_data=val_batches,
              epochs=5,
              steps_per_epoch=train_steps_per_epoch,
              validation_steps=val_steps_per_epoch,
              workers=workers,
              callbacks=[logging])
    model.save_weights('trained_stage_1.h5')

    model_body, model = create_model(anchors, class_names, load_pretrained=False, freeze_body=False)

    model.load_weights('trained_stage_1.h5')

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    batch_size = 8
    
    train_batches = batch_generator(train_image_filenames, train_boxes, images_path, anchors, batch_size)
    train_steps_per_epoch = len(train_image_filenames) // batch_size
    
    val_batches = batch_generator(val_image_filenames, val_boxes, images_path, anchors, batch_size)
    val_steps_per_epoch = len(val_image_filenames) // batch_size
    
    model.fit_generator(train_batches,
              validation_data=val_batches,
              epochs=30,
              steps_per_epoch=train_steps_per_epoch,
              validation_steps=val_steps_per_epoch,
              workers=workers,
              callbacks=[logging])

    model.fit_generator(train_batches,
              validation_data=val_batches,
              epochs=30,
              steps_per_epoch=train_steps_per_epoch,
              validation_steps=val_steps_per_epoch,
              workers=workers,
              callbacks=[logging, checkpoint, early_stopping])


if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
