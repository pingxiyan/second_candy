import numpy as np
import cv2
import glob
import tensorflow as tf
import math
import os
from PIL import Image
from matplotlib import pyplot as plt
import copy
import sys
sys.path.append('../train_data')
import sctools

'''

CREATE TF RECORD:
python dataset_tools/create_tower_tf_record.py --data_dir=/dockerv0/data/voc/VOCdevkit/VOC2019/ --output_path=./data-tower/tower.record --label_map_path=./data/tower_label_map.pbtxt

TRAIN:
python model_main.py --pipeline_config_path=ssd_mobilenet_v1_tower/pipeline_300x300.config --model_dir=ssd_mobilenet_v1_tower/   --num_train_steps=50000  --num_eval_steps=2000 --alsologtostderr
python model_main.py --pipeline_config_path=ssd_mobilenet_v1_tower/pipeline_300x300.config --model_dir=ssd_mobilenet_v1_tower/r3 --num_train_steps=500000 --num_eval_steps=2000 --alsologtostderr

EXPORT:
python ./export_inference_graph.py --input_type=image_tensor --pipeline_config_path=ssd_mobilenet_v1_tower/r3/pipeline_300x300.config --trained_checkpoint_prefix ssd_mobilenet_v1_tower/r3/model.ckpt-140079 --output_directory=./ssd_mobilenet_v1_tower/r3/export

'''


#if tf.__version__ < '1.4.0':
#    raise ImportError('Please upgrade your tensorflow installation {} to v1.4.* or later!'.format(tf.__version__))

#from utils import label_map_util
#from utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, sess):
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks', 'Shape']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
              
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
        
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    output_dict['imageshape'] = image.shape
    
    #LOGDIR='/dockerv0/tblogdir/1/'
    #train_writer = tf.summary.FileWriter(LOGDIR)
    #train_writer.add_graph(sess.graph)
    return output_dict

rootpath = '/home/hddl'
test_sources = rootpath+'/dockerv0/data/voc/VOCdevkit/VOC2019/JPEGImages/*.jpg'
test_sources = rootpath+'/dockerv0/data/voc/VOCdevkit/tower/JPEGImages/*.jpg'

test_sources = []
for xmlname in glob.glob(rootpath+'/dockerv0/data/voc/VOCdevkit/tower/Annotations/*.xml'):
    basename = os.path.splitext(os.path.basename(xmlname))[0]
    test_sources.append((rootpath+"/dockerv0/data/voc/VOCdevkit/tower/JPEGImages/{}.jpg").format(basename))

PATH_TO_FROZEN_GRAPH = rootpath+'/dockerv0/second_candy/tf-models/research/object_detection/ssd_mobilenet_v1_tower/r8/export/frozen_inference_graph.pb'
LABEL_NAMES = ['bg', 'c', 'cR', 'cL']

# Load fronzen graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

def infer(image):
    # tensorflow use RGB order while OpenCV has BGR order
    image_np = image[:,:,[2,1,0]]

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, sess)
    return output_dict

def draw_output(image, output_dict, classfilter = None, mapper = None):
    for i in range(output_dict['detection_boxes'].shape[0]):
        y0, x0, y1, x1 = output_dict['detection_boxes'][i]
        conf = float(output_dict['detection_scores'][i])
        if conf < 0.01: continue
        aspect_ratio = float(x1 - x0) / float(y1 - y0)

        cls = int(output_dict['detection_classes'][i])

        if classfilter and (LABEL_NAMES[cls] not in classfilter): continue

        tag = "{} {:.2%}".format(LABEL_NAMES[cls], float(conf))

        # use mapper to get polygon
        pts0 = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).T
        ptsB = mapper(pts0)
        pts = ptsB.T.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (90, 255, 90), 3)
        cv2.putText(image, tag, (int(pts[0,0,0]), int(pts[0,0,1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
    return

def process_one_image(image_np_raw_input):

    preRotate = 0
    while(1):
        if preRotate:
            image_np_raw, _ = sctools.transform.rotateImage(image_np_raw_input, preRotate)
        else:
            image_np_raw = image_np_raw_input

        output_dict = infer(image_np_raw)

        image_np_orig = np.copy(image_np_raw)

        def mapper_no_rotate(xy):
            # input is 2XN
            for i in range(xy.shape[1]):
                xy[1, i] *= image_np_orig.shape[0]  # y
                xy[0, i] *= image_np_orig.shape[1]  # x
            return xy.astype(np.int32)

        draw_output(image_np_orig, output_dict, mapper = mapper_no_rotate)

        rotates = []
        rotates_confs = []
        # the result boxes is scaled isotropically (scaleX == scaleY == 300/640)
        for i in range(output_dict['detection_boxes'].shape[0]):
            y0,x0,y1,x1 = output_dict['detection_boxes'][i]
            conf = float(output_dict['detection_scores'][i])
            if conf < 0.01: continue

            cls = int(output_dict['detection_classes'][i])
            x0 *= image_np_orig.shape[1]
            x1 *= image_np_orig.shape[1]
            y0 *= image_np_orig.shape[0]
            y1 *= image_np_orig.shape[0]

            aspect_ratio = float(x1-x0)/float(y1-y0)
            rot = math.atan2(x1-x0, y1-y0) * 180 / math.pi

            # bottom to right
            if cls == 2: rot = -rot
            rot_raw = rot

            if cls == 1: rot = 0

            # detect vertical view angle
            #if aspect_ratio < 0.25:
            #    rot = 0

            print("cls={} conf={} rot_raw={} rot={} aspect_ratio={}".format(cls, conf, rot_raw, rot, aspect_ratio))

            rotates.append(rot)
            rotates_confs.append(conf)

        cv2.imshow("image_np_orig", image_np_orig)

        image_np_final = image_np_orig

        # rotation degree
        if len(rotates_confs) > 0:

            # find the most possible rotation angle
            imax = 0
            for irc, rconf in enumerate(rotates_confs):
                if rconf > rotates_confs[imax]:
                    imax = irc

            r = rotates[imax]

            # if we need rotate, do it and infer again
            if r != 0:
                print("rotate {:.2f} degree".format(r))

                image_np_rotated, mapper = sctools.transform.rotateImage(image_np_raw, r)
                image_np_rotated2 = cv2.resize(image_np_rotated, (300, 300))

                output_dict_rotated = infer(image_np_rotated2)
                def mapper_rotate_back(xy):
                    for i in range(xy.shape[1]):
                        xy[0, i] *= image_np_rotated.shape[1]
                        xy[1, i] *= image_np_rotated.shape[0]
                    return mapper(xy, src2dst = False)

                cv2.imshow("image_np_rotated", image_np_rotated2)

                # replace the final result
                image_np_final = np.copy(image_np_raw)
                draw_output(image_np_final, output_dict_rotated, classfilter=["c"], mapper = mapper_rotate_back)

        cv2.imshow("image_np_final", image_np_final)

        key = cv2.waitKey(0) & 0xFF
        if(key== ord('q')):
            return False
        if(key== ord(' ')):
            break

        if (key== ord('[')): preRotate -= 1
        if (key == ord(']')): preRotate += 1
        preRotate = max(min(preRotate, 90), -90)

    return True



with detection_graph.as_default():
    with tf.Session() as sess:
        # infer
        if not test_sources:
            cap = cv2.VideoCapture(0)
            while(1):
                ret, image_np = cap.read()
                assert(ret)
                if not process_one_image(image_np): break
        else:
            for image_path in test_sources:
                print("===== {} =====".format(image_path))
                image_np = cv2.imread(image_path)
                if not process_one_image(image_np): break
            

