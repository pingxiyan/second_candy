import numpy as np
import cv2
import glob
import tensorflow as tf
import math
from PIL import Image
from matplotlib import pyplot as plt

'''

CREATE TF RECORD:
python dataset_tools/create_tower_tf_record.py --data_dir=/dockerv0/data/voc/VOCdevkit/VOC2019/ --output_path=./data-tower/tower.record --label_map_path=./data/tower_label_map.pbtxt

TRAIN:
python model_main.py --pipeline_config_path=ssd_mobilenet_v1_tower/pipeline_300x300.config --model_dir=ssd_mobilenet_v1_tower/ --num_train_steps=50000 --num_eval_steps=2000 --alsologtostderr

EXPORT:
python ./export_inference_graph.py --input_type=image_tensor --pipeline_config_path=ssd_mobilenet_v1_tower/pipeline.config --trained_checkpoint_prefix ssd_mobilenet_v1_tower/model.ckpt-42571 --output_directory=./ssd_mobilenet_v1_tower/export


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

test_sources = '/dockerv0/data/voc/VOCdevkit/VOC2019/JPEGImages/*.jpg'
#test_sources = '/dockerv0/data/card/JPEGImages/*.jpg'
#test_sources = ''
PATH_TO_FROZEN_GRAPH = './ssd_mobilenet_v1_tower/export/frozen_inference_graph.pb'
#PATH_TO_FROZEN_GRAPH = '/dockerv0/tf-models/research/object_detection/ssd_mobilenet_v1_hanzi/128x128/export/frozen_inference_graph.pb'
#PATH_TO_LABELS='/dockerv0/tf-models/research/object_detection/data/hanzi_label_map.pbtxt'

# Load fronzen graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

def rotateImage(image, angle):

    height, width = image.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    result = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(110,0,30))
    return result

def process_one_image(image_np_raw):

    def infer(image):
        image_np = image[:,:,[2,1,0]]

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, sess)
        return output_dict

    def draw_output(image, output_dict):
        for i in range(output_dict['detection_boxes'].shape[0]):
            y0, x0, y1, x1 = output_dict['detection_boxes'][i]
            conf = float(output_dict['detection_scores'][i])
            if conf < 0.1: continue

            cls = int(output_dict['detection_classes'][i])
            y0 *= image.shape[0]
            x0 *= image.shape[1]
            y1 *= image.shape[0]
            x1 *= image.shape[1]

            cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 3);

            names = ['c', 'r', 'l']
            cv2.putText(image, "{}:{:.2%}".format(names[cls], float(conf)),
                        (int(x0), int(y0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return

    output_dict = infer(image_np_raw)
    rotates = []
    rotates_confs = []
    image_np_orig = np.copy(image_np_raw)
    draw_output(image_np_orig, output_dict)

    # the result boxes is scaled isotropically (scaleX == scaleY == 300/640)
    for i in range(output_dict['detection_boxes'].shape[0]):
        y0,x0,y1,x1 = output_dict['detection_boxes'][i]
        conf = float(output_dict['detection_scores'][i])
        if conf < 0.1: continue

        cls = int(output_dict['detection_classes'][i])
        y0 *= image_np.shape[0]
        x0 *= image_np.shape[1]
        y1 *= image_np.shape[0]
        x1 *= image_np.shape[1]
        
        rot = math.atan2(x1-x0, y1-y0)
        if cls == 2:
            # bottom to right
            rot = -rot
        rotates.append(rot)
        rotates_confs.append(conf)

    cv2.imshow("image_np_orig", image_np_orig)
    cv2.imshow("image_np_rotated",image_np_orig)

    # rotation degree
    if len(rotates_confs) > 0:
        print(rotates_confs)
        r = sum([r*c for r,c in zip(rotates, rotates_confs)]) / sum(rotates_confs)
        r = r * 180 / math.pi
        print(r)

        if abs(r) > 20:
            image_np_rotated = rotateImage(image_np_raw, r)
            output_dict_rotated = infer(image_np_rotated)
            draw_output(image_np_rotated, output_dict_rotated)
            cv2.imshow("image_np_rotated", image_np_rotated)
    #

    key = cv2.waitKey(0)
    if((key & 0xFF)== ord('q') and (key>0)):
        return False
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
            for image_path in glob.glob(test_sources):
                image_np = cv2.imread(image_path)
                if not process_one_image(image_np): break
            

