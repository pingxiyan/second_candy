from easydict import EasyDict as edict
import xml.etree.cElementTree as ET
import os
import cv2

def object(name, x0, y0, x1, y1):
    obj = edict()
    obj.name = name
    obj.bndbox = edict()
    obj.bndbox.xmin = x0
    obj.bndbox.ymin = y0
    obj.bndbox.xmax = x1
    obj.bndbox.ymax = y1
    return obj


# {"name":xxx, "bndbox":{"xmin":0,"ymin":0,"xmax":0,"ymax":0}, ...}
# rotation_degree is the degree of rotaion needs to be made on image to get correct labeling
# the rotation is made so no raw image info is missing
def save(image_filename, objs, rotation_degree=0, img=None, folder="VOC2007",
                        xmlpath="Annotations", common_info={"difficult": "0"}):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = image_filename
    ET.SubElement(annotation, "rotation_degree").text = str(rotation_degree)

    if img is None:
        newimg = cv2.imread(image_filename)
        W = newimg.shape[1]
        H = newimg.shape[0]
        depth = newimg.shape[2]
    else:
        W = img.shape[1]
        H = img.shape[0]
        depth = img.shape[2]

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(W)
    ET.SubElement(size, "height").text = str(H)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(annotation, "segmented").text = str(0)

    for o in objs:
        obj = ET.SubElement(annotation, "object")

        for k, v in o.items():
            if isinstance(v, dict):
                subroot = ET.SubElement(obj, k)
                for k2, v2 in v.items():
                    ET.SubElement(subroot, str(k2)).text = str(v2)
            else:
                ET.SubElement(obj, str(k)).text = str(v)

        for k, v in common_info.items():
            ET.SubElement(obj, str(k)).text = str(v)

    tree = ET.ElementTree(annotation)

    filename = os.path.splitext(os.path.basename(image_filename))[0]
    xmlfilename = "{}/{}.xml".format(xmlpath, filename)

    tree.write(xmlfilename)


def load(image_filename, xmlpath="Annotations"):
    filename = os.path.splitext(os.path.basename(image_filename))[0]
    xmlfilename = "{}/{}.xml".format(xmlpath, filename)
    objs = []
    rotation_degree = 0

    try:
        root = ET.parse(xmlfilename).getroot()
    except IOError:
        # print("XML file {} doesn't exists!".format(xmlfilename))
        return objs, rotation_degree

    rot = root.findall('rotation_degree')
    if rot:
        rotation_degree = float(rot[0].text)
    else:
        print("XML file {} has no rotaion!".format(xmlfilename))

    for o in root.findall('object'):
        name = o.find('name').text

        x0 = o.find('bndbox/xmin').text
        x1 = o.find('bndbox/xmax').text
        y0 = o.find('bndbox/ymin').text
        y1 = o.find('bndbox/ymax').text

        objs.append(object(name, int(x0), int(y0), int(x1), int(y1)))

    return objs, rotation_degree


