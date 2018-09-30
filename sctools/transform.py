import cv2
import numpy as np
import math

def get_rotMapper(height, width, angle):
    cosA = math.cos(angle * math.pi / 180.0)
    sinA = math.sin(angle * math.pi / 180.0)

    if (angle >= 0):
        dx = 0
        dy = width * sinA
    else:
        dx = -height * sinA
        dy = 0

    rotMat = np.array([[cosA, sinA, dx],
                       [-sinA, cosA, dy]])

    rotBack = np.copy(rotMat)
    rotBack[1, 0], rotBack[0, 1] = rotBack[0, 1], rotBack[1, 0]
    rotBack[:, 2] = np.matmul(rotBack[:2, :2], -rotMat[:, 2])

    def mapper(xy, src2dst=True, rounding=True):
        xy = np.array(xy).reshape((2, -1))
        if src2dst:
            ret = np.matmul(rotMat[:2, :2], xy) + rotMat[:, 2].reshape((2, 1))
        else:
            ret = np.matmul(rotBack[:2, :2], xy) + rotBack[:, 2].reshape((2, 1))

        if rounding:
            ret = ret.astype(np.int32)
        return ret

    return rotMat, rotBack, mapper


def rotateImage(image, angle):
    # this function will return exactly same image when angle is zero
    height, width = image.shape[:2]

    rotMat, rotBack, mapper = get_rotMapper(height, width, angle)

    bound_w, bound_h = np.maximum(np.maximum(mapper((width, height)), mapper((width, 0))),
                                  np.maximum(mapper((0, 0)), mapper((0, height))))

    result = cv2.warpAffine(image, rotMat, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(110, 0, 30))
    return result, mapper


if 0:
    # debug rotateImage
    img = cv2.imread("/home/hddl/Pictures/Webcam/card/2018-08-27-121722.jpg")
    h, w = img.shape[:2]
    r, mapper = rotateImage(img, 20)

    print(r.shape, img.shape)
    # print("r == img ? {}".format((r == img).all()))

    pts0 = np.array([[102, 300]]).T
    pts1 = mapper(pts0)
    pts2 = mapper(pts1, False)
    print(pts0)
    print(pts1)
    print(pts2)
    pts = mapper(np.array([[0, 0], [w, 0], [w, h], [0, h]]).T).T.reshape((-1, 1, 2))
    print(pts)
    cv2.polylines(r, [pts], True, (0, 255, 255), 3)

    cv2.imshow("XXX", r)
    cv2.waitKey(0)
