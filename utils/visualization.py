"""visualization.py

The BBoxVisualization class implements drawing of nice looking
bounding boxes based on object detection results.
"""


import numpy as np
import cv2


# Constants
ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def gen_colors(num_colors):
    """Generate different colors.

    # Arguments
      num_colors: total number of colors/classes.

    # Output
      bgrs: a list of (B, G, R) tuples which correspond to each of
            the colors/classes.
    """
    import random
    import colorsys

    hsvs = [[float(x) / num_colors, 1., 0.7] for x in range(num_colors)]
    random.seed(1234)
    random.shuffle(hsvs)
    rgbs = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), hsvs))
    bgrs = [(int(rgb[2] * 255), int(rgb[1] * 255),  int(rgb[0] * 255))
            for rgb in rgbs]
    return bgrs


def draw_boxed_text(img, text, topleft, color):
    """Draw a transluent boxed text in white, overlayed on top of a
    colored patch surrounded by a black border. FONT, TEXT_SCALE,
    TEXT_THICKNESS and ALPHA values are constants (fixed) as defined
    on top.

    # Arguments
      img: the input image as a numpy array.
      text: the text to be drawn.
      topleft: XY coordinate of the topleft corner of the boxed text.
      color: color of the patch, i.e. background of the text.

    # Output
      img: note the original image is modified inplace.
    """
    assert img.dtype == np.uint8
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 3
    size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2
    # the patch is used to draw boxed text
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    cv2.putText(patch, text, (margin+1, h-margin-2), FONT, TEXT_SCALE,
                WHITE, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
    cv2.rectangle(patch, (0, 0), (w-1, h-1), BLACK, thickness=1)
    w = min(w, img_w - topleft[0])  # clip overlay at image boundary
    h = min(h, img_h - topleft[1])
    # Overlay the boxed text onto region of interest (roi) in img
    roi = img[topleft[1]:topleft[1]+h, topleft[0]:topleft[0]+w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img


class BBoxVisualization():
    """BBoxVisualization class implements nice drawing of bounding boxes.

    # Arguments
      cls_dict: a dictionary used to translate class id to its name.
    """

    def __init__(self, cls_dict):
        self.cls_dict = cls_dict
        self.colors = gen_colors(len(cls_dict))

    def draw_bboxes(self, img, boxes, confs, clss):
        """Draw detected bounding boxes on the original image."""
        for bb, cf, cl in zip(boxes, confs, clss):
            cl = int(cl)
            # min = topLeft, max = bottomRight  
            if cl == 0:
                x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
                color = self.colors[cl]
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                txt_loc = (max(x_min+2, 0), max(y_min+2, 0))
                cls_name = self.cls_dict.get(cl, 'CLS{}'.format(cl))
                txt = '{} {:.2f}'.format(cls_name, cf)
                img = draw_boxed_text(img, txt, txt_loc, color)
        return img
   
    def draw_centers(self, img, bb_values):
        for bb in bb_values:
            cv2.circle(img, (int(bb[4]), int(bb[5])), radius=8, color=(255,255,0), thickness=-1)
        return img
    
    def draw_torso(self, img, bb_values):
        for bb in bb_values:
            cv2.rectangle(img, (int(bb[6]), int(bb[7])), (int(bb[8]), int(bb[9])), color=(255,255,0), thickness=2)
        return img

    def draw_distances(self, img, distances):
        for dd in distances:
            txt = "Distance: " + '{:0.2f}'.format(dd[0][2]) + ' meters'
            txt_loc = (int(dd[1]), int(dd[2]))
            color = (255,255,0)
            img = draw_boxed_text(img, txt, txt_loc, color)
        return img

    def draw_social_distance(self, img, social_distance, thresh):
      if(len(social_distance)>= 1):
        for i in social_distance:
            print(i[0])
            if(i[0] <= thresh):
              cv2.line(img,(int(i[1]), int(i[2])) , (int(i[3]), int(i[4])) , (0, 255, 255), 2)
              cv2.circle(img, (int(i[1]), int(i[2])), radius=4, color=(0, 255, 255), thickness=-1)
              cv2.circle(img, (int(i[3]), int(i[4])), radius=4, color=(0, 255, 255), thickness=-1)
        return img
    
    def draw_bird_eye(self, background_eye, dist, dist2, reprj_point, fig, ax, redline, img):
        dim_x = 4.95
        dim_y = 16.70

        dist_orig = 2.49
        alfa = 27.961 * np.pi / 180.
        
        ax.cla()
        for i in range(len(dist)):
                ax.set_xlim([0, dim_x])
                ax.set_ylim([0, dim_y])
                ax.scatter(redline[0], redline[1], c="r")
                ax.scatter(redline[2], redline[3], c="r")
                ax.scatter(redline[4], redline[5], c="r")
                ax.scatter(redline[6], redline[7], c="r")                
                #ax.scatter(float(dist[i][0][0] + dist_orig - (float(dist[i][0][2] * np.cos(alfa)) * np.sin(0.17/9.05)) ), float(dist[i][0][2] * np.cos(alfa)) )
                ax.scatter(float(dist[i][0][0] + dist_orig), float(dist[i][0][2]))
                #ax.scatter(float(dist[i][0][0] + 2.40), float(dist[i][0][2] * np.sin(np.arccos((3.35/float(dist[i][0][2])) * np.pi / 180.)) ))
                ax.grid(True)               
                ax.imshow(img, extent=[0, dim_x, 0, dim_y])               
                print("x", str(dist[i][0][0]))
                print("y", str(dist[i][0][1]))
                print("z", str(dist[i][0][2]))
        fig.canvas.draw()

                #draw_boxed_text(background_eye, "first", (200 + (int(dist[i][0][0] * 50)), int(dist[i][0][2] * 50)), (0, 255, 0))
                #cv2.circle(background_eye, (200 + (int(dist[i][0][0] * 50)), int(dist[i][0][2] * 50)), radius=8, color=(0, 255, 0), thickness=-1)
                #print(dist[i][0][2])
        
                #f = open("test_asse_x.csv", "a+")
                #f.write(str(dist[i][0][1]) + "," + str(dist[i][0][2]) + "\n")
               
                #f.close()

        for j in range(len(dist2)):
                draw_boxed_text(background_eye, "second", (200 + (int(dist2[j][0][0] * 50)), int(dist2[j][0][2] * 50)), (0, 255, 0))
                cv2.circle(background_eye, (200 + (int(dist2[j][0][0] * 50)), int(dist2[j][0][2] * 50)), radius=2, color=(255, 255, 0), thickness=-1)
                #f = open("test_asse_x.csv", "a+")
                #f.write(str(dist2[j][0][1]) + "," + str(dist2[j][0][2]) + "\n")
                #print("x", str(dist2[j][0][0]))
                #print("z", str(dist2[j][0][2]))
                #f.close()

        for k in range(len(reprj_point)):
                draw_boxed_text(background_eye, "rep", (200 + (int(reprj_point[0] * 50)), int(reprj_point[2] * 50)), (0, 255, 0))
                cv2.circle(background_eye, (200 + (int(reprj_point[0] * 50)), int(reprj_point[2] * 50)), radius=8, color=(0, 0, 0), thickness=-1)
        
        return background_eye
