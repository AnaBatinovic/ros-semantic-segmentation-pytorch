#!/usr/bin/env python2

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from color_coding_dict import *
import time
from semantic_segmentation_ros.msg import SegmentationNameMask


class CreateMask:

    def __init__(self):
        # Instantiate CvBridge
        self.bridge = CvBridge()
        
        self.input_segm_sub = rospy.Subscriber("segmentation/color/image_raw", Image, self.image_callback, queue_size=1)
        self.mask_image_pub = rospy.Publisher("segmentation/mask/image_raw", Image, queue_size=1)
        self.label_image_pub = rospy.Publisher("segmentation/label/image_raw", Image, queue_size=1)
        self.segmentation_mask_pub = rospy.Publisher("segmentation/mask", SegmentationNameMask, queue_size=1)

        self.segmentation_name_mask = SegmentationNameMask()

    def image_callback(self, msg):
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            object_name, object_mask, label_image = self.get_mask_and_name(cv2_img)
            object_mask_img = self.bridge.cv2_to_imgmsg(object_mask, "bgr8")
            label_image = self.bridge.cv2_to_imgmsg(label_image, "bgr8")
        except CvBridgeError as e:
            print(e)
        else:
            self.mask_image_pub.publish(object_mask_img)
            self.label_image_pub.publish(label_image)
            # Fill out msg and publish
            self.segmentation_name_mask.name = object_name
            self.segmentation_name_mask.mask = object_mask_img            
            self.segmentation_mask_pub.publish(self.segmentation_name_mask)


    def get_mask_and_name(self, img):
        # Convert to hsv colorspace
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for name in color_dict.keys():
            # Get colors from color coding dict - object name, segmentation color
            r, g, b = color_dict[name]
            h, s, v = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            lower_bound = np.array([h, s, v])
            upper_bound = np.array([h, s, v])

            # Find the colors within the boundaries, mask with all objects
            mask_img = cv2.inRange(hsv_img, lower_bound, upper_bound)
            # Define kernel size, filter
            kernel = np.ones((10, 10), np.uint8)
            # Remove unnecessary noise from mask
            mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
            mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)

            # Segment only the detected region
            _, contours, hierarchy = cv2.findContours(mask_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create mask for each object and name, publish it
            for contour in contours:
                # Create a mask
                mask_img_object = np.zeros(hsv_img.shape, np.uint8)
                cv2.drawContours(mask_img_object, [contour], -1, (255, 255, 255), cv2.FILLED)
                # Compute the center of the contour to put the text
                M = cv2.moments(contour)
                
                if M["m00"] > 300:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                        
                    cv2.circle(hsv_img, (cX, cY), 3, (0, 0, 255), cv2.FILLED)
                    # Reaching pixels [y, x]
                    cv2.putText(hsv_img, "{}".format(name, cY, cX, 15),
                        (cX - 40, cY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                # Return name and mask to be published
                return name, mask_img_object, hsv_img, 

if __name__ == '__main__':
    rospy.init_node('create_mask')
    mask = CreateMask()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS create mask.")
    cv2.destroyAllWindows()
