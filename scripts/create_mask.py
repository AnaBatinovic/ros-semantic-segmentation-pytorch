#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from color_coding_dict import *
import time
from semantic_segmentation_ros.msg import SegmentationNameMask, SegmentationNameMaskArray
import copy

class CreateMask:

    def __init__(self):
        # Instantiate CvBridge
        self.bridge = CvBridge()
        
        self.mask_image_pub = rospy.Publisher(
            "segmentation/mask/image_raw", Image, queue_size=1)
        self.label_image_pub = rospy.Publisher(
            "segmentation/label/image_raw", Image, queue_size=1)
        self.segmentation_mask_pub = rospy.Publisher(
            "segmentation/mask", SegmentationNameMaskArray, queue_size=1)
        self.input_segm_sub = rospy.Subscriber(
            "segmentation/color/image_raw", Image, self.image_callback, queue_size=1)

        self.segmentation_name_mask = SegmentationNameMask()
        self.segmentation_name_mask_array = SegmentationNameMaskArray()
    
    def image_callback(self, msg):
        self.segmentation_name_mask_array.masks = []
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            object_name_array, object_mask_array, label_image_array = self.get_mask_and_name(cv2_img)
            for i in range (0, len(object_mask_array)):
                object_mask_img = self.bridge.cv2_to_imgmsg(object_mask_array[i], "bgr8")
                label_image = self.bridge.cv2_to_imgmsg(label_image_array[i], "bgr8")
                self.label_image_pub.publish(label_image)
                self.mask_image_pub.publish(object_mask_img)
                # Fill out msg and publish
                self.segmentation_name_mask.name = object_name_array[i]
                self.segmentation_name_mask.mask = object_mask_img
                self.segmentation_name_mask_array.masks.append(copy.deepcopy(self.segmentation_name_mask))       
            self.segmentation_mask_pub.publish(self.segmentation_name_mask_array)
        except CvBridgeError as e:
            print(e)
    
    def get_mask_and_name(self, img):
        # Convert to hsv colorspace
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_img_object_array = []
        hsv_image_array = []
        name_array = []
        for name in color_dict.keys():
            # Get colors from color coding dict - object name, segmentation color
            r, g, b = color_dict[name]
            h, s, v = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            lower_bound = np.array([h, s, v])
            upper_bound = np.array([h, s, v])

            # Find the colors within the boundaries, mask with all objects
            mask_img = cv2.inRange(hsv_img, lower_bound, upper_bound)

            # Segment only the detected region
            contours, hierarchy = cv2.findContours(mask_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            
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
                
                    # cv2.imshow('image1',mask_img_object)
                    # cv2.waitKey(0)
                    mask_img_object_array.append(mask_img_object)
                    hsv_image_array.append(hsv_img)
                    name_array.append(name)                
        # Return name and mask to be published
        return name_array, mask_img_object_array, hsv_image_array

if __name__ == '__main__':
    rospy.init_node('create_mask')
    mask = CreateMask()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS create mask.")
    cv2.destroyAllWindows()
