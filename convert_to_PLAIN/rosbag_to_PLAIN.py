#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extract images from a rosbag.
"""

import os
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

import rosbag
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

def main(output_dir, bag_file, downsample_rate, pose_topic, rgb_topic, depth_topic):
    """Extract a folder of images from a rosbag.
    """
    topic_list = [rgb_topic, depth_topic, pose_topic]

    os.makedirs(output_dir, exist_ok=True)

    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()

    extracted_dict = {}

    for topic in topic_list:
        extracted_dict[topic] = []
    
    pose_timestamps = []

    for topic, msg, t in bag.read_messages():
        if topic in topic_list:
            if topic.endswith('odom'):
                extracted_dict[topic].append(msg)
                pose_timestamps.append(t.to_sec())
            elif topic.endswith('image_raw'):
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                extracted_dict[topic].append((cv_img, t.to_sec()))
            else:
                raise ValueError('Topic not recognized: {}'.format(topic))

    bag.close()

    pose_timestamps = np.array(pose_timestamps)

    # Pair RGB and Depth images by timestamp
    timestamp_max_diff = 0.02
    rgb_depth_pairs = []

    for rgb in extracted_dict['/rgb/image_raw']:
        for depth in extracted_dict['/depth_to_rgb/image_raw']:
            if abs(rgb[1] - depth[1]) < timestamp_max_diff:
                rgb_depth_pairs.append((rgb[0], depth[0], np.mean([rgb[1], depth[1]])))
                print('paired rgb and depth. timestamp diff: {}'.format(abs(rgb[1] - depth[1])))
                break

    print('paired {} rgb and depth images'.format(len(rgb_depth_pairs)))

    posed_RGBD_list = []

    for rgb, depth, timestamp in rgb_depth_pairs:
        pose_timestamp_diff_list = np.abs(pose_timestamps - timestamp)
        selected_pose_idx = np.argmin(pose_timestamp_diff_list)
        selected_pose = extracted_dict[pose_topic][selected_pose_idx]
        print("Selected pose w/ timestamp diff: {}".format(pose_timestamp_diff_list[selected_pose_idx]))
        posed_RGBD_list.append((rgb, depth, selected_pose))
    
    # Save RGBD images and poses.

    # Save RGB image to args.output_dir/color as jpg
    # Save depth image to args.output_dir/depth as png
    # Save pose to args.output_dir/pose as txt

    posed_RGBD_list = posed_RGBD_list[::downsample_rate]

    for i, (rgb, depth, pose) in enumerate(posed_RGBD_list):
        rgb_filename = os.path.join(output_dir, 'color', '{:06d}.jpg'.format(i))
        depth_filename = os.path.join(output_dir, 'depth', '{:06d}.png'.format(i))
        pose_filename = os.path.join(output_dir, 'pose', '{:06d}.txt'.format(i))

        os.makedirs(os.path.dirname(rgb_filename), exist_ok=True)
        os.makedirs(os.path.dirname(depth_filename), exist_ok=True)
        os.makedirs(os.path.dirname(pose_filename), exist_ok=True)

        cv2.imwrite(rgb_filename, rgb)
        cv2.imwrite(depth_filename, depth)
        with open(pose_filename, 'w') as f:
            # Save pose as 4x4 matrix
            translation = np.array([
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z
            ])

            orientation = np.array([
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w
            ])

            pose_matrix = np.eye(4)
            pose_matrix[:3, 3] = translation
            pose_matrix[:3, :3] = R.from_quat(orientation).as_matrix()

            np.savetxt(f, pose_matrix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--bag_file", help="Input ROS bag.")
    parser.add_argument("--output_dir", help="Output directory.")
    parser.add_argument("--downsample_rate", type=int, default=1, help="Downsample rate.")
    parser.add_argument("--pose_topic", default='/map_odom', help="Name of ROS Topic for pose.")
    parser.add_argument("--rgb_topic", default='/rgb/image_raw', help="Name of ROS Topic for RGB image.")
    parser.add_argument("--depth_topic", default='/depth_to_rgb/image_raw', help="Name of ROS Topic for depth image.")
    args = parser.parse_args()
    # TODO(roger): support saving intrinsics as well.
    main(args.output_dir,
         args.bag_file,
         args.downsample_rate,
         args.pose_topic,
         args.rgb_topic,
         args.depth_topic)
