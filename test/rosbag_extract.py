import sys

sys.path.append("/opt/ros/noetic/lib/python3.11/site-packages")
sys.path.append("/usr/lib/python3.11/site-packages")

import rospy
import rosbag
import tf2_py as tf2
import geometry_msgs
import std_msgs
import numpy as np
import tf_conversions
from tqdm import tqdm


def load_transform_stamped(msg_tf_raw):
    msg_tf = geometry_msgs.msg.TransformStamped()
    msg_tf.header = std_msgs.msg.Header()
    msg_tf.header.frame_id = msg_tf_raw.header.frame_id
    msg_tf.header.seq = msg_tf_raw.header.seq
    msg_tf.header.stamp = msg_tf_raw.header.stamp

    msg_tf.child_frame_id = msg_tf_raw.child_frame_id

    msg_tf.transform.translation.x = msg_tf_raw.transform.translation.x
    msg_tf.transform.translation.y = msg_tf_raw.transform.translation.y
    msg_tf.transform.translation.z = msg_tf_raw.transform.translation.z
    msg_tf.transform.rotation.x = msg_tf_raw.transform.rotation.x
    msg_tf.transform.rotation.y = msg_tf_raw.transform.rotation.y
    msg_tf.transform.rotation.z = msg_tf_raw.transform.rotation.z
    msg_tf.transform.rotation.w = msg_tf_raw.transform.rotation.w
    return msg_tf


def transform_to_matrix(msg_tf):
    position = (
        msg_tf.transform.translation.x,
        msg_tf.transform.translation.y,
        msg_tf.transform.translation.z,
    )
    quaternion = (
        msg_tf.transform.rotation.x,
        msg_tf.transform.rotation.y,
        msg_tf.transform.rotation.z,
        msg_tf.transform.rotation.w,
    )
    mat = tf_conversions.toMatrix(tf_conversions.fromTf((position, quaternion)))
    return mat[:2, [0, 1, 3]]


def main():
    # filename = 'clf_cbf_sdf_sep_28_2023-09-28-14-49-57.bag'
    filename = "long_succ4_till_no_path.bag"
    bag = rosbag.Bag(filename, "r")
    # list all topics
    type_and_topic_info_list = bag.get_type_and_topic_info()
    max_topic_path_len = max([len(topic_path) for topic_path in type_and_topic_info_list.topics])
    for topic in type_and_topic_info_list.topics:
        msg_type = type_and_topic_info_list.topics[topic].msg_type
        topic = " " * (max_topic_path_len - len(topic)) + topic
        print(topic, msg_type, sep="    ")

    # get transform
    tf_buffer = tf2.BufferCore(rospy.Duration(1000000000))
    messages = list(bag.read_messages(topics=["/tf", "/tf_static"]))
    for topic, msg, t in tqdm(messages, ncols=80, desc="Loading TF"):
        if topic == "/tf_static":
            for msg_tf_raw in msg.transforms:
                msg_tf = load_transform_stamped(msg_tf_raw)
                tf_buffer.set_transform_static(msg_tf, "default_authority")
        else:
            for msg_tf_raw in msg.transforms:
                msg_tf = load_transform_stamped(msg_tf_raw)
                tf_buffer.set_transform(msg_tf, "default_authority")
    del messages

    # get lidar data
    time_stamps = []
    lidar_angles = []
    lidar_ranges = []
    lidar_poses = []
    messages = list(bag.read_messages(topics=["/front/scan"]))
    for topic, msg, t in tqdm(messages, ncols=80, desc="Loading Lidar"):
        try:
            lidar_poses.append(transform_to_matrix(tf_buffer.lookup_transform_core("map", msg.header.frame_id, t)))
        except tf2.ExtrapolationException or tf2.LookupException:
            break
        time_stamps.append(t.to_time())
        lidar_angles.append(np.arange(msg.angle_min, msg.angle_max, msg.angle_increment))
        lidar_ranges.append(np.array(msg.ranges))
    del messages

    # save data
    # output_filename = filename[:-4] + ".npz"
    # np.savez(
    #     output_filename,
    #     time_stamps=time_stamps,
    #     lidar_angles=lidar_angles,
    #     lidar_ranges=lidar_ranges,
    #     lidar_poses=lidar_poses,
    # )

    csv_data = np.concatenate(
        [
            np.array(time_stamps).reshape(-1, 1),
            np.array(lidar_poses).reshape(-1, 6),
            np.array(lidar_angles),
            np.array(lidar_ranges),
        ],
        axis=1,
    )
    order = np.argsort(csv_data[:, 0])  # sort by time
    csv_data = np.ascontiguousarray(csv_data[order, :])
    np.savetxt(filename[:-4] + ".csv", csv_data, delimiter=",")
    with open("ros_bag.dat", "wb") as f:
        f.write(np.array(csv_data.shape).astype(np.int64).tobytes())
        f.write(csv_data.tobytes("F"))  # for Eigen default column major


if __name__ == "__main__":
    main()
