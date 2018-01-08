#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    pcl_msg_new = ros_to_pcl(pcl_msg)

    # TODO: Voxel Grid Downsampling
    # voxel is short for volume element, so you can divide your 3D point cloud into a regular 3D grid of volume elements creating a voxel grid
    # we want to downsample the data to improve computation time so we take a spatial average of the points confined by each voxel
    # the set of points which lie in the bounds of a voxel are assigned to that voxel and statisically combined into one output point

    # Create a VoxelGrid filter object for our input point cloud
    vox = pcl_msg_new.make_voxel_grid_filter()
 
    # Leaf size (voxel size in meters)
    LEAF_SIZE = 0.01

    # set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()
    filename = 'voxel_downsampled.pcd'
    pcl.save(cloud_filtered, filename)

    # TODO: PassThrough Filter
    # if you have prior info about the location of your target in the scene, you can apply a pass through filter to remove useless data from your point cloud
    # Create a PassThrough filter object
    passthrough = cloud_filtered.make_passthrough_filter()

    # specify cut-off values along the z-axis to indicate where to crop the scene. The region you allow to pass through is called the region of interest
    # here we crop out up to the bottom of the objects so table and everything below is removed
    # Assign axis and range to the passthrough filter object, values in meters
    filter_axis = 'z'
    passthrough.set_filter_field_name (filter_axis)
    axis_min = 0.757
    axis_max = 1.1
    passthrough.set_filter_limits (axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()
    filename = 'pass_through_filtered.pcd'
    pcl.save(cloud_filtered, filename)


    # TODO: RANSAC Plane Segmentation
    # RANSAC is an algorithm used to identify points in your dataset that belong to a particular model (ex plane, cylinder, box, or any other common shape)
    # RANSAC assumes that all data in a dataset is composed of both inliers and outliers, where inliers can be defined by a particular model with a specific set of parameters, and outliers do not fit that model and can be discared.
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()
 
    # we model the table as a plane, so we can remove it from the point cloud using RANSAC
    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)


    # TODO: Extract inliers and outliers
    # Call the segment function to obtain set of inlier indices and model coefficients
    # inliers refers to the table
    inliers, coefficients = seg.segment()

    # TODO: Extract table and objects
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    filename = 'cloud_table.pcd'
    pcl.save(cloud_table, filename)

    # use the negative flag to extract the objects rather than the table
    cloud_objects = cloud_filtered.extract(inliers, negative=True)
    filename = 'cloud_objects.pcd'
    pcl.save(cloud_objects, filename)


    # TODO: Euclidean Clustering
    # use color and/or distance information to perform further segmentation. If we rely only on RANSAC, you can produce false positives since it only relies on shape. A soda can looks similar to a beer can so it can mix these two objects up if only looking at shape. Plus you would have to run through the entire point cloud several times for each model shape, which is not optimal
    # Euclidean creates clusters by grouping data points that are within some threshold distance from the nearest other point in the data
    # coordinates don't need to be defined by position but can be defined in color space or any other feature

    # PCL's Eucliean Clustering algorithm requires point cloud data with only spatial information so convert from color/space to just space
    white_cloud = XYZRGB_to_XYZ(cloud_objects)

    # construct a k-d tree to decrease the computational burden of searching for neighboring points
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()

    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(175)
    ec.set_MaxClusterSize(1200)

    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)

    # Extract indices for each of the discovered clusters
    # cluster_indices contains a list of indices for each cluster (a list of lists)
    cluster_indices = ec.Extract()


    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Assign an individual color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []
 
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])
 
    # Create new cloud containing all clusters, each with unique color
    # cluster_cloud is the final point cloud which contains points for each of the segmented objects, with each set of points having a unique color. The cluser_cloud is of type PointCloud_PointXYZRGB
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages (of type PointCloud2)
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)


    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    # Exercise-3 TODOs: 

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)

        # Convert the cluster from pcl to ROS using helper function
        ros_pcl_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # Compute the associated feature vector
        chists = compute_color_histograms(ros_pcl_cluster, using_hsv=True)
        normals = get_normals(ros_pcl_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
#        object_markers_pub.publish(label, label_pos, index)

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_pcl_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
#    detected_objects_pub.publish(detected_objects)
    detected_objects_pub = rospy.Publisher("/detected_objects", Marker, queue_size=1)

if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    object_markers_pub = rospy.Publisher("/object_markers", PointCloud2, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", PointCloud2, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()

