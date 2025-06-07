#include "erl_common/eigen.hpp"
#include "erl_common/yaml.hpp"
#include "erl_geometry/abstract_occupancy_octree.hpp"
#include "erl_geometry/abstract_occupancy_quadtree.hpp"
#include "erl_geometry/ros_msgs/occupancy_tree_msg.hpp"
#include "erl_sdf_mapping/bayesian_hilbert_surface_mapping.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping.hpp"
#include "erl_sdf_mapping/SaveMap.h"
#include "erl_sdf_mapping/SdfQuery.h"

#include <geometry_msgs/Vector3.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <rviz/default_plugin/point_cloud_transformers.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Temperature.h>
#include <tf/transform_datatypes.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>

using namespace erl::common;
using namespace erl::sdf_mapping;

struct SdfMappingNodeSetting : Yamlable<SdfMappingNodeSetting> {
    // setting class for the surface mapping. For example, to use
    // erl::sdf_mapping::GpOccSurfaceMapping<float, 2>, you should use its setting class
    // erl::sdf_mapping::GpOccSurfaceMapping<float, 2>::Setting.
    std::string surface_mapping_setting_type = "";
    // path to the yaml file for the surface mapping setting
    std::string surface_mapping_setting_file = "";
    // setting class for the SDF mapping. For example, to use single float precision in 2D,
    // you should use erl::sdf_mapping::GpSdfMappingSetting<float, 2>.
    std::string sdf_mapping_setting_type = "";
    // path to the yaml file for the SDF mapping setting
    std::string sdf_mapping_setting_file = "";
    // type of the sdf mapping. For example, to use single float precision in 2D, use
    // GpOccSurfaceMapping for the surface mapping, you should use
    // erl::sdf_mapping::GpSdfMapping<float, 2, erl::sdf_mapping::GpOccSurfaceMapping<float, 2>>.
    std::string sdf_mapping_type = "";
    // whether to use the odometry topic to get the sensor pose
    bool use_odom = false;
    // name of the odometry topic
    std::string odom_topic = "/jackal_velocity_controller/odom";
    // can be "nav_msgs::Odometry" or "geometry_msgs::TransformStamped"
    std::string odom_msg_type = "nav_msgs::Odometry";
    // size of the odometry queue
    int odom_queue_size = 100;
    // name of the world frame
    std::string world_frame = "map";
    // name of the sensor frame
    std::string sensor_frame = "front_laser";
    // name of the scan topic
    std::string scan_topic = "/front/scan";
    // type of the scan: `sensor_msgs::LaserScan`, `sensor_msgs::PointCloud2`, or
    // `sensor_msgs::Image`.
    std::string scan_type = "sensor_msgs::LaserScan";
    // if the scan data is in the local frame, set this to true.
    bool scan_in_local_frame = false;
    // scale for depth image, 0.001 converts mm to m.
    float depth_scale = 0.001f;
    // if true, publish the occupancy tree used by the surface mapping.
    bool publish_tree = false;
    // name of the topic to publish the occupancy tree
    std::string publish_tree_topic = "surface_mapping_tree";
    // if true, use binary format to publish the occupancy tree, which makes the message smaller.
    bool publish_tree_binary = true;
    // frequency to publish the occupancy tree
    double publish_tree_frequency = 5.0;
    // if true, publish the surface points used by the sdf mapping.
    bool publish_surface_points = false;
    // name of the topic to publish the surface points
    std::string publish_surface_points_topic = "surface_points";
    // frequency to publish the surface points
    double publish_surface_points_frequency = 5.0;
};

class SdfMappingNode {
    enum class ScanType {
        Laser = 0,
        PointCloud = 1,
        Depth = 2,
    };

    SdfMappingNodeSetting m_setting_;
    ScanType m_scan_type_ = ScanType::Laser;
    ros::NodeHandle m_nh_;
    ros::Subscriber m_sub_odom_, m_sub_scan_;
    ros::ServiceServer m_srv_query_sdf_, m_srv_save_map_;
    ros::Publisher m_pub_tree_;
    ros::Publisher m_pub_surface_points_;
    ros::Publisher m_pub_update_time_;
    ros::Publisher m_pub_query_time_;
    ros::Timer m_pub_tree_timer_;
    ros::Timer m_pub_surface_points_timer_;
    erl_geometry::OccupancyTreeMsg m_msg_tree_;
    sensor_msgs::PointCloud2 m_msg_surface_points_;
    sensor_msgs::Temperature m_msg_update_time_;
    sensor_msgs::Temperature m_msg_query_time_;

    std::shared_ptr<YamlableBase> m_surface_mapping_cfg_ = nullptr;
    std::shared_ptr<YamlableBase> m_sdf_mapping_cfg_ = nullptr;
    std::shared_ptr<AbstractGpSdfMapping> m_sdf_mapping_ = nullptr;
    std::shared_ptr<const void> m_tree_ = nullptr;  // used to store the occupancy tree
    bool m_tree_is_2d_ = false;                     // is the occupancy tree 2D?
    bool m_tree_is_double_ = false;                 // is the occupancy tree double precision?

    // for the sensor pose

    std::mutex m_odom_queue_lock_;
    std::vector<geometry_msgs::TransformStamped> m_odom_queue_{};
    int m_odom_queue_head_ = -1;
    tf2_ros::Buffer m_tf_buffer_;
    tf2_ros::TransformListener m_tf_listener_{m_tf_buffer_};

    // for the scan data

    sensor_msgs::LaserScan::ConstPtr m_lidar_scan_2d_ = nullptr;
    sensor_msgs::PointCloud2::ConstPtr m_lidar_scan_3d_ = nullptr;
    sensor_msgs::Image::ConstPtr m_depth_image_ = nullptr;

public:
    SdfMappingNode(ros::NodeHandle& nh)
        : m_nh_(nh) {
        if (!LoadParameters()) {
            ERL_FATAL("Failed to load parameters");
            ros::shutdown();
            return;
        }
        auto& setting_factory = YamlableBase::Factory::GetInstance();
        // load the surface mapping config
        m_surface_mapping_cfg_ = setting_factory.Create(m_setting_.surface_mapping_setting_type);
        if (!m_surface_mapping_cfg_) {
            ERL_FATAL("Failed to create surface mapping config");
            ros::shutdown();
            return;
        }
        if (!m_surface_mapping_cfg_->FromYamlFile(m_setting_.surface_mapping_setting_file)) {
            ERL_FATAL("Failed to load {}", m_setting_.surface_mapping_setting_file);
            ros::shutdown();
            return;
        }
        m_sdf_mapping_cfg_ = setting_factory.Create(m_setting_.sdf_mapping_setting_type);
        if (!m_sdf_mapping_cfg_) {
            ERL_FATAL("Failed to create SDF mapping config");
            ros::shutdown();
            return;
        }
        if (!m_sdf_mapping_cfg_->FromYamlFile(m_setting_.sdf_mapping_setting_file)) {
            ERL_FATAL("Failed to load {}", m_setting_.sdf_mapping_setting_file);
            ros::shutdown();
            return;
        }
        m_sdf_mapping_ = AbstractGpSdfMapping::Create(
            m_setting_.sdf_mapping_type,
            m_surface_mapping_cfg_,
            m_sdf_mapping_cfg_);
        if (!m_sdf_mapping_) {
            ERL_FATAL("Failed to create SDF mapping");
            ros::shutdown();
            return;
        }

        if (m_setting_.use_odom) {
            if (m_setting_.odom_msg_type == "nav_msgs::Odometry") {
                m_sub_odom_ = m_nh_.subscribe(
                    m_setting_.odom_topic,
                    1,
                    &SdfMappingNode::CallbackOdomOdometry,
                    this);
            } else if (m_setting_.odom_msg_type == "geometry_msgs::TransformStamped") {
                m_sub_odom_ = m_nh_.subscribe(
                    m_setting_.odom_topic,
                    1,
                    &SdfMappingNode::CallbackOdomTransformStamped,
                    this);
            } else {
                ERL_FATAL("Invalid odometry message type: {}", m_setting_.odom_msg_type);
                ros::shutdown();
                return;
            }
            m_odom_queue_.resize(m_setting_.odom_queue_size);
        }
        switch (m_scan_type_) {
            case ScanType::Laser:
                ERL_INFO("Subscribing to {} as laser scan", m_setting_.scan_topic);
                m_sub_scan_ = m_nh_.subscribe(
                    m_setting_.scan_topic,
                    1,
                    &SdfMappingNode::CallbackLaserScan,
                    this);
                break;
            case ScanType::PointCloud:
                ERL_INFO("Subscribing to {} as point cloud", m_setting_.scan_topic);
                m_sub_scan_ = m_nh_.subscribe(
                    m_setting_.scan_topic,
                    1,
                    &SdfMappingNode::CallbackPointCloud2,
                    this);
                break;
            case ScanType::Depth:
                ERL_INFO("Subscribing to {} as depth image", m_setting_.scan_topic);
                m_sub_scan_ = m_nh_.subscribe(
                    m_setting_.scan_topic,
                    1,
                    &SdfMappingNode::CallbackDepthImage,
                    this);
                break;
        }

        // advertise the service to query the SDF mapping
        m_srv_query_sdf_ =
            m_nh_.advertiseService("sdf_query", &SdfMappingNode::CallbackSdfQuery, this);
        m_srv_save_map_ =
            m_nh_.advertiseService("save_map", &SdfMappingNode::CallbackSaveMap, this);

        // publish the occupancy tree used by the surface mapping
        if (m_setting_.publish_tree) {
            if (!TryToGetSurfaceMappingTree()) {
                ERL_FATAL("Failed to get surface mapping tree");
                ros::shutdown();
                return;
            }
            m_pub_tree_ = m_nh_.advertise<erl_geometry::OccupancyTreeMsg>(
                m_setting_.publish_tree_topic,
                1,
                true);
            m_pub_tree_timer_ = m_nh_.createTimer(
                ros::Duration(1.0 / m_setting_.publish_tree_frequency),
                &SdfMappingNode::CallbackPublishTree,
                this);
            m_msg_tree_.header.frame_id = m_setting_.world_frame;
            m_msg_tree_.header.seq = -1;
        }

        // publish the surface points used by the sdf mapping
        if (m_setting_.publish_surface_points) {
            m_pub_surface_points_ = m_nh_.advertise<sensor_msgs::PointCloud2>(
                m_setting_.publish_surface_points_topic,
                1,
                true);
            m_pub_surface_points_timer_ = m_nh_.createTimer(
                ros::Duration(1.0 / m_setting_.publish_surface_points_frequency),
                &SdfMappingNode::CallbackPublishSurfacePoints,
                this);
            m_msg_surface_points_.header.frame_id = m_setting_.world_frame;
            m_msg_surface_points_.header.seq = -1;
            m_msg_surface_points_.fields.resize(8);
            static const char* kSurfacePointsFieldNames[] = {
                "x",
                "y",
                "z",
                "normal_x",
                "normal_y",
                "normal_z",
                "var_position",
                "var_normal",
            };
            for (int i = 0; i < 8; ++i) {
                m_msg_surface_points_.fields[i].name = kSurfacePointsFieldNames[i];
                m_msg_surface_points_.fields[i].offset = i * 4;  // each field is 4 bytes (float)
                m_msg_surface_points_.fields[i].datatype = sensor_msgs::PointField::FLOAT32;
            }
            m_msg_surface_points_.point_step = 32;       // 8 fields * 4 bytes each
            m_msg_surface_points_.is_bigendian = false;  // little-endian
            m_msg_surface_points_.is_dense = false;      // there may be NaN values in the normals
            m_msg_surface_points_.height = 1;            // unorganized point cloud
        }

        m_pub_update_time_ = m_nh_.advertise<sensor_msgs::Temperature>("update_time", 1, true);
        m_pub_query_time_ = m_nh_.advertise<sensor_msgs::Temperature>("query_time", 1, true);
        m_msg_update_time_.header.frame_id = m_setting_.world_frame;
        m_msg_update_time_.header.seq = -1;
        m_msg_update_time_.temperature = 0.0;
        m_msg_update_time_.variance = 0.0;
        m_msg_query_time_.header.frame_id = m_setting_.world_frame;
        m_msg_query_time_.header.seq = -1;
        m_msg_query_time_.temperature = 0.0;
        m_msg_query_time_.variance = 0.0;

        ERL_INFO("SdfMappingNode ready. Waiting for scans + queries...");
    }

private:
    template<typename T>
    bool
    LoadParam(const std::string& param_name, T& param) {
        if (!m_nh_.hasParam(param_name)) { return true; }
        if (!m_nh_.getParam(param_name, param)) {
            ERL_WARN("Failed to load param {}", param_name);
            return false;
        }
        return true;
    }

    bool
    LoadParameters() {
        std::string setting_file;
        m_nh_.param<std::string>("setting_file", setting_file, "");
        if (!setting_file.empty()) {  // load the setting from the file first
            if (!m_setting_.FromYamlFile(setting_file)) {
                ERL_FATAL("Failed to load {}", setting_file);
                return false;
            }
        }
        // load the parameters from the node handle
        if (!LoadParam("surface_mapping_setting_type", m_setting_.surface_mapping_setting_type)) {
            return false;
        }
        if (!LoadParam("surface_mapping_setting_file", m_setting_.surface_mapping_setting_file)) {
            return false;
        }
        if (!LoadParam("sdf_mapping_setting_type", m_setting_.sdf_mapping_setting_type)) {
            return false;
        }
        if (!LoadParam("sdf_mapping_setting_file", m_setting_.sdf_mapping_setting_file)) {
            return false;
        }
        if (!LoadParam("sdf_mapping_type", m_setting_.sdf_mapping_type)) { return false; }
        if (!LoadParam("use_odom", m_setting_.use_odom)) { return false; }
        if (!LoadParam("odom_topic", m_setting_.odom_topic)) { return false; }
        if (!LoadParam("odom_msg_type", m_setting_.odom_msg_type)) { return false; }
        if (!LoadParam("odom_queue_size", m_setting_.odom_queue_size)) { return false; }
        if (!LoadParam("world_frame", m_setting_.world_frame)) { return false; }
        if (!LoadParam("sensor_frame", m_setting_.sensor_frame)) { return false; }
        if (!LoadParam("scan_topic", m_setting_.scan_topic)) { return false; }
        if (!LoadParam("scan_type", m_setting_.scan_type)) { return false; }
        if (!LoadParam("scan_in_local_frame", m_setting_.scan_in_local_frame)) { return false; }
        if (!LoadParam("depth_scale", m_setting_.depth_scale)) { return false; }
        if (!LoadParam("publish_tree", m_setting_.publish_tree)) { return false; }
        if (!LoadParam("publish_tree_topic", m_setting_.publish_tree_topic)) { return false; }
        if (!LoadParam("publish_tree_binary", m_setting_.publish_tree_binary)) { return false; }
        if (!LoadParam("publish_tree_frequency", m_setting_.publish_tree_frequency)) {
            return false;
        }
        if (!LoadParam("publish_surface_points", m_setting_.publish_surface_points)) {
            return false;
        }
        if (!LoadParam("publish_surface_points_topic", m_setting_.publish_surface_points_topic)) {
            return false;
        }
        if (!LoadParam(
                "publish_surface_points_frequency",
                m_setting_.publish_surface_points_frequency)) {
            return false;
        }
        // check the parameters
        if (m_setting_.surface_mapping_setting_type.empty()) {
            ERL_WARN("You must set ~surface_mapping_setting_type");
            return false;
        }
        if (m_setting_.surface_mapping_setting_file.empty()) {
            ERL_WARN("You must set ~surface_mapping_config");
            return false;
        }
        if (!std::filesystem::exists(m_setting_.surface_mapping_setting_file)) {
            ERL_WARN(
                "Surface mapping setting file {} does not exist",
                m_setting_.surface_mapping_setting_file);
            return false;
        }
        if (m_setting_.sdf_mapping_setting_type.empty()) {
            ERL_WARN("You must set ~sdf_mapping_setting_type");
            return false;
        }
        if (m_setting_.sdf_mapping_setting_file.empty()) {
            ERL_WARN("You must set ~sdf_mapping_config");
            return false;
        }
        if (!std::filesystem::exists(m_setting_.sdf_mapping_setting_file)) {
            ERL_WARN(
                "SDF mapping setting file {} does not exist",
                m_setting_.sdf_mapping_setting_file);
            return false;
        }
        if (m_setting_.sdf_mapping_type.empty()) {
            ERL_WARN("You must set ~sdf_mapping_type");
            return false;
        }
        if (m_setting_.use_odom && m_setting_.odom_topic.empty()) {
            ERL_WARN("Odometry topic is empty but use_odom is true");
            return false;
        }
        if (m_setting_.odom_queue_size <= 0) {
            ERL_WARN("Odometry queue size must be positive");
            return false;
        }
        if (m_setting_.world_frame.empty()) {
            ERL_WARN("World frame is empty");
            return false;
        }
        if (!m_setting_.use_odom && m_setting_.sensor_frame.empty()) {
            ERL_WARN("Sensor frame is empty but use_odom is false");
            return false;
        }
        if (m_setting_.scan_topic.empty()) {
            ERL_WARN("Scan topic is empty");
            return false;
        }
        if (m_setting_.scan_type == "sensor_msgs::LaserScan") {
            m_scan_type_ = ScanType::Laser;
        } else if (m_setting_.scan_type == "sensor_msgs::PointCloud2") {
            m_scan_type_ = ScanType::PointCloud;
        } else if (m_setting_.scan_type == "sensor_msgs::Image") {
            m_scan_type_ = ScanType::Depth;
        } else {
            ERL_WARN("Unknown scan type {}", m_setting_.scan_type);
            return false;
        }
        if (m_setting_.publish_tree && m_setting_.publish_tree_topic.empty()) {
            ERL_WARN("Publish tree topic is empty but publish_tree is true");
            return false;
        }
        if (m_setting_.publish_tree && m_setting_.publish_tree_frequency <= 0.0) {
            ERL_WARN("Publish tree frequency must be positive");
            return false;
        }
        if (m_setting_.publish_surface_points && m_setting_.publish_surface_points_topic.empty()) {
            ERL_WARN("Publish surface points topic is empty but publish_surface_points is true");
            return false;
        }
        if (m_setting_.publish_surface_points &&
            m_setting_.publish_surface_points_frequency <= 0.0) {
            ERL_WARN("Publish surface points frequency must be positive");
            return false;
        }
        return true;
    }

    bool
    TryToGetSurfaceMappingTree() {
        auto surface_mapping = m_sdf_mapping_->GetAbstractSurfaceMapping();
        m_tree_is_2d_ = surface_mapping->GetMapDim() == 2;
        m_tree_is_double_ = surface_mapping->IsDoublePrecision();
        if (m_tree_is_2d_) {
            if (m_tree_is_double_) {
                m_tree_ = GetTreeFromGpOccSurfaceMapping<double, 2>();
                if (m_tree_) { return true; }
                m_tree_ = GetTreeFromBayesianHilbertSurfaceMapping<double, 2>();
                if (m_tree_) { return true; }
                return false;
            }
            m_tree_ = GetTreeFromGpOccSurfaceMapping<float, 2>();
            if (m_tree_) { return true; }
            m_tree_ = GetTreeFromBayesianHilbertSurfaceMapping<float, 2>();
            if (m_tree_) { return true; }
            return false;
        }
        if (m_tree_is_double_) {
            m_tree_ = GetTreeFromGpOccSurfaceMapping<double, 3>();
            if (m_tree_) { return true; }
            m_tree_ = GetTreeFromBayesianHilbertSurfaceMapping<double, 3>();
            if (m_tree_) { return true; }
            return false;
        }
        m_tree_ = GetTreeFromGpOccSurfaceMapping<float, 3>();
        if (m_tree_) { return true; }
        m_tree_ = GetTreeFromBayesianHilbertSurfaceMapping<float, 3>();
        if (m_tree_) { return true; }
        return false;
    }

    template<typename Dtype, int Dim>
    std::shared_ptr<const void>
    GetTreeFromGpOccSurfaceMapping() {
        auto mapping = std::dynamic_pointer_cast<GpOccSurfaceMapping<Dtype, Dim>>(
            m_sdf_mapping_->GetAbstractSurfaceMapping());
        if (!mapping) { return nullptr; }
        return mapping->GetTree();
    }

    template<typename Dtype, int Dim>
    std::shared_ptr<const void>
    GetTreeFromBayesianHilbertSurfaceMapping() {
        auto mapping = std::dynamic_pointer_cast<BayesianHilbertSurfaceMapping<Dtype, Dim>>(
            m_sdf_mapping_->GetAbstractSurfaceMapping());
        if (!mapping) { return nullptr; }
        return mapping->GetTree();
    }

    // get the pose for time t
    std::tuple<bool, Eigen::MatrixXd, Eigen::VectorXd>
    GetSensorPose(const ros::Time& time) {
        if (m_setting_.use_odom) {
            std::lock_guard<std::mutex> lock(m_odom_queue_lock_);
            // get the latest odometry message
            const int& head = m_odom_queue_head_;
            if (head < 0) {
                ERL_WARN("No odometry message available");
                return {false, {}, {}};
            }
            geometry_msgs::TransformStamped* transform = nullptr;
            for (int i = head; i >= 0; --i) {
                if (m_odom_queue_[i].header.stamp <= time) {
                    transform = &m_odom_queue_[i];
                    break;
                }
            }
            if (!transform) {  // search older messages
                const int size = static_cast<int>(m_odom_queue_.size());
                for (int i = size - 1; i > head; --i) {
                    if (m_odom_queue_[i].header.stamp <= time) {
                        transform = &m_odom_queue_[i];
                        break;
                    }
                }
            }
            if (!transform) {
                ERL_WARN("No odometry message available for time {}", time.toSec());
                return {false, {}, {}};
            }
            auto& pose = transform->transform;
            if (m_sdf_mapping_->GetMapDim() == 2) {
                const double yaw = tf::getYaw(pose.rotation);
                Eigen::Matrix2d rotation = Eigen::Rotation2Dd(yaw).toRotationMatrix();
                Eigen::Vector2d translation(pose.translation.x, pose.translation.y);
                return {true, rotation, translation};
            }
            Eigen::Matrix3d rotation = Eigen::Quaterniond(
                                           pose.rotation.w,
                                           pose.rotation.x,
                                           pose.rotation.y,
                                           pose.rotation.z)
                                           .toRotationMatrix();
            Eigen::Vector3d translation(pose.translation.x, pose.translation.y, pose.translation.z);
            return {true, rotation, translation};
        }
        // get the latest transform from the tf buffer
        geometry_msgs::TransformStamped transform_stamped;
        while (true) {
            try {
                transform_stamped = m_tf_buffer_.lookupTransform(
                    m_setting_.world_frame,
                    m_setting_.sensor_frame,
                    time,
                    ros::Duration(5.0));
                break;
            } catch (tf2::TransformException& ex) {
                ERL_WARN(ex.what());
                (void) ros::Duration(1.0).sleep();
            }
        }
        Eigen::Matrix4d pose = tf2::transformToEigen(transform_stamped).matrix();
        if (m_sdf_mapping_->GetMapDim() == 2) {
            Eigen::Matrix2d rotation = pose.block<2, 2>(0, 0);
            Eigen::Vector2d translation = pose.block<2, 1>(0, 3);
            return {true, rotation, translation};
        }
        Eigen::Matrix3d rotation = pose.block<3, 3>(0, 0);
        Eigen::Vector3d translation = pose.block<3, 1>(0, 3);
        return {true, rotation, translation};
    }

    // --- callbacks to collect pose+scan and then update the map ---
    void
    CallbackOdomOdometry(const nav_msgs::Odometry::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(m_odom_queue_lock_);
        if (static_cast<int>(m_odom_queue_.size()) >= m_setting_.odom_queue_size) {
            auto& transform = m_odom_queue_[m_odom_queue_head_];
            transform.header = msg->header;
            transform.child_frame_id = msg->child_frame_id;
            transform.transform.rotation = msg->pose.pose.orientation;
            transform.transform.translation.x = msg->pose.pose.position.x;
            transform.transform.translation.y = msg->pose.pose.position.y;
            transform.transform.translation.z = msg->pose.pose.position.z;
            m_odom_queue_head_ = (m_odom_queue_head_ + 1) % m_setting_.odom_queue_size;
        } else {
            geometry_msgs::TransformStamped transform;
            transform.header = msg->header;
            transform.child_frame_id = msg->child_frame_id;
            transform.transform.rotation = msg->pose.pose.orientation;
            transform.transform.translation.x = msg->pose.pose.position.x;
            transform.transform.translation.y = msg->pose.pose.position.y;
            transform.transform.translation.z = msg->pose.pose.position.z;
            m_odom_queue_.push_back(transform);
            ++m_odom_queue_head_;
        }
    }

    void
    CallbackOdomTransformStamped(const geometry_msgs::TransformStamped::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(m_odom_queue_lock_);
        if (static_cast<int>(m_odom_queue_.size()) >= m_setting_.odom_queue_size) {
            auto& transform = m_odom_queue_[m_odom_queue_head_];
            transform.header = msg->header;
            transform.child_frame_id = msg->child_frame_id;
            transform.transform = msg->transform;
            m_odom_queue_head_ = (m_odom_queue_head_ + 1) % m_setting_.odom_queue_size;
        } else {
            m_odom_queue_.push_back(*msg);
            ++m_odom_queue_head_;
        }
    }

    void
    CallbackLaserScan(const sensor_msgs::LaserScan::ConstPtr& msg) {
        m_lidar_scan_2d_ = msg;
        TryUpdate(msg->header.stamp);
    }

    void
    CallbackPointCloud2(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        m_lidar_scan_3d_ = msg;
        TryUpdate(msg->header.stamp);
    }

    void
    CallbackDepthImage(const sensor_msgs::Image::ConstPtr& msg) {
        m_depth_image_ = msg;
        TryUpdate(msg->header.stamp);
    }

    bool
    GetScanFromLaserScan(Eigen::MatrixXd& scan) {
        if (!m_lidar_scan_2d_) {
            ERL_WARN("No laser scan data available");
            return false;
        }
        auto& scan_msg = *m_lidar_scan_2d_;
        if (scan_msg.ranges.empty()) {
            ERL_WARN("Laser scan data is empty");
            m_lidar_scan_2d_.reset();
            return false;
        }
        scan = Eigen::Map<const Eigen::VectorXf>(scan_msg.ranges.data(), scan_msg.ranges.size())
                   .cast<double>();
        m_lidar_scan_2d_.reset();
        return true;
    }

    bool
    GetScanFromPointCloud2(Eigen::MatrixXd& scan) {
        if (!m_lidar_scan_3d_) {
            ERL_WARN("No point cloud data available");
            return false;
        }
        auto& cloud = *m_lidar_scan_3d_;
        if (cloud.fields.empty() || cloud.data.empty()) {
            ERL_WARN("Point cloud data is empty");
            m_lidar_scan_3d_.reset();
            return false;
        }
        if (cloud.data.size() != cloud.width * cloud.height * cloud.point_step) {
            ERL_WARN("Point cloud data size does not match width, height, and point step");
            m_lidar_scan_3d_.reset();
            return false;
        }
        if (cloud.row_step != cloud.width * cloud.point_step) {
            ERL_WARN("Point cloud row step does not match width and point step");
            m_lidar_scan_3d_.reset();
            return false;
        }

        // validate x, y, z are present
        const int32_t xi = rviz::findChannelIndex(m_lidar_scan_3d_, "x");
        const int32_t yi = rviz::findChannelIndex(m_lidar_scan_3d_, "y");
        const int32_t zi = rviz::findChannelIndex(m_lidar_scan_3d_, "z");
        if (xi < 0 || yi < 0 || zi < 0) {
            ERL_WARN("Point cloud does not contain x, y, z fields");
            m_lidar_scan_3d_.reset();
            return false;
        }
        // validate x, y, z fields have the same data type
        const uint8_t& xtype = cloud.fields[xi].datatype;
        const uint8_t& ytype = cloud.fields[yi].datatype;
        const uint8_t& ztype = cloud.fields[zi].datatype;
        if (xtype != ytype || xtype != ztype || ytype != ztype) {
            ERL_WARN("Point cloud x, y, z fields have different data types");
            m_lidar_scan_3d_.reset();
            return false;
        }
        const uint32_t xoff = cloud.fields[xi].offset;
        const uint32_t yoff = cloud.fields[yi].offset;
        const uint32_t zoff = cloud.fields[zi].offset;
        const uint32_t point_step = cloud.point_step;
        scan.resize(3, static_cast<long>(cloud.width * cloud.height));
        long point_count = 0;
        const uint8_t* ptr = cloud.data.data();
        const uint8_t* ptr_end = cloud.data.data() + cloud.data.size();
        if (xtype == sensor_msgs::PointField::FLOAT32) {
            for (; ptr < ptr_end; ptr += point_step) {
                double* p_out = scan.col(point_count).data();
                p_out[0] = static_cast<double>(*reinterpret_cast<const float*>(ptr + xoff));
                p_out[1] = static_cast<double>(*reinterpret_cast<const float*>(ptr + yoff));
                p_out[2] = static_cast<double>(*reinterpret_cast<const float*>(ptr + zoff));
                if (std::isfinite(p_out[0]) && std::isfinite(p_out[1]) && std::isfinite(p_out[2])) {
                    ++point_count;
                }
            }
        } else if (xtype == sensor_msgs::PointField::FLOAT64) {
            for (; ptr < ptr_end; ptr += point_step) {
                double* p_out = scan.col(point_count).data();
                p_out[0] = *reinterpret_cast<const double*>(ptr + xoff);
                p_out[1] = *reinterpret_cast<const double*>(ptr + yoff);
                p_out[2] = *reinterpret_cast<const double*>(ptr + zoff);
                if (std::isfinite(p_out[0]) && std::isfinite(p_out[1]) && std::isfinite(p_out[2])) {
                    ++point_count;
                }
            }
        } else {
            ERL_WARN("Unsupported point cloud data type {}", xtype);
            m_lidar_scan_3d_.reset();
            return false;
        }
        if (point_count == 0) {
            ERL_WARN("No valid points in point cloud");
            m_lidar_scan_3d_.reset();
            return false;
        }
        scan.conservativeResize(3, point_count);
        m_lidar_scan_3d_.reset();
        return true;
    }

    bool
    GetScanFromDepthImage(Eigen::MatrixXd& scan) {
        if (!m_depth_image_) {
            ERL_WARN("No depth image available");
            return false;
        }
        using namespace sensor_msgs::image_encodings;
        if (m_depth_image_->encoding == TYPE_32FC1) {
            Eigen::MatrixXf depth_image = Eigen::Map<const Eigen::MatrixXf>(
                reinterpret_cast<const float*>(m_depth_image_->data.data()),
                m_depth_image_->width,
                m_depth_image_->height);
            scan = depth_image.cast<double>().transpose();
        } else if (m_depth_image_->encoding == TYPE_64FC1) {
            scan = Eigen::Map<const Eigen::MatrixXd>(
                       reinterpret_cast<const double*>(m_depth_image_->data.data()),
                       m_depth_image_->width,
                       m_depth_image_->height)
                       .transpose();
        } else if (m_depth_image_->encoding == TYPE_16UC1) {
            scan = Eigen::Map<const Eigen::MatrixX<uint16_t>>(
                       reinterpret_cast<const uint16_t*>(m_depth_image_->data.data()),
                       m_depth_image_->width,
                       m_depth_image_->height)
                       .cast<double>()
                       .transpose();
        } else {
            ERL_WARN("Unsupported depth image encoding {}", m_depth_image_->encoding);
            m_depth_image_.reset();
            return false;
        }
        if (scan.size() > 0) { scan.array() *= m_setting_.depth_scale; }  // convert to meters
        m_depth_image_.reset();
        return true;
    }

    void
    TryUpdate(const ros::Time& time) {
        if (!m_lidar_scan_2d_ && !m_lidar_scan_3d_ && !m_depth_image_) {
            ERL_WARN("No scan data available");
            return;
        }
        const auto [ok, rotation, translation] = GetSensorPose(time);
        if (!ok) {
            ERL_WARN("Failed to get sensor pose");
            return;
        }

        bool is_point = false;
        Eigen::MatrixXd scan;
        switch (m_scan_type_) {
            case ScanType::Laser:
                if (!GetScanFromLaserScan(scan)) { return; }
                is_point = false;
                break;
            case ScanType::PointCloud:
                if (!GetScanFromPointCloud2(scan)) { return; }
                is_point = true;
                break;
            case ScanType::Depth:
                if (!GetScanFromDepthImage(scan)) { return; }
                break;
        }
        const bool in_local = m_setting_.scan_in_local_frame;
        auto t1 = ros::WallTime::now();
        bool success = m_sdf_mapping_->Update(rotation, translation, scan, is_point, in_local);
        auto t2 = ros::WallTime::now();
        m_msg_update_time_.header.stamp = time;
        m_msg_update_time_.header.seq++;
        m_msg_update_time_.temperature = (t2 - t1).toSec();
        m_pub_update_time_.publish(m_msg_update_time_);
        if (!success) { ERL_WARN("Failed to update SDF mapping"); }
    }

    // --- service handler: runs Test() on the current map ---
    bool
    CallbackSdfQuery(
        erl_sdf_mapping::SdfQuery::Request& req,
        erl_sdf_mapping::SdfQuery::Response& res) {

        if (!m_sdf_mapping_) {
            ERL_WARN("SDF mapping is not initialized");
            return false;
        }

        const int dim = m_sdf_mapping_->GetMapDim();
        const bool is_double = m_sdf_mapping_->IsDoublePrecision();
        if (dim == 2) {
            if (is_double) {
                return GetQueryResponse<double, 2>(req, res);
            } else {
                return GetQueryResponse<float, 2>(req, res);
            }
        }
        if (dim == 3) {
            if (is_double) {
                return GetQueryResponse<double, 3>(req, res);
            } else {
                return GetQueryResponse<float, 3>(req, res);
            }
        }
        ERL_WARN("Unknown map dimension {}", dim);
        return false;
    }

    template<typename Dtype, int Dim>
    bool
    GetQueryResponse(
        erl_sdf_mapping::SdfQuery::Request& req,
        erl_sdf_mapping::SdfQuery::Response& res) {

        using SdfMappingSetting = GpSdfMappingSetting<Dtype, Dim>;
        using QuerySetting = typename SdfMappingSetting::TestQuery;
        auto sdf_mapping_setting = std::dynamic_pointer_cast<SdfMappingSetting>(m_sdf_mapping_cfg_);
        if (!sdf_mapping_setting) {
            ERL_WARN("Wrong Dtype or Dim for GetQueryResponse");
            return false;
        }

        const auto n = static_cast<int>(req.query_points.size());
        Eigen::Map<const Eigen::MatrixXd> positions(
            reinterpret_cast<const double*>(req.query_points.data()),
            3,
            n);
        Eigen::VectorXd distances(n);
        Eigen::MatrixXd gradients(Dim, n);
        Eigen::MatrixXd variances(Dim + 1, n);
        Eigen::MatrixXd covariances(Dim * (Dim + 1) / 2, n);
        distances.setConstant(0.0);
        gradients.setConstant(0.0);
        variances.setConstant(1.0e6);

        auto t1 = ros::WallTime::now();
        bool ok = m_sdf_mapping_->Predict(positions, distances, gradients, variances, covariances);
        auto t2 = ros::WallTime::now();
        m_msg_query_time_.header.stamp = ros::Time::now();
        m_msg_query_time_.header.seq++;
        m_msg_query_time_.temperature = (t2 - t1).toSec();
        m_pub_query_time_.publish(m_msg_query_time_);
        res.success = ok;
        if (!ok) { return false; }

        // TODO: CHECK ALL std::memcpy() CALLS FOR CORRECTNESS

        // store the results in the response
        res.dim = Dim;
        /// SDF
        res.signed_distances.resize(n);
        std::memcpy(
            reinterpret_cast<double*>(res.signed_distances.data()),
            reinterpret_cast<const double*>(distances.data()),
            n * sizeof(double));
        /// store the remaining results based on the query setting
        const QuerySetting& query_setting = sdf_mapping_setting->test_query;
        res.compute_gradient = query_setting.compute_gradient;
        res.compute_gradient_variance = query_setting.compute_gradient_variance;
        res.compute_covariance = query_setting.compute_covariance;
        /// gradients
        res.gradients.clear();
        if (query_setting.compute_gradient) {
            res.gradients.resize(n);
            if (Dim == 2) {
                for (int i = 0; i < n; ++i) {
                    res.gradients[i].x = gradients(0, i);
                    res.gradients[i].y = gradients(1, i);
                    res.gradients[i].z = 0.0;  // z is not used in 2D
                }
            } else {
                std::memcpy(
                    reinterpret_cast<char*>(res.gradients.data()),
                    reinterpret_cast<const char*>(gradients.data()),
                    3 * n * sizeof(double));
            }
        }
        /// variances
        if (query_setting.compute_gradient_variance) {
            res.variances.resize(n * (Dim + 1));
            std::memcpy(
                reinterpret_cast<char*>(res.variances.data()),
                reinterpret_cast<const char*>(variances.data()),
                n * (Dim + 1) * sizeof(double));
        } else {
            res.variances.resize(n);
            for (int i = 0; i < n; ++i) { res.variances[i] = variances(0, i); }
        }
        /// covariances
        res.covariances.clear();
        if (query_setting.compute_covariance) {
            res.covariances.resize(n * Dim * (Dim + 1) / 2);
            std::memcpy(
                reinterpret_cast<char*>(res.covariances.data()),
                reinterpret_cast<const char*>(covariances.data()),
                n * Dim * (Dim + 1) / 2 * sizeof(double));
        }
        return true;
    }

    bool
    CallbackSaveMap(
        erl_sdf_mapping::SaveMap::Request& req,
        erl_sdf_mapping::SaveMap::Response& res) {
        if (!m_sdf_mapping_) {
            ERL_WARN("SDF mapping is not initialized");
            res.success = false;
            return false;
        }
        if (req.name.empty()) {
            ERL_WARN("Map file name is empty");
            res.success = false;
            return false;
        }
        std::filesystem::path map_file = req.name;
        map_file = std::filesystem::absolute(map_file);
        std::filesystem::create_directories(map_file.parent_path());
        {
            auto lock = m_sdf_mapping_->GetLockGuard();
            using Serializer = Serialization<AbstractGpSdfMapping>;
            res.success = Serializer::Write(map_file, m_sdf_mapping_.get());
        }
        return true;
    }

    void
    CallbackPublishTree(const ros::TimerEvent& /* event */) {
        if (!m_tree_) { return; }
        if (m_pub_tree_.getNumSubscribers() == 0) { return; }  // no subscribers
        bool success = false;
        if (m_tree_is_2d_) {
            if (m_tree_is_double_) {
                success = GenerateOccupancyTreeMsgForQuadtree<double>(m_msg_tree_);
            } else {
                success = GenerateOccupancyTreeMsgForQuadtree<float>(m_msg_tree_);
            }
        } else {
            if (m_tree_is_double_) {
                success = GenerateOccupancyTreeMsgForOctree<double>(m_msg_tree_);
            } else {
                success = GenerateOccupancyTreeMsgForOctree<float>(m_msg_tree_);
            }
        }
        if (!success) { return; }
        m_msg_tree_.header.stamp = ros::Time::now();
        m_pub_tree_.publish(m_msg_tree_);
    }

    template<typename Dtype>
    bool
    GenerateOccupancyTreeMsgForQuadtree(erl_geometry::OccupancyTreeMsg& msg) {
        using namespace erl::geometry;
        auto tree = std::reinterpret_pointer_cast<const AbstractOccupancyQuadtree<Dtype>>(m_tree_);
        if (!tree) {
            ERL_WARN("Failed to cast to quadtree");
            return false;
        }
        {
            auto lock = m_sdf_mapping_->GetAbstractSurfaceMapping()->GetLockGuard();
            SaveToOccupancyTreeMsg<Dtype>(tree, m_setting_.publish_tree_binary, msg);
        }
        return true;
    }

    template<typename Dtype>
    bool
    GenerateOccupancyTreeMsgForOctree(erl_geometry::OccupancyTreeMsg& msg) {
        using namespace erl::geometry;
        auto tree = std::reinterpret_pointer_cast<const AbstractOccupancyOctree<Dtype>>(m_tree_);
        if (!tree) {
            ERL_WARN("Failed to cast to octree");
            return false;
        }
        {
            auto lock = m_sdf_mapping_->GetAbstractSurfaceMapping()->GetLockGuard();
            SaveToOccupancyTreeMsg<Dtype>(tree, m_setting_.publish_tree_binary, msg);
        }
        return true;
    }

    void
    CallbackPublishSurfacePoints(const ros::TimerEvent& /* event */) {
        if (m_pub_surface_points_.getNumSubscribers() == 0) { return; }  // no subscribers

        using namespace erl::sdf_mapping;
        std::vector<SurfaceData<double, 3>> data =
            m_sdf_mapping_->GetAbstractSurfaceMapping()->GetSurfaceData();

        auto& msg = m_msg_surface_points_;
        msg.header.stamp = ros::Time::now();
        msg.header.seq++;
        for (int i = 0; i < 8; ++i) { msg.fields[i].count = static_cast<uint32_t>(data.size()); }
        msg.width = static_cast<uint32_t>(data.size());
        msg.row_step = msg.point_step * msg.width;
        msg.data.resize(msg.row_step * msg.height);
        // uint8_t* ptr = msg.data.data();
        std::vector<float> point_data(8);
        float* ptr = reinterpret_cast<float*>(msg.data.data());
        for (const auto& d: data) {
            ptr[0] = static_cast<float>(d.position.x());
            ptr[1] = static_cast<float>(d.position.y());
            ptr[2] = static_cast<float>(d.position.z());
            ptr[3] = static_cast<float>(d.normal.x());
            ptr[4] = static_cast<float>(d.normal.y());
            ptr[5] = static_cast<float>(d.normal.z());
            ptr[6] = static_cast<float>(d.var_position);
            ptr[7] = static_cast<float>(d.var_normal);
            ptr += 8;
        }
        m_pub_surface_points_.publish(msg);
    }
};

template<>
struct YAML::convert<SdfMappingNodeSetting> {
    static YAML::Node
    encode(const SdfMappingNodeSetting& setting) {
        YAML::Node node;
        ERL_YAML_SAVE_ATTR(node, setting, surface_mapping_setting_type);
        ERL_YAML_SAVE_ATTR(node, setting, surface_mapping_setting_file);
        ERL_YAML_SAVE_ATTR(node, setting, sdf_mapping_setting_type);
        ERL_YAML_SAVE_ATTR(node, setting, sdf_mapping_setting_file);
        ERL_YAML_SAVE_ATTR(node, setting, sdf_mapping_type);
        ERL_YAML_SAVE_ATTR(node, setting, use_odom);
        ERL_YAML_SAVE_ATTR(node, setting, odom_topic);
        ERL_YAML_SAVE_ATTR(node, setting, odom_queue_size);
        ERL_YAML_SAVE_ATTR(node, setting, world_frame);
        ERL_YAML_SAVE_ATTR(node, setting, sensor_frame);
        ERL_YAML_SAVE_ATTR(node, setting, scan_type);
        ERL_YAML_SAVE_ATTR(node, setting, scan_in_local_frame);
        ERL_YAML_SAVE_ATTR(node, setting, depth_scale);
        ERL_YAML_SAVE_ATTR(node, setting, publish_tree);
        ERL_YAML_SAVE_ATTR(node, setting, publish_tree_topic);
        ERL_YAML_SAVE_ATTR(node, setting, publish_tree_binary);
        ERL_YAML_SAVE_ATTR(node, setting, publish_tree_frequency);
        ERL_YAML_SAVE_ATTR(node, setting, publish_surface_points);
        ERL_YAML_SAVE_ATTR(node, setting, publish_surface_points_topic);
        return node;
    }

    static bool
    decode(const YAML::Node& node, SdfMappingNodeSetting& setting) {
        ERL_YAML_LOAD_ATTR(node, setting, surface_mapping_setting_type);
        ERL_YAML_LOAD_ATTR(node, setting, surface_mapping_setting_file);
        ERL_YAML_LOAD_ATTR(node, setting, sdf_mapping_setting_type);
        ERL_YAML_LOAD_ATTR(node, setting, sdf_mapping_setting_file);
        ERL_YAML_LOAD_ATTR(node, setting, sdf_mapping_type);
        ERL_YAML_LOAD_ATTR(node, setting, use_odom);
        ERL_YAML_LOAD_ATTR(node, setting, odom_topic);
        ERL_YAML_LOAD_ATTR(node, setting, odom_queue_size);
        ERL_YAML_LOAD_ATTR(node, setting, world_frame);
        ERL_YAML_LOAD_ATTR(node, setting, sensor_frame);
        ERL_YAML_LOAD_ATTR(node, setting, scan_type);
        ERL_YAML_LOAD_ATTR(node, setting, scan_in_local_frame);
        ERL_YAML_LOAD_ATTR(node, setting, depth_scale);
        ERL_YAML_LOAD_ATTR(node, setting, publish_tree);
        ERL_YAML_LOAD_ATTR(node, setting, publish_tree_topic);
        ERL_YAML_LOAD_ATTR(node, setting, publish_tree_binary);
        ERL_YAML_LOAD_ATTR(node, setting, publish_tree_frequency);
        ERL_YAML_LOAD_ATTR(node, setting, publish_surface_points);
        ERL_YAML_LOAD_ATTR(node, setting, publish_surface_points_topic);
        return true;
    }
};

int
main(int argc, char** argv) {
    ros::init(argc, argv, "sdf_mapping_node");
    ros::NodeHandle nh("~");  // ~: shorthand for the private namespace
    SdfMappingNode node(nh);
    ros::spin();
    return 0;
}
