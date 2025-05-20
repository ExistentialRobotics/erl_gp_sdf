#include "erl_common/yaml.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping.hpp"
#include "erl_sdf_mapping/SaveMap.h"
#include "erl_sdf_mapping/SdfQuery.h"

#include <geometry_msgs/Vector3.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
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
};

class SdfMappingNode {
public:
private:
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
    std::shared_ptr<YamlableBase> m_surface_mapping_cfg_ = nullptr;
    std::shared_ptr<YamlableBase> m_sdf_mapping_cfg_ = nullptr;
    std::shared_ptr<AbstractGpSdfMapping> m_sdf_mapping_ = nullptr;
    bool m_is_double_ = false;
    std::mutex m_odom_queue_lock_;
    std::vector<nav_msgs::Odometry::ConstPtr> m_odom_queue_{};
    int m_odom_queue_head_ = -1;
    tf2_ros::Buffer m_tf_buffer_;
    tf2_ros::TransformListener m_tf_listener_{m_tf_buffer_};
    sensor_msgs::LaserScan::ConstPtr m_lidar_scan_2d_ = nullptr;
    sensor_msgs::PointCloud2::ConstPtr m_lidar_scan_3d_ = nullptr;
    sensor_msgs::Image::ConstPtr m_depth_image_ = nullptr;

public:
    SdfMappingNode(ros::NodeHandle& nh)
        : m_nh_(nh) {
        if (!LoadParameters()) {
            ROS_FATAL("Failed to load parameters");
            ros::shutdown();
            return;
        }
        auto& setting_factory = YamlableBase::Factory::GetInstance();
        // load the surface mapping config
        m_surface_mapping_cfg_ = setting_factory.Create(m_setting_.surface_mapping_setting_type);
        if (!m_surface_mapping_cfg_) {
            ROS_FATAL("Failed to create surface mapping config");
            ros::shutdown();
            return;
        }
        if (!m_surface_mapping_cfg_->FromYamlFile(m_setting_.surface_mapping_setting_file)) {
            ROS_FATAL("Failed to load %s", m_setting_.surface_mapping_setting_file.c_str());
            ros::shutdown();
            return;
        }
        m_sdf_mapping_cfg_ = setting_factory.Create(m_setting_.sdf_mapping_setting_type);
        if (!m_sdf_mapping_cfg_) {
            ROS_FATAL("Failed to create SDF mapping config");
            ros::shutdown();
            return;
        }
        if (!m_sdf_mapping_cfg_->FromYamlFile(m_setting_.sdf_mapping_setting_file)) {
            ROS_FATAL("Failed to load %s", m_setting_.sdf_mapping_setting_file.c_str());
            ros::shutdown();
            return;
        }
        m_sdf_mapping_ = AbstractGpSdfMapping::Create(
            m_setting_.sdf_mapping_type,
            m_surface_mapping_cfg_,
            m_sdf_mapping_cfg_);
        if (!m_sdf_mapping_) {
            ROS_FATAL("Failed to create SDF mapping");
            ros::shutdown();
            return;
        }
        if (m_sdf_mapping_->map_dim == 2) {
            auto setting = std::dynamic_pointer_cast<GpSdfMappingSetting2Dd>(m_sdf_mapping_cfg_);
            m_is_double_ = setting != nullptr;
        } else {
            auto setting = std::dynamic_pointer_cast<GpSdfMappingSetting3Dd>(m_sdf_mapping_cfg_);
            m_is_double_ = setting != nullptr;
        }

        if (m_setting_.use_odom) {
            m_sub_odom_ =
                m_nh_.subscribe(m_setting_.odom_topic, 1, &SdfMappingNode::CallbackOdom, this);
            m_odom_queue_.resize(m_setting_.odom_queue_size);
        }
        switch (m_scan_type_) {
            case ScanType::Laser:
                ROS_INFO("Subscribing to %s as laser scan", m_setting_.scan_topic.c_str());
                m_sub_scan_ = m_nh_.subscribe(
                    m_setting_.scan_topic,
                    1,
                    &SdfMappingNode::CallbackLaserScan,
                    this);
                break;
            case ScanType::PointCloud:
                ROS_INFO("Subscribing to %s as point cloud", m_setting_.scan_topic.c_str());
                m_sub_scan_ = m_nh_.subscribe(
                    m_setting_.scan_topic,
                    1,
                    &SdfMappingNode::CallbackPointCloud2,
                    this);
                break;
            case ScanType::Depth:
                ROS_INFO("Subscribing to %s as depth image", m_setting_.scan_topic.c_str());
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

        ROS_INFO("SdfMappingNode ready. Waiting for scans + queries...");
    }

private:
    template<typename T>
    bool
    LoadParam(const std::string& param_name, T& param) {
        if (!m_nh_.hasParam(param_name)) { return true; }
        if (!m_nh_.getParam(param_name, param)) {
            ROS_WARN("Failed to load param %s", param_name.c_str());
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
                ROS_FATAL("Failed to load %s", setting_file.c_str());
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
        if (!LoadParam("use_odom", m_setting_.use_odom)) { return false; }
        if (!LoadParam("odom_topic", m_setting_.odom_topic)) { return false; }
        if (!LoadParam("odom_queue_size", m_setting_.odom_queue_size)) { return false; }
        if (!LoadParam("world_frame", m_setting_.world_frame)) { return false; }
        if (!LoadParam("sensor_frame", m_setting_.sensor_frame)) { return false; }
        if (!LoadParam("scan_topic", m_setting_.scan_topic)) { return false; }
        if (!LoadParam("scan_type", m_setting_.scan_type)) { return false; }

        if (m_setting_.surface_mapping_setting_type.empty()) {
            ROS_WARN("You must set ~surface_mapping_setting_type");
            return false;
        }
        if (m_setting_.sdf_mapping_setting_type.empty()) {
            ROS_WARN("You must set ~sdf_mapping_setting_type");
            return false;
        }
        if (m_setting_.sdf_mapping_type.empty()) {
            ROS_WARN("You must set ~sdf_mapping_type");
            return false;
        }
        if (m_setting_.surface_mapping_setting_file.empty()) {
            ROS_WARN("You must set ~surface_mapping_config");
            return false;
        }
        if (m_setting_.sdf_mapping_setting_file.empty()) {
            ROS_WARN("You must set ~sdf_mapping_config");
            return false;
        }
        if (!std::filesystem::exists(m_setting_.surface_mapping_setting_file)) {
            ROS_WARN(
                "Surface mapping setting file %s does not exist",
                m_setting_.surface_mapping_setting_file.c_str());
            return false;
        }
        if (!std::filesystem::exists(m_setting_.sdf_mapping_setting_file)) {
            ROS_WARN(
                "SDF mapping setting file %s does not exist",
                m_setting_.sdf_mapping_setting_file.c_str());
            return false;
        }
        if (m_setting_.use_odom && m_setting_.odom_topic.empty()) {
            ROS_WARN("Odometry topic is empty but use_odom is true");
            return false;
        }
        if (m_setting_.odom_queue_size <= 0) {
            ROS_WARN("Odometry queue size must be positive");
            return false;
        }
        if (m_setting_.scan_topic.empty()) {
            ROS_WARN("Scan topic is empty");
            return false;
        }
        if (m_setting_.scan_type == "sensor_msgs::LaserScan") {
            m_scan_type_ = ScanType::Laser;
        } else if (m_setting_.scan_type == "sensor_msgs::PointCloud2") {
            m_scan_type_ = ScanType::PointCloud;
        } else if (m_setting_.scan_type == "sensor_msgs::Image") {
            m_scan_type_ = ScanType::Depth;
        } else {
            ROS_WARN("Unknown scan type %s", m_setting_.scan_type.c_str());
            return false;
        }
        return true;
    }

    // get the pose for time t
    std::tuple<bool, Eigen::MatrixXd, Eigen::VectorXd>
    GetSensorPose(const ros::Time& time) {
        if (m_setting_.use_odom) {
            std::lock_guard<std::mutex> lock(m_odom_queue_lock_);
            // get the latest odometry message
            const int& head = m_odom_queue_head_;
            if (head < 0) {
                ROS_WARN("No odometry message available");
                return {false, {}, {}};
            }
            nav_msgs::Odometry::ConstPtr odom = nullptr;
            for (int i = head; i >= 0; --i) {
                if (m_odom_queue_[i]->header.stamp <= time) {
                    odom = m_odom_queue_[i];
                    break;
                }
            }
            if (!odom) {  // search older messages
                const int size = static_cast<int>(m_odom_queue_.size());
                for (int i = size - 1; i > head; --i) {
                    if (m_odom_queue_[i]->header.stamp <= time) {
                        odom = m_odom_queue_[i];
                        break;
                    }
                }
            }
            if (!odom) {
                ROS_WARN("No odometry message available for time %f", time.toSec());
                return {false, {}, {}};
            }
            auto& pose = odom->pose.pose;
            if (m_sdf_mapping_->map_dim == 2) {
                const double yaw = tf::getYaw(pose.orientation);
                Eigen::Matrix2d rotation = Eigen::Rotation2Dd(yaw).toRotationMatrix();
                Eigen::Vector2d translation(pose.position.x, pose.position.y);
                return {true, rotation, translation};
            }
            Eigen::Matrix3d rotation = Eigen::Quaterniond(
                                           pose.orientation.w,
                                           pose.orientation.x,
                                           pose.orientation.y,
                                           pose.orientation.z)
                                           .toRotationMatrix();
            Eigen::Vector3d translation(pose.position.x, pose.position.y, pose.position.z);
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
                ROS_WARN("%s", ex.what());
                (void) ros::Duration(1.0).sleep();
            }
        }
        Eigen::Matrix4d pose = tf2::transformToEigen(transform_stamped).matrix();
        if (m_sdf_mapping_->map_dim == 2) {
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
    CallbackOdom(const nav_msgs::Odometry::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(m_odom_queue_lock_);
        if (static_cast<int>(m_odom_queue_.size()) >= m_setting_.odom_queue_size) {
            m_odom_queue_[m_odom_queue_head_] = msg;
            m_odom_queue_head_ = (m_odom_queue_head_ + 1) % m_setting_.odom_queue_size;
        } else {
            m_odom_queue_.push_back(msg);
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

    void
    TryUpdate(const ros::Time& time) {
        if (!m_lidar_scan_2d_ && !m_lidar_scan_3d_ && !m_depth_image_) {
            ROS_WARN("No scan data available");
            return;
        }
        const auto [ok, rotation, translation] = GetSensorPose(time);
        if (!ok) {
            ROS_WARN("Failed to get sensor pose");
            return;
        }

        const bool in_local = m_setting_.scan_in_local_frame;
        bool is_point = false;
        Eigen::MatrixXd scan;
        switch (m_scan_type_) {
            case ScanType::Laser: {
                if (!m_lidar_scan_2d_) {
                    ROS_WARN("No laser scan data available");
                    return;
                }
                scan = Eigen::Map<const Eigen::VectorXf>(
                           m_lidar_scan_2d_->ranges.data(),
                           m_lidar_scan_2d_->ranges.size())
                           .cast<double>();
                is_point = false;
                m_lidar_scan_2d_.reset();
                break;
            }
            case ScanType::PointCloud: {
                if (!m_lidar_scan_3d_) {
                    ROS_WARN("No point cloud data available");
                    return;
                }
                uint32_t num_points = m_lidar_scan_3d_->width * m_lidar_scan_3d_->height;
                const uint8_t& data_type = m_lidar_scan_3d_->fields[0].datatype;
                if (data_type == sensor_msgs::PointField::FLOAT32) {
                    scan = Eigen::Map<const Eigen::Matrix3X<float>>(
                               reinterpret_cast<const float*>(m_lidar_scan_3d_->data.data()),
                               3,
                               num_points)
                               .cast<double>();
                } else if (data_type == sensor_msgs::PointField::FLOAT64) {
                    scan = Eigen::Map<const Eigen::Matrix3X<double>>(
                        reinterpret_cast<const double*>(m_lidar_scan_3d_->data.data()),
                        3,
                        num_points);
                } else {
                    ROS_WARN("Unsupported point cloud data type");
                    return;
                }
                is_point = true;
                m_lidar_scan_3d_.reset();
                break;
            }
            case ScanType::Depth: {
                if (!m_depth_image_) {
                    ROS_WARN("No depth image data available");
                    return;
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
                    ROS_WARN(
                        "Unsupported depth image encoding %s",
                        m_depth_image_->encoding.c_str());
                    return;
                }
                scan.array() *= m_setting_.depth_scale;  // convert to meters
                m_depth_image_.reset();
                break;
            }
        }
        bool success = m_sdf_mapping_->Update(rotation, translation, scan, is_point, in_local);
        if (!success) { ROS_WARN("Failed to update SDF mapping"); }
    }

    // --- service handler: runs Test() on the current map ---
    bool
    CallbackSdfQuery(
        erl_sdf_mapping::SdfQuery::Request& req,
        erl_sdf_mapping::SdfQuery::Response& res) {

        if (!m_sdf_mapping_) {
            ROS_WARN("SDF mapping is not initialized");
            return false;
        }

        const int dim = m_sdf_mapping_->map_dim;
        if (dim == 2) {
            if (m_is_double_) {
                return GetQueryResponse<double, 2>(req, res);
            } else {
                return GetQueryResponse<float, 2>(req, res);
            }
        }
        if (dim == 3) {
            if (m_is_double_) {
                return GetQueryResponse<double, 3>(req, res);
            } else {
                return GetQueryResponse<float, 3>(req, res);
            }
        }
        ROS_WARN("Unknown map dimension %d", dim);
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
            ROS_WARN("Wrong Dtype or Dim for GetQueryResponse");
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

        bool ok = m_sdf_mapping_->Predict(positions, distances, gradients, variances, covariances);
        res.success = ok;
        if (!ok) { return false; }

        // store the results in the response
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
            std::memcpy(
                reinterpret_cast<char*>(res.gradients.data()),
                reinterpret_cast<const char*>(gradients.data()),
                n * Dim * sizeof(double));
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
            ROS_WARN("SDF mapping is not initialized");
            res.success = false;
            return false;
        }
        if (req.name.empty()) {
            ROS_WARN("Map file name is empty");
            res.success = false;
            return false;
        }
        std::filesystem::path map_file = req.name;
        map_file = std::filesystem::absolute(map_file);
        std::filesystem::create_directories(map_file.parent_path());
        res.success = Serialization<AbstractGpSdfMapping>::Write(map_file, m_sdf_mapping_.get());
        return true;
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
        ERL_YAML_SAVE_ATTR(node, setting, use_odom);
        ERL_YAML_SAVE_ATTR(node, setting, odom_topic);
        ERL_YAML_SAVE_ATTR(node, setting, odom_queue_size);
        ERL_YAML_SAVE_ATTR(node, setting, world_frame);
        ERL_YAML_SAVE_ATTR(node, setting, sensor_frame);
        ERL_YAML_SAVE_ATTR(node, setting, scan_topic);
        ERL_YAML_SAVE_ATTR(node, setting, scan_type);
        ERL_YAML_SAVE_ATTR(node, setting, scan_in_local_frame);
        return node;
    }

    static bool
    decode(const YAML::Node& node, SdfMappingNodeSetting& setting) {
        ERL_YAML_LOAD_ATTR(node, setting, surface_mapping_setting_type);
        ERL_YAML_LOAD_ATTR(node, setting, surface_mapping_setting_file);
        ERL_YAML_LOAD_ATTR(node, setting, sdf_mapping_setting_type);
        ERL_YAML_LOAD_ATTR(node, setting, sdf_mapping_setting_file);
        ERL_YAML_LOAD_ATTR(node, setting, use_odom);
        ERL_YAML_LOAD_ATTR(node, setting, odom_topic);
        ERL_YAML_LOAD_ATTR(node, setting, odom_queue_size);
        ERL_YAML_LOAD_ATTR(node, setting, world_frame);
        ERL_YAML_LOAD_ATTR(node, setting, sensor_frame);
        ERL_YAML_LOAD_ATTR(node, setting, scan_topic);
        ERL_YAML_LOAD_ATTR(node, setting, scan_type);
        ERL_YAML_LOAD_ATTR(node, setting, scan_in_local_frame);
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
