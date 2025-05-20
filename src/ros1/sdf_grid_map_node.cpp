#include "erl_sdf_mapping/SdfQuery.h"

#include <geometry_msgs/Vector3.h>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>

struct SdfGridMapNodeSetting {
    // resolution of the grid map.
    double resolution = 0.1;
    // number of cells in the grid map along the x-axis.
    int x_cells = 101;
    // number of cells in the grid map along the y-axis.
    int y_cells = 101;
    // the z coordinate of the grid map.
    double z = 0.0;
    // if true, the gradient of the SDF is published.
    bool publish_gradient = false;
    // if true, the variance of the SDF is published.
    bool publish_sdf_variance = false;
    // if true, the variance of the SDF gradient is published.
    bool publish_gradient_variance = false;
    // if true, the covariance between the SDF and the gradient is published.
    bool publish_covariance = false;
    // the frequency of publishing the grid map.
    double publish_freq = 2;
    // if true, the grid map is moved with the frame.
    bool attached_to_frame = false;
    // the name of the frame to attach the grid map to.
    std::string attached_frame = "map";
    // the name of the world frame, used when attached_to_frame is true.
    std::string world_frame = "map";
    // the name of the service to query the SDF.
    std::string service_name = "/sdf_mapping_node/sdf_query";
    // the name of the topic to publish the grid map.
    std::string map_topic_name = "sdf_grid_map";
};

class SdfGridMapNode {
    SdfGridMapNodeSetting m_setting_;
    ros::NodeHandle m_nh_;
    ros::ServiceClient m_sdf_client_;
    ros::Publisher m_map_pub_;
    ros::Timer m_timer_;
    tf2_ros::Buffer m_tf_buffer_;
    tf2_ros::TransformListener m_tf_listener_{m_tf_buffer_};
    std::vector<geometry_msgs::Vector3> m_query_points_;

public:
    SdfGridMapNode(ros::NodeHandle& nh)
        : m_nh_(nh) {
        if (!LoadParameters()) {
            ROS_FATAL("Failed to load parameters");
            ros::shutdown();
            return;
        }

        InitQueryPoints();

        m_sdf_client_ = m_nh_.serviceClient<erl_sdf_mapping::SdfQuery>(m_setting_.service_name);
        m_map_pub_ = m_nh_.advertise<grid_map_msgs::GridMap>(m_setting_.map_topic_name, 1, true);
        m_timer_ = m_nh_.createTimer(ros::Duration(0.5), &SdfGridMapNode::CallbackTimer, this);
        ROS_INFO("SdfGridMapNode initialized");
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
        if (!LoadParam("resolution", m_setting_.resolution)) { return false; }
        if (!LoadParam("x_cells", m_setting_.x_cells)) { return false; }
        if (!LoadParam("y_cells", m_setting_.y_cells)) { return false; }
        if (!LoadParam("z", m_setting_.z)) { return false; }
        if (!LoadParam("publish_gradient", m_setting_.publish_gradient)) { return false; }
        if (!LoadParam("publish_sdf_variance", m_setting_.publish_sdf_variance)) { return false; }
        if (!LoadParam("publish_gradient_variance", m_setting_.publish_gradient_variance)) {
            return false;
        }
        if (!LoadParam("publish_covariance", m_setting_.publish_covariance)) { return false; }
        if (!LoadParam("publish_freq", m_setting_.publish_freq)) { return false; }
        if (!LoadParam("attached_to_frame", m_setting_.attached_to_frame)) { return false; }
        if (!LoadParam("attached_frame", m_setting_.attached_frame)) { return false; }
        if (!LoadParam("service_name", m_setting_.service_name)) { return false; }
        if (!LoadParam("map_topic_name", m_setting_.map_topic_name)) { return false; }
        if (m_setting_.resolution <= 0) {
            ROS_WARN("Resolution must be positive");
            return false;
        }
        if (m_setting_.x_cells <= 0) {
            ROS_WARN("X cells must be positive");
            return false;
        }
        if (m_setting_.x_cells % 2 == 0) {
            m_setting_.x_cells += 1;
            ROS_WARN("X cells must be odd, set to %d", m_setting_.x_cells);
        }
        if (m_setting_.y_cells <= 0) {
            ROS_WARN("Y cells must be positive");
            return false;
        }
        if (m_setting_.y_cells % 2 == 0) {
            m_setting_.y_cells += 1;
            ROS_WARN("Y cells must be odd, set to %d", m_setting_.y_cells);
        }
        if (m_setting_.publish_freq <= 0) {
            ROS_WARN("Publish frequency must be positive");
            return false;
        }
        if (m_setting_.attached_to_frame) {
            if (m_setting_.attached_frame.empty()) {
                ROS_WARN("Attached frame is empty but attached_to_frame is true");
                return false;
            }
            if (m_setting_.world_frame.empty()) {
                ROS_WARN("World frame is empty but attached_to_frame is true");
                return false;
            }
        }
        if (m_setting_.service_name.empty()) {
            ROS_WARN("Service name is empty");
            return false;
        }
        if (m_setting_.map_topic_name.empty()) {
            ROS_WARN("Map topic name is empty");
            return false;
        }
        return true;
    }

    void
    InitQueryPoints() {
        m_query_points_.clear();
        m_query_points_.reserve(m_setting_.x_cells * m_setting_.y_cells);
        int half_x = (m_setting_.x_cells - 1) / 2;
        int half_y = (m_setting_.y_cells - 1) / 2;
        for (int j = -half_y; j <= half_y; ++j) {      // y-axis
            for (int i = -half_x; i <= half_x; ++i) {  // x-axis (column major)
                geometry_msgs::Vector3 p;
                p.x = static_cast<double>(i) * m_setting_.resolution;
                p.y = static_cast<double>(j) * m_setting_.resolution;
                p.z = m_setting_.z;
                m_query_points_.push_back(p);
            }
        }
        ROS_INFO(
            "Query points initialized, %d x %d points",
            m_setting_.x_cells,
            m_setting_.y_cells);
    }

    void
    CallbackTimer(const ros::TimerEvent&) {
        erl_sdf_mapping::SdfQuery srv;
        if (m_setting_.attached_to_frame) {
            geometry_msgs::TransformStamped transform_stamped;
            try {
                transform_stamped = m_tf_buffer_.lookupTransform(
                    m_setting_.attached_frame,
                    "map",
                    ros::Time(0),
                    ros::Duration(1.0));
            } catch (tf2::TransformException& ex) {
                ROS_WARN("%s", ex.what());
                return;
            }
            const double x = transform_stamped.transform.translation.x;
            const double y = transform_stamped.transform.translation.y;
            srv.request.query_points.clear();
            srv.request.query_points.reserve(m_query_points_.size());
            for (auto& ps: m_query_points_) {
                geometry_msgs::Vector3 p;
                p.x = ps.x + x;
                p.y = ps.y + y;
                p.z = ps.z;
                srv.request.query_points.emplace_back(std::move(p));
            }
        } else {
            srv.request.query_points = m_query_points_;
        }

        if (!m_sdf_client_.call(srv)) {
            ROS_WARN("Failed to call %s", m_sdf_client_.getService().c_str());
            return;
        }

        auto& ans = srv.response;

        grid_map::GridMap map;
        if (m_setting_.attached_to_frame) {
            map.setFrameId(m_setting_.attached_frame);
        } else {
            map.setFrameId(m_setting_.world_frame);
        }
        map.setGeometry(
            grid_map::Length(
                static_cast<double>(m_setting_.x_cells) * m_setting_.resolution,
                static_cast<double>(m_setting_.y_cells) * m_setting_.resolution),
            m_setting_.resolution,
            grid_map::Position(0.0, 0.0));

        const auto n = static_cast<int>(ans.signed_distances.size());
        const auto map_size = map.getSize();
        if (n != map_size[0] * map_size[1]) {
            ROS_WARN(
                "Query points size %d does not match map size %d",
                n,
                map_size[0] * map_size[1]);
            return;
        }
        // SDF
        map.add(
            "sdf",
            Eigen::Map<const Eigen::MatrixXd>(ans.signed_distances.data(), map_size[0], map_size[1])
                .cast<float>());
        // gradient
        if (m_setting_.publish_gradient) {  // dim layers
            if (ans.compute_gradient) {
                Eigen::Map<const Eigen::MatrixXd> gradients(
                    reinterpret_cast<const double*>(ans.gradients.data()),
                    3,
                    n);
                static const char* gradient_names[3] = {"gradient_x", "gradient_y", "gradient_z"};
                for (int i = 0; i < ans.dim; ++i) {
                    Eigen::VectorXf grad_i = gradients.row(i).transpose().cast<float>();
                    Eigen::Map<Eigen::MatrixXf> grad_map(grad_i.data(), map_size[0], map_size[1]);
                    map.add(gradient_names[i], grad_map);
                }
            } else {
                ROS_WARN("Gradient is not computed");
            }
        }
        Eigen::Map<const Eigen::MatrixXd> variances(
            reinterpret_cast<const double*>(ans.variances.data()),
            ans.compute_gradient_variance ? ans.dim + 1 : 1,
            n);
        // SDF variance
        if (m_setting_.publish_sdf_variance) {  // 1 layer
            if (ans.compute_gradient_variance) {
                Eigen::VectorXf sdf_variance = variances.row(0).transpose().cast<float>();
                Eigen::Map<Eigen::MatrixXf> sdf_variance_map(
                    sdf_variance.data(),
                    map_size[0],
                    map_size[1]);
                map.add("sdf_variance", sdf_variance_map);
                // layer_names.push_back("sdf_variance");
            } else {
                map.add(
                    "sdf_variance",
                    Eigen::Map<const Eigen::MatrixXd>(
                        ans.variances.data(),
                        map_size[0],
                        map_size[1])
                        .cast<float>());
            }
        }
        // gradient variance
        if (m_setting_.publish_gradient_variance) {  // dim layers
            if (ans.compute_gradient_variance) {
                static const char* gradient_variance_names[3] = {
                    "gradient_variance_x",
                    "gradient_variance_y",
                    "gradient_variance_z"};
                for (int i = 0; i < ans.dim; ++i) {
                    Eigen::VectorXf grad_variance_i =
                        variances.row(i + 1).transpose().cast<float>();
                    Eigen::Map<Eigen::MatrixXf> grad_variance_map(
                        grad_variance_i.data(),
                        map_size[0],
                        map_size[1]);
                    map.add(gradient_variance_names[i], grad_variance_map);
                }

            } else {
                ROS_WARN("Gradient variance is not computed");
            }
        }
        // covariance
        if (m_setting_.publish_covariance) {  // dim * (dim + 1) / 2 layers
            if (ans.compute_covariance) {
                Eigen::Map<const Eigen::MatrixXd> covariances(
                    ans.covariances.data(),
                    ans.dim * (ans.dim + 1) / 2,
                    n);
                if (ans.dim == 2) {
                    static const char* covariance_names[3] = {
                        "covariance_gx_sdf",
                        "covariance_gy_sdf",
                        "covariance_gy_gx"};
                    for (long i = 0; i < covariances.rows(); ++i) {
                        Eigen::VectorXf covariance_i = covariances.row(i).transpose().cast<float>();
                        Eigen::Map<Eigen::MatrixXf> covariance_map(
                            covariance_i.data(),
                            map_size[0],
                            map_size[1]);
                        map.add(covariance_names[i], covariance_map);
                    }
                } else if (ans.dim == 3) {
                    static const char* covariance_names[6] = {
                        "covariance_gx_sdf",
                        "covariance_gy_sdf",
                        "covariance_gz_sdf",
                        "covariance_gy_gx",
                        "covariance_gz_gx",
                        "covariance_gz_gy"};
                    for (long i = 0; i < covariances.rows(); ++i) {
                        Eigen::VectorXf covariance_i = covariances.row(i).transpose().cast<float>();
                        Eigen::Map<Eigen::MatrixXf> covariance_map(
                            covariance_i.data(),
                            map_size[0],
                            map_size[1]);
                        map.add(covariance_names[i], covariance_map);
                    }
                } else {
                    ROS_WARN("Unknown dimension %d", ans.dim);
                }
            } else {
                ROS_WARN("Covariance is not computed");
            }
        }

        // Publish
        grid_map_msgs::GridMap msg;
        grid_map::GridMapRosConverter::toMessage(map, msg);
        m_map_pub_.publish(msg);
    }
};

int
main(int argc, char** argv) {
    ros::init(argc, argv, "sdf_grid_map_node");
    ros::NodeHandle nh("~");  // ~: shorthand for the private namespace
    SdfGridMapNode node(nh);
    ros::spin();
    return 0;
}
