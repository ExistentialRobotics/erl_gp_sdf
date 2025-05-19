#include "erl_sdf_mapping/gp_occ_surface_mapping.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping.hpp"
#include "erl_sdf_mapping/SdfQuery.h"

#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Vector3.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <tf/transform_datatypes.h>

using Dtype = float;
static constexpr int Dim = 2;
// alias your GP‐based surface+SDF mapping types:
using SurfaceMapping = erl::sdf_mapping::GpOccSurfaceMapping<Dtype, Dim>;
using SdfMapping = erl::sdf_mapping::GpSdfMapping<Dtype, Dim, SurfaceMapping>;

// param list:
//  - surface_mapping_config: path to the yaml file for surface mapping
//  - sdf_mapping_config: path to the yaml file for SDF mapping
//  - use_odom: whether to use the odometry topic to get the sensor pose (default: false)
//  - odom_topic: name of the odometry topic (default: "/jackal_velocity_controller/odom")
//  - world_frame: name of the world frame (default: "map")
//  - sensor_frame: name of the sensor frame (default: "front_laser")
//  - scan_topic: name of the laser scan topic (default: "/front/scan")

class SdfMappingNode {
public:
    SdfMappingNode(ros::NodeHandle& nh)
        : nh_(nh) {
        // get config files
        std::string surface_cfg, sdf_cfg;
        nh_.param<std::string>("surface_mapping_config", surface_cfg, "");
        nh_.param<std::string>("sdf_mapping_config", sdf_cfg, "");
        if (surface_cfg.empty() || sdf_cfg.empty()) {
            ROS_FATAL("You must set ~surface_mapping_config and ~sdf_mapping_config");
            ros::shutdown();
            return;
        }

        // load the surface mapping config
        auto surf_setting = std::make_shared<SurfaceMapping::Setting>();
        if (!surf_setting->FromYamlFile(surface_cfg)) {
            ROS_FATAL("Failed to load %s", surface_cfg.c_str());
            ros::shutdown();
            return;
        }
        // create the surface mapping object
        surface_mapping_ = std::make_shared<SurfaceMapping>(surf_setting);

        // load the SDF mapping config
        auto sdf_setting = std::make_shared<SdfMapping::Setting>();
        if (!sdf_setting->FromYamlFile(sdf_cfg)) {
            ROS_FATAL("Failed to load %s", sdf_cfg.c_str());
            ros::shutdown();
            return;
        }
        // create the SDF mapping object
        sdf_mapping_ = std::make_shared<SdfMapping>(sdf_setting, surface_mapping_);

        std::string odom_topic;
        nh_.param<std::string>("odom_topic", odom_topic, "/jackal_velocity_controller/odom");

        odom_sub_ = nh_.subscribe(odom_topic, 1, &SdfMappingNode::odomCallback, this);
        scan_sub_ = nh_.subscribe("/front/scan", 1, &SdfMappingNode::scanCallback, this);

        // why is it called advertise?
        service_ = nh_.advertiseService("sdf_query", &SdfMappingNode::sdfQueryCallback, this);

        ROS_INFO("SdfMappingNode ready. Waiting for scans + queries...");
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber odom_sub_, scan_sub_;
    ros::ServiceServer service_;

    std::shared_ptr<SurfaceMapping> surface_mapping_;
    std::shared_ptr<SdfMapping> sdf_mapping_;

    nav_msgs::Odometry::ConstPtr latest_odom_;
    sensor_msgs::LaserScan::ConstPtr latest_scan_;

    // --- callbacks to collect pose+scan and then update the map ---
    void
    odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        latest_odom_ = msg;
        tryUpdate();
    }

    void
    scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
        latest_scan_ = msg;
        tryUpdate();
    }

    void
    tryUpdate() {
        if (!latest_odom_ || !latest_scan_) return;

        // extract (x,y,yaw)
        double x = latest_odom_->pose.pose.position.x;
        double y = latest_odom_->pose.pose.position.y;
        double yaw = tf::getYaw(latest_odom_->pose.pose.orientation);

        // build Eigen rotation & translation
        Eigen::Rotation2Df rot2d(yaw);
        Eigen::Matrix<Dtype, Dim, Dim> R = rot2d.toRotationMatrix();
        Eigen::Matrix<Dtype, Dim, 1> t;
        t << x, y;
        // basically the SE(3) matrix from the test?
        //  copy ranges into an Eigen vector
        const auto& scan = *latest_scan_;
        Eigen::Matrix<Dtype, Eigen::Dynamic, 1> ranges(scan.ranges.size());
        for (size_t i = 0; i < scan.ranges.size(); ++i)
            ranges[i] = static_cast<Dtype>(scan.ranges[i]);

        // feed the new scan into the GP‐based mapper and do GPSdf update
        if (!sdf_mapping_->Update(R, t, ranges))
            ROS_WARN("GpSdfMapping::Update failed for this scan.");

        // reset so we only process each pair once
        latest_odom_.reset();
        latest_scan_.reset();
    }

    // --- service handler: runs Test() on the current map ---
    bool
    sdfQueryCallback(
        erl_sdf_mapping::SdfQuery::Request& req,
        erl_sdf_mapping::SdfQuery::Response& res) {
        using Distances = typename SdfMapping::Distances;
        using Gradients = typename SdfMapping::Gradients;
        using Variances = typename SdfMapping::Variances;
        using Covariances = typename SdfMapping::Covariances;

        size_t N = req.query_points.size();
        // Pre-allocate response arrays
        res.signed_distances.resize(N);
        res.gradients.resize(N);
        res.variances.resize(N);
        res.successes.resize(N);
        res.messages.resize(N);

        for (size_t i = 0; i < N; ++i) {
            // unpack input
            const auto& pt = req.query_points[i];
            Eigen::Matrix<Dtype, Dim, 1> query;
            query << pt.point.x, pt.point.y;

            // prepare containers
            Distances distance;
            distance.resize(1);
            Gradients gradient;       // fixed size
            Variances variances;      // fixed size
            Covariances covariances;  // fixed size

            // do the query
            bool ok = sdf_mapping_->Test(query, distance, gradient, variances, covariances);

            // fill response
            res.signed_distances[i] = distance(0);
            res.gradients[i].x = gradient(0);
            res.gradients[i].y = gradient(1);
            res.gradients[i].z = 0.0;
            res.variances[i] = variances(0);
            res.successes[i] = ok;
            res.messages[i] = ok ? "OK" : "Test() failed";
        }

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
