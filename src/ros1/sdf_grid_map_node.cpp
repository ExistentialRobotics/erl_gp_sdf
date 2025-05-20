#include "erl_common/yaml.hpp"
#include "erl_sdf_mapping/SdfQuery.h"

#include <geometry_msgs/PointStamped.h>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <ros/ros.h>

using namespace erl::common;

struct SdfGridMapNodeSetting : Yamlable<SdfGridMapNodeSetting> {
    double resolution = 0.1;
    int cells = 101;
    std::string service_name = "/sdf_mapping_node/sdf_query";
};

class SdfGridMapNode {
public:
    SdfGridMapNode() {
        // Param
        ros::NodeHandle pnh("~");
        pnh.param("resolution", resolution_, 0.1);
        pnh.param("cells", cells_, 101);
        std::string service_name;
        pnh.param("sdf_service", service_name, std::string("/sdf_mapping_node/sdf_query"));

        // TODO querry points. Currently it is a bit naive resolution * cell will give us lengths.
        // Initializating properly with a config might be better.
        query_points_ = makeQuery(resolution_, cells_);

        sdf_client_ = nh_.serviceClient<erl_sdf_mapping::SdfQuery>(service_name);
        map_pub_ = nh_.advertise<grid_map_msgs::GridMap>("sdf_grid_map", 1, true);

        timer_ = nh_.createTimer(ros::Duration(0.5), &SdfGridMapNode::timerCallback, this);
        ROS_INFO("SdfGridMapNode initialized, querying %zu points...", query_points_.size());
    }

private:
    ros::NodeHandle nh_;
    ros::ServiceClient sdf_client_;
    ros::Publisher map_pub_;
    ros::Timer timer_;
    double resolution_;
    int cells_;
    std::vector<geometry_msgs::PointStamped> query_points_;

    // building the square grid (for now) to get geo messages to push
    std::vector<geometry_msgs::PointStamped>
    makeQuery(double res, int n) {
        std::vector<geometry_msgs::PointStamped> pts;
        int half = (n - 1) / 2;
        pts.reserve(n * n);
        for (int i = -half; i <= half; ++i) {
            for (int j = -half; j <= half; ++j) {
                geometry_msgs::PointStamped p;
                p.header.frame_id = "map";
                p.header.stamp = ros::Time::now();
                p.point.x = i * res;
                p.point.y = j * res;
                p.point.z = 0.0;
                pts.push_back(p);
            }
        }
        return pts;
    }

    void
    timerCallback(const ros::TimerEvent&) {
        erl_sdf_mapping::SdfQuery srv;
        srv.request.query_points = query_points_;
        if (!sdf_client_.call(srv)) {
            ROS_WARN("Failed to call %s", sdf_client_.getService().c_str());
            return;
        }

        grid_map::GridMap map({"distance"});
        map.setFrameId("map");
        grid_map::Length length(cells_ * resolution_, cells_ * resolution_);
        map.setGeometry(length, resolution_, grid_map::Position(0.0, 0.0));

        for (size_t i = 0; i < srv.response.signed_distances.size(); ++i) {
            const auto& ps = query_points_[i];
            grid_map::Position pos(ps.point.x, ps.point.y);
            grid_map::Index idx;
            if (map.getIndex(pos, idx)) {
                map.at("distance", idx) = srv.response.signed_distances[i];
            }
        }

        // Publish
        grid_map_msgs::GridMap msg;
        grid_map::GridMapRosConverter::toMessage(map, msg);
        map_pub_.publish(msg);
    }
};

int
main(int argc, char** argv) {
    ros::init(argc, argv, "sdf_grid_map_node");
    SdfGridMapNode node;
    ros::spin();
    return 0;
}
