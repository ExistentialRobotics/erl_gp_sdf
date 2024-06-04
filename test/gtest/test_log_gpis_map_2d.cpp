#include "erl_common/test_helper.hpp"
#include "erl_geometry/gazebo_room.hpp"
#include "erl_sdf_mapping/gpis/log_gpis_map_2d.hpp"

#include <filesystem>

using namespace erl::common;
using namespace erl::sdf_mapping::gpis;

TEST(ERL_SDF_MAPPING, LogGpisMap2D) {
    std::filesystem::path file_path = __FILE__;
    std::filesystem::path dir_path = file_path.parent_path();

    auto setting = std::make_shared<LogGpisMap2D::Setting>();
    setting->quadtree->min_half_area_size = 0.025;
    setting->quadtree->cluster_half_area_size = 0.1;
    setting->node_container->min_squared_distance = 0.025 * 0.025;
    LogGpisMap2D log_gpis_map(setting);

    std::cout << *log_gpis_map.GetSetting() << std::endl;

    std::cout.precision(5);
    std::cout << std::scientific;

    double t_train_ans = 0;
    int cnt = 0;
    std::stringstream ss;
    const char *train_dat_file = "double/train.dat";
    auto train_data_loader = erl::geometry::GazeboRoom::TrainDataLoader((dir_path / train_dat_file).c_str());
    for (auto df: train_data_loader) {
        ss.str(std::string());
        ss << "LogGpisMap2D-Train[" << cnt << ']';
        t_train_ans += ReportTime<std::chrono::microseconds>(ss.str().c_str(), 0, true, [&]() { log_gpis_map.Update(df.angles, df.distances, df.pose_numpy); });
        cnt++;
        std::cout << "===============================================" << std::endl;
    }

    t_train_ans /= static_cast<double>(train_data_loader.size());

    const char *test_dat_file = "double/test.dat";
    auto df = erl::geometry::GazeboRoom::TestDataFrame((dir_path / test_dat_file).c_str());
    Eigen::VectorXd distance_ans, distance_variance_ans;
    Eigen::Matrix2Xd gradient_ans, gradient_variance_ans;
    // gradient_variance_ans;
    double t_test_ans = ReportTime<std::chrono::milliseconds>("LogGpisMap2D-test", 10, false, [&]() {
        log_gpis_map.Test(df.positions, distance_ans, gradient_ans, distance_variance_ans, gradient_variance_ans);
    });
    double t_per_point = t_test_ans / static_cast<double>(df.positions.cols() * 1000);  // us
    std::cout << "===============================================" << std::endl;

    Logging::Info("Average training time:");
    std::cout << "LogGpisMap2D: " << t_train_ans << " us" << std::endl;
    Logging::Info("Average testing time:");
    std::cout << "LogGpisMap2D: " << t_test_ans << " ms for " << df.positions.cols() << " points, " << t_per_point << " us per point" << std::endl;
}

#if defined(ERL_ROS_VERSION_1)
int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
