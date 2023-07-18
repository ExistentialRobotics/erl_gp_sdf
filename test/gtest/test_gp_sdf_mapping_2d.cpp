#include "erl_geometry/house_expo.hpp"
#include "erl_geometry/lidar_2d.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping_2d.hpp"
#include "erl_geometry/occupancy_quadtree_drawer.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_2d.hpp"
#include "erl_common/csv.hpp"
#include "erl_geometry/gazebo_room.hpp"
#include "erl_common/random.hpp"
#include <boost/program_options.hpp>
#include <gtest/gtest.h>
#include <filesystem>

static std::filesystem::path g_file_path = __FILE__;
static std::filesystem::path g_dir_path = g_file_path.parent_path();
static std::string g_window_name = "ERL_SDF_MAPPING";

struct Options {
    std::string gazebo_train_file = (g_dir_path / "train.dat").string();
    std::string gazebo_test_file = (g_dir_path / "test.dat").string();
    std::string house_expo_map_file = (g_dir_path / "house_expo_room_1451.json").string();
    std::string house_expo_traj_file = (g_dir_path / "house_expo_room_1451.csv").string();
    bool use_gazebo_data = false;
    bool visualize = false;
    bool hold = false;
};

static Options g_options;

TEST(ERL_SDF_MAPPING, GpSdfMapping2D) {
    long max_update_cnt;
    std::vector<Eigen::VectorXd> train_angles;
    std::vector<Eigen::VectorXd> train_ranges;
    std::vector<Eigen::Matrix23d> train_poses;
    Eigen::Vector2d map_min(0, 0);
    Eigen::Vector2d map_max(0, 0);
    Eigen::Vector2d map_resolution(0.01, 0.01);
    Eigen::Vector2i map_padding(100, 100);
    Eigen::Matrix2Xd cur_traj;

    if (g_options.use_gazebo_data) {
        auto train_data_loader = erl::geometry::GazeboRoom::TrainDataLoader(g_options.gazebo_train_file.c_str());
        max_update_cnt = long(train_data_loader.size());
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        int i = 0;
        for (auto &df: train_data_loader) {
            train_angles.push_back(df.angles);
            train_ranges.push_back(df.distances);
            train_poses.emplace_back(df.pose_numpy);
            cur_traj.col(i) << df.pose_numpy(0, 2), df.pose_numpy(1, 2);
            double &x = cur_traj(0, i);
            double &y = cur_traj(1, i);
            if (x < map_min[0]) { map_min[0] = x; }
            if (x > map_max[0]) { map_max[0] = x; }
            if (y < map_min[1]) { map_min[1] = y; }
            if (y > map_max[1]) { map_max[1] = y; }
            map_min[0] -= 0.15;
            map_min[1] -= 0.4;
            map_max[0] += 0.2;
            map_max[1] += 0.15;
            map_resolution.array() = 0.02;
            map_padding.setZero();
            i++;
        }
    } else {
        erl::geometry::HouseExpoMap house_expo_map(g_options.house_expo_map_file.c_str(), 0.2);
        map_min = house_expo_map.GetMeterSpace()->GetSurface()->vertices.rowwise().minCoeff();
        map_max = house_expo_map.GetMeterSpace()->GetSurface()->vertices.rowwise().maxCoeff();
        erl::geometry::Lidar2D lidar(house_expo_map.GetMeterSpace());
        double angle_min = -M_PI;
        double angle_max = M_PI;
        double res = 0.5 * M_PI / 180.0;
        bool scan_in_parallel = true;
        lidar.SetNumLines(int((angle_max - angle_min) / res));
        auto trajectory =
            erl::common::LoadAndCastCsvFile<double>(g_options.house_expo_traj_file.c_str(), [](const std::string &str) -> double { return std::stod(str); });
        max_update_cnt = long(trajectory.size());
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        for (long i = 0; i < max_update_cnt; i++) {
            std::vector<double> &waypoint = trajectory[i];
            cur_traj.col(i) << waypoint[0], waypoint[1];
            lidar.SetTranslation(cur_traj.col(i));
            lidar.SetRotation(waypoint[2]);
            auto lidar_ranges = lidar.Scan(scan_in_parallel);
            lidar_ranges += erl::common::GenerateGaussianNoise(lidar_ranges.size(), 0.0, 0.01);
            train_angles.push_back(lidar.GetAngles());
            train_ranges.push_back(lidar_ranges);
            train_poses.emplace_back(lidar.GetPose().topRows<2>());
        }
    }

    auto surface_mapping = std::make_shared<erl::sdf_mapping::GpOccSurfaceMapping2D>();
    erl::sdf_mapping::GpSdfMapping2D sdf_mapping(surface_mapping);
    std::cout << surface_mapping->GetSetting()->AsYamlString() << std::endl;
    std::cout << sdf_mapping.GetSetting()->AsYamlString() << std::endl;
    using OccupancyQuadtreeDrawer = erl::geometry::OccupancyQuadtreeDrawer<erl::sdf_mapping::SurfaceMappingQuadtree>;
    auto drawer_setting = std::make_shared<OccupancyQuadtreeDrawer::Setting>();
    drawer_setting->area_min = map_min;
    drawer_setting->area_max = map_max;
    drawer_setting->resolution = map_resolution[0];
    drawer_setting->padding = map_padding[0];
    drawer_setting->border_color = cv::Scalar(255, 0, 0, 255);
    auto drawer = std::make_shared<OccupancyQuadtreeDrawer>(drawer_setting);
    std::vector<std::pair<cv::Point, cv::Point>> arrowed_lines;
    drawer->SetDrawTreeCallback([&](const OccupancyQuadtreeDrawer *self, cv::Mat &img, erl::sdf_mapping::SurfaceMappingQuadtree::TreeIterator &it) {
        unsigned int cluster_depth = surface_mapping->GetQuadtree()->GetTreeDepth() - surface_mapping->GetClusterLevel();
        auto grid_map_info = self->GetGridMapInfo();
        if (it.GetDepth() == cluster_depth) {
            Eigen::Vector2i position_px = grid_map_info->MeterToPixelForPoints(Eigen::Vector2d(it.GetX(), it.GetY()));
            cv::Point position_px_cv(position_px[0], position_px[1]);
            cv::circle(img, position_px_cv, 2, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
            return;
        }
        if (it->GetSurfaceData() == nullptr) { return; }
        Eigen::Vector2i position_px = grid_map_info->MeterToPixelForPoints(it->GetSurfaceData()->position);
        cv::Point position_px_cv(position_px[0], position_px[1]);
        cv::circle(img, cv::Point(position_px[0], position_px[1]), 1, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
        Eigen::Vector2i normal_px = grid_map_info->MeterToPixelForVectors(it->GetSurfaceData()->normal * 0.5);
        cv::Point arrow_end_px(position_px[0] + normal_px[0], position_px[1] + normal_px[1]);
        arrowed_lines.emplace_back(position_px_cv, arrow_end_px);
    });
    drawer->SetDrawLeafCallback([&](const OccupancyQuadtreeDrawer *self, cv::Mat &img, erl::sdf_mapping::SurfaceMappingQuadtree::LeafIterator &it) {
        if (it->GetSurfaceData() == nullptr) { return; }
        auto grid_map_info = self->GetGridMapInfo();
        Eigen::Vector2i position_px = grid_map_info->MeterToPixelForPoints(it->GetSurfaceData()->position);
        cv::Point position_px_cv(position_px[0], position_px[1]);
        cv::circle(img, position_px_cv, 1, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
        Eigen::Vector2i normal_px = grid_map_info->MeterToPixelForVectors(it->GetSurfaceData()->normal * 0.5);
        cv::Point arrow_end_px(position_px[0] + normal_px[0], position_px[1] + normal_px[1]);
        arrowed_lines.emplace_back(position_px_cv, arrow_end_px);
    });

    bool pixel_based = true;
    cv::Scalar trajectory_color(0, 0, 255, 255);
    cv::Mat img;
    bool drawer_connected = false;
    bool update_occupancy = surface_mapping->GetSetting()->update_occupancy;
    if (g_options.visualize) {
        if (update_occupancy) {
            drawer->DrawLeaves(img);
        } else {
            drawer->DrawTree(img);
        }
        cv::imshow(g_window_name, img);
        cv::waitKey(10);
    }
    if (g_options.hold) {
        if (!g_options.visualize) { cv::namedWindow(g_window_name, cv::WINDOW_AUTOSIZE); }
        std::cout << "Press any key to start" << std::endl;
        cv::waitKey(0);
    }
    auto grid_map_info = drawer->GetGridMapInfo();

    long t = 0;
    for (long i = 0; i < max_update_cnt; i++) {
        Eigen::VectorXd noise = erl::common::GenerateGaussianNoise(train_ranges[i].size(), 0.0, 0.01);

        auto t0 = std::chrono::high_resolution_clock::now();
        sdf_mapping.Update(train_angles[i], train_ranges[i], train_poses[i]);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        ERL_INFO("Update time: %ld ms.", dt);
        t += dt;

        if (g_options.visualize) {
            if (!drawer_connected) {
                drawer->SetQuadtree(surface_mapping->GetQuadtree());
                drawer_connected = true;
            }
            img.setTo(cv::Scalar(128, 128, 128, 255));
            arrowed_lines.clear();
            if (update_occupancy) {
                drawer->DrawLeaves(img);
            } else {
                drawer->DrawTree(img);
            }
            for (auto &[position_px_cv, arrow_end_px]: arrowed_lines) {
                cv::arrowedLine(img, position_px_cv, arrow_end_px, cv::Scalar(0, 0, 255, 255), 1, 8, 0, 0.1);
            }
            erl::common::DrawTrajectoryInplace(img, cur_traj.block(0, 0, 2, i), grid_map_info, trajectory_color, 2, pixel_based);
            cv::imshow(g_window_name, img);
            cv::waitKey(10);
        }
        std::cout << "=====================================" << std::endl;
    }
    ERL_INFO("Average update time: %ld ms.", t / max_update_cnt);

    // Test SDF Estimation
    bool c_stride = true;
    Eigen::Matrix2Xd positions_in = grid_map_info->GenerateMeterCoordinates(c_stride);
    Eigen::VectorXd distances_out, distance_variances_out;
    Eigen::Matrix2Xd gradients_out, gradient_variances_out;
    auto t0 = std::chrono::high_resolution_clock::now();
    sdf_mapping.Test(positions_in, distances_out, gradients_out, distance_variances_out, gradient_variances_out);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    double t_per_point = double(dt) / (double) positions_in.cols() * 1000;  // us
    ERL_INFO("Test time: %ld ms for %ld points, %f us per point.", dt, positions_in.cols(), t_per_point);

    if (g_options.visualize) {
        Eigen::MatrixXd distances_out_mat = distances_out.reshaped(grid_map_info->Height(), grid_map_info->Width());
        double min_distance = distances_out.minCoeff();
        double max_distance = distances_out.maxCoeff();
        Eigen::MatrixX8U distances_out_mat_normalized = ((distances_out_mat.array() - min_distance) / (max_distance - min_distance) * 255).cast<uint8_t>();
        cv::Mat src, dst;
        cv::eigen2cv(distances_out_mat_normalized, src);
        cv::flip(src, src, 0);  // flip along y axis
        cv::applyColorMap(src, dst, cv::COLORMAP_JET);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2BGRA);
        cv::addWeighted(dst, 0.5, img, 0.5, 0.0, dst);
        cv::imshow("distances_out", dst);
        cv::waitKey(100);

        Eigen::MatrixXd variances_out_mat = distance_variances_out.reshaped(grid_map_info->Height(), grid_map_info->Width());
        double min_variance = variances_out_mat.minCoeff();
        double max_variance = variances_out_mat.maxCoeff();
        Eigen::MatrixX8U variances_out_mat_normalized = ((variances_out_mat.array() - min_variance) / (max_variance - min_variance) * 255).cast<uint8_t>();
        cv::eigen2cv(variances_out_mat_normalized, src);
        cv::flip(src, src, 0);  // flip along y axis
        cv::applyColorMap(src, dst, cv::COLORMAP_JET);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2BGRA);
        cv::addWeighted(dst, 0.5, img, 0.5, 0.0, dst);
        cv::imshow("distance_variances_out", dst);
        cv::waitKey(100);
    }

    surface_mapping->GetQuadtree()->WriteBinary("tree.bt");
    ERL_ASSERT(surface_mapping->GetQuadtree()->Write("tree.ot"));
    if (g_options.hold) {
        std::cout << "Press any key to exit." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    } else {
        cv::waitKey(10000);  // 10 seconds
        cv::destroyAllWindows();
    }
}

int
main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    try {
        namespace po = boost::program_options;
        po::options_description desc;
        // clang-format off
        desc.add_options()
            ("help", "produce help message")
            ("use-gazebo-data", po::bool_switch(&g_options.use_gazebo_data)->default_value(g_options.use_gazebo_data), "Use Gazebo data instead of HouseExpo data")
            ("visualize", po::bool_switch(&g_options.visualize)->default_value(g_options.visualize), "Visualize the mapping")
            ("hold", po::bool_switch(&g_options.hold)->default_value(g_options.hold), "Hold the test until a key is pressed")
            (
                "house-expo-map-file",
                po::value<std::string>(&g_options.house_expo_map_file)->default_value(g_options.house_expo_map_file)->value_name("file"),
                "HouseExpo map file"
            )(
                "house-expo-traj-file",
                po::value<std::string>(&g_options.house_expo_traj_file)->default_value(g_options.house_expo_traj_file)->value_name("file"),
                "HouseExpo trajectory file"
            )(
                "gazebo-train-file",
                po::value<std::string>(&g_options.gazebo_train_file)->default_value(g_options.gazebo_train_file)->value_name("file"),
                "Gazebo train data file"
            )(
                "gazebo-test-file",
                po::value<std::string>(&g_options.gazebo_test_file)->default_value(g_options.gazebo_test_file)->value_name("file"),
                "Gazebo test data file");
        // clang-format on

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        if (vm.count("help")) {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl << desc << std::endl;
            return 0;
        }
        po::notify(vm);
    } catch (std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    g_window_name = argv[0];
    return RUN_ALL_TESTS();
}
