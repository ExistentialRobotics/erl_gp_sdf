#include "erl_common/csv.hpp"
#include "erl_common/progress_bar.hpp"
#include "erl_common/random.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_geometry/gazebo_room_2d.hpp"
#include "erl_geometry/house_expo_map.hpp"
#include "erl_geometry/lidar_2d.hpp"
#include "erl_geometry/occupancy_quadtree_drawer.hpp"
#include "erl_geometry/ucsd_fah_2d.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping_2d.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_2d.hpp"

#include <boost/program_options.hpp>

#include <filesystem>

struct Options {
    std::string gazebo_train_file;
    std::string house_expo_map_file;
    std::string house_expo_traj_file;
    std::string ucsd_fah_2d_file;
    std::string surface_mapping_config_file;
    std::string output_dir;
    bool use_gazebo_room_2d = false;
    bool use_house_expo_lidar_2d = false;
    bool use_ucsd_fah_2d = false;
    bool visualize = false;
    bool test_io = false;
    bool hold = false;
    int stride = 1;
    double map_resolution = 0.025;
    double surf_normal_scale = 0.35;
    int init_frame = 0;
};

static Options g_options;
const std::filesystem::path kProjectRootDir = ERL_SDF_MAPPING_ROOT_DIR;

TEST(GpOccSurfaceMapping2D, Build_Save_Load) {
    GTEST_PREPARE_OUTPUT_DIR();

    g_options.gazebo_train_file = kProjectRootDir / "data" / "gazebo_train.dat";
    g_options.house_expo_map_file = kProjectRootDir / "data" / "house_expo_room_1451.json";
    g_options.house_expo_traj_file = kProjectRootDir / "data" / "house_expo_room_1451.csv";
    g_options.ucsd_fah_2d_file = kProjectRootDir / "data" / "ucsd_fah_2d.dat";
    g_options.surface_mapping_config_file = kProjectRootDir / "config" / "surface_mapping_2d.yaml";
    g_options.output_dir = (test_output_dir / "results").string();

    ASSERT_TRUE(g_options.use_gazebo_room_2d || g_options.use_house_expo_lidar_2d || g_options.use_ucsd_fah_2d)
        << "Please specify one of --use-gazebo-data, --use-house-expo-data, --use-ros-bag-data.";
    if (g_options.use_gazebo_room_2d) {
        ASSERT_TRUE(std::filesystem::exists(g_options.gazebo_train_file)) << "Gazebo train data file " << g_options.gazebo_train_file << " does not exist.";
    }
    if (g_options.use_house_expo_lidar_2d) {
        ASSERT_TRUE(std::filesystem::exists(g_options.house_expo_map_file)) << "HouseExpo map file " << g_options.house_expo_map_file << " does not exist.";
        ASSERT_TRUE(std::filesystem::exists(g_options.house_expo_traj_file))
            << "HouseExpo trajectory file " << g_options.house_expo_traj_file << " does not exist.";
    }
    if (g_options.use_ucsd_fah_2d) {
        ASSERT_TRUE(std::filesystem::exists(g_options.ucsd_fah_2d_file)) << "ROS bag dat file " << g_options.ucsd_fah_2d_file << " does not exist.";
    }
    std::filesystem::create_directories(g_options.output_dir);

    long max_update_cnt;
    std::vector<Eigen::VectorXd> train_angles;
    std::vector<Eigen::VectorXd> train_ranges;
    std::vector<std::pair<Eigen::Matrix2d, Eigen::Vector2d>> train_poses;
    Eigen::Vector2d map_min(0, 0);
    Eigen::Vector2d map_max(0, 0);
    Eigen::Vector2d map_resolution(g_options.map_resolution, g_options.map_resolution);
    Eigen::Vector2i map_padding(10, 10);
    Eigen::Matrix2Xd cur_traj;

    if (g_options.use_gazebo_room_2d) {
        auto train_data_loader = erl::geometry::GazeboRoom2D::TrainDataLoader(g_options.gazebo_train_file.c_str());
        max_update_cnt = static_cast<long>(train_data_loader.size() - g_options.init_frame) / g_options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        auto bar_setting = std::make_shared<erl::common::ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = erl::common::ProgressBar::Open(bar_setting, std::cout);
        int cnt = 0;
        for (int i = g_options.init_frame; i < static_cast<int>(train_data_loader.size()); i += g_options.stride, ++cnt) {
            auto &df = train_data_loader[i];
            train_angles.push_back(df.angles);
            train_ranges.push_back(df.ranges);
            train_poses.emplace_back(df.rotation, df.translation);
            cur_traj.col(cnt) << df.translation;
            const double &x = cur_traj(0, cnt);
            const double &y = cur_traj(1, cnt);
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
            bar->Update();
        }
        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else if (g_options.use_house_expo_lidar_2d) {
        erl::geometry::HouseExpoMap house_expo_map(g_options.house_expo_map_file.c_str(), 0.2);
        map_min = house_expo_map.GetMeterSpace()->GetSurface()->vertices.rowwise().minCoeff();
        map_max = house_expo_map.GetMeterSpace()->GetSurface()->vertices.rowwise().maxCoeff();
        auto lidar_setting = std::make_shared<erl::geometry::Lidar2D::Setting>();
        lidar_setting->num_lines = 720;
        erl::geometry::Lidar2D lidar(lidar_setting, house_expo_map.GetMeterSpace());
        auto trajectory =
            erl::common::LoadAndCastCsvFile<double>(g_options.house_expo_traj_file.c_str(), [](const std::string &str) -> double { return std::stod(str); });
        max_update_cnt = static_cast<long>(trajectory.size() - g_options.init_frame) / g_options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        auto bar_setting = std::make_shared<erl::common::ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = erl::common::ProgressBar::Open(bar_setting, std::cout);
        int cnt = 0;
        for (std::size_t i = g_options.init_frame; i < trajectory.size(); i += g_options.stride, cnt++) {
            bool scan_in_parallel = true;
            std::vector<double> &waypoint = trajectory[i];
            cur_traj.col(cnt) << waypoint[0], waypoint[1];

            Eigen::Matrix2d rotation = Eigen::Rotation2Dd(waypoint[2]).toRotationMatrix();
            Eigen::Vector2d translation(waypoint[0], waypoint[1]);
            auto lidar_ranges = lidar.Scan(rotation, translation, scan_in_parallel);
            lidar_ranges += erl::common::GenerateGaussianNoise(lidar_ranges.size(), 0.0, 0.01);
            train_angles.push_back(lidar.GetAngles());
            train_ranges.push_back(lidar_ranges);
            train_poses.emplace_back(rotation, translation);
            bar->Update();
        }
        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else if (g_options.use_ucsd_fah_2d) {
        erl::geometry::UcsdFah2D ucsd_fah(g_options.ucsd_fah_2d_file);
        map_min = erl::geometry::UcsdFah2D::kMapMin;
        map_max = erl::geometry::UcsdFah2D::kMapMax;
        // prepare buffer
        max_update_cnt = (ucsd_fah.Size() - g_options.init_frame) / g_options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        // load data into buffer
        auto bar_setting = std::make_shared<erl::common::ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = erl::common::ProgressBar::Open(bar_setting, std::cout);
        long cnt = 0;
        for (long i = g_options.init_frame; i < ucsd_fah.Size(); i += g_options.stride, ++cnt) {
            auto [sequence_number, timestamp, header_timestamp, rotation, translation, angles, ranges] = ucsd_fah[i];
            cur_traj.col(cnt) << translation;
            train_angles.emplace_back(angles);
            train_ranges.emplace_back(ranges);
            train_poses.emplace_back(std::move(rotation), std::move(translation));
            bar->Update();
        }
        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else {
        std::cerr << "Please specify one of --use-gazebo-data, --use-house-expo-data, --use-ros-bag-data." << std::endl;
        return;
    }
    max_update_cnt = cur_traj.cols();

    const auto gp_setting = std::make_shared<erl::sdf_mapping::GpOccSurfaceMapping2D::Setting>();
    ASSERT_TRUE(gp_setting->FromYamlFile(g_options.surface_mapping_config_file)) << "Failed to load config file.";
    gp_setting->sensor_gp->lidar_frame->angle_min = train_angles[0].minCoeff();
    gp_setting->sensor_gp->lidar_frame->angle_max = train_angles[0].maxCoeff();
    gp_setting->sensor_gp->lidar_frame->num_rays = train_ranges[0].size();
    erl::sdf_mapping::GpOccSurfaceMapping2D gp(gp_setting);
    std::cout << "Surface Mapping Setting:" << std::endl << *gp_setting << std::endl;

    using OccupancyQuadtreeDrawer = erl::sdf_mapping::SurfaceMappingQuadtree::Drawer;
    auto drawer_setting = std::make_shared<OccupancyQuadtreeDrawer::Setting>();
    drawer_setting->area_min = map_min;
    drawer_setting->area_max = map_max;
    drawer_setting->resolution = map_resolution[0];
    drawer_setting->padding = map_padding[0];
    drawer_setting->border_color = cv::Scalar(255, 0, 0, 255);
    std::cout << "Quadtree Drawer Setting:" << std::endl << *drawer_setting << std::endl;
    auto drawer = std::make_shared<OccupancyQuadtreeDrawer>(drawer_setting);
    std::vector<std::pair<cv::Point, cv::Point>> arrowed_lines;
    auto &surface_data_manager = gp.GetSurfaceDataManager();
    drawer->SetDrawTreeCallback([&](const OccupancyQuadtreeDrawer *self, cv::Mat &img, erl::sdf_mapping::SurfaceMappingQuadtree::TreeIterator &it) {
        const uint32_t cluster_depth = gp.GetQuadtree()->GetTreeDepth() - gp.GetClusterLevel();
        const auto grid_map_info = self->GetGridMapInfo();
        if (it->GetDepth() == cluster_depth) {
            Eigen::Vector2i position_px = grid_map_info->MeterToPixelForPoints(Eigen::Vector2d(it.GetX(), it.GetY()));
            const cv::Point position_px_cv(position_px[0], position_px[1]);
            cv::circle(img, position_px_cv, 2, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
            return;
        }
        if (!it->HasSurfaceData()) { return; }
        auto &surface_data = surface_data_manager[it->surface_data_index];
        Eigen::Vector2i position_px = grid_map_info->MeterToPixelForPoints(surface_data.position);
        cv::Point position_px_cv(position_px[0], position_px[1]);
        cv::circle(img, cv::Point(position_px[0], position_px[1]), 1, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
        Eigen::Vector2i normal_px = grid_map_info->MeterToPixelForVectors(surface_data.normal * g_options.surf_normal_scale);
        cv::Point arrow_end_px(position_px[0] + normal_px[0], position_px[1] + normal_px[1]);
        arrowed_lines.emplace_back(position_px_cv, arrow_end_px);
    });
    drawer->SetDrawLeafCallback([&](const OccupancyQuadtreeDrawer *self, cv::Mat &img, erl::sdf_mapping::SurfaceMappingQuadtree::LeafIterator &it) {
        if (!it->HasSurfaceData()) { return; }
        const auto grid_map_info = self->GetGridMapInfo();
        auto &surface_data = surface_data_manager[it->surface_data_index];
        Eigen::Vector2i position_px = grid_map_info->MeterToPixelForPoints(surface_data.position);
        cv::Point position_px_cv(position_px[0], position_px[1]);
        cv::circle(img, position_px_cv, 1, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
        Eigen::Vector2i normal_px = grid_map_info->MeterToPixelForVectors(surface_data.normal * g_options.surf_normal_scale);
        cv::Point arrow_end_px(position_px[0] + normal_px[0], position_px[1] + normal_px[1]);
        arrowed_lines.emplace_back(position_px_cv, arrow_end_px);
    });

    cv::Scalar trajectory_color(0, 0, 0, 255);
    cv::Mat img;
    bool drawer_connected = false;
    const bool update_occupancy = gp_setting->update_occupancy;
    if (g_options.visualize) {
        if (update_occupancy) {
            drawer->DrawLeaves(img);
        } else {
            drawer->DrawTree(img);
        }
    }
    if (g_options.hold) {
        std::cout << "Press any key to start" << std::endl;
        cv::waitKey();  // wait for any key
    }
    auto grid_map_info = drawer->GetGridMapInfo();
    const std::string bin_file = g_options.output_dir + "/gp_occ_surface_mapping_2d.bin";
    double t_ms = 0;
    for (long i = 0; i < max_update_cnt; i++) {
        const auto &[rotation, translation] = train_poses[i];
        const Eigen::VectorXd &ranges = train_ranges[i];
        auto t0 = std::chrono::high_resolution_clock::now();
        EXPECT_TRUE(gp.Update(rotation, translation, ranges));
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("Update time: {:f} ms.", dt);
        t_ms += dt;

        if (g_options.visualize) {
            bool pixel_based = true;
            if (!drawer_connected) {
                drawer->SetQuadtree(gp.GetQuadtree());
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

            // draw trajectory
            erl::common::DrawTrajectoryInplace(img, cur_traj.block(0, 0, 2, i), grid_map_info, trajectory_color, 2, pixel_based);

            // draw fps
            cv::putText(img, std::to_string(1000.0 / dt), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0, 255), 2);

            cv::imshow("GP Occ Surface Mapping", img);
            int key = cv::waitKey(1);
            if (key == 27) { break; }  // ESC
            if (key == 'q') { break; }
        }
        std::cout << "=====================================" << std::endl;

        if (g_options.test_io) {
            ASSERT_TRUE(gp.Write(bin_file)) << "Failed to write to " << bin_file;
            erl::sdf_mapping::GpOccSurfaceMapping2D gp_load(std::make_shared<erl::sdf_mapping::GpOccSurfaceMapping2D::Setting>());
            ASSERT_TRUE(gp_load.Read(bin_file)) << "Failed to read from " << bin_file;
            ASSERT_TRUE(gp == gp_load) << "Loaded GP is not equal to the original GP.";
        }
    }

    if (g_options.hold) {
        std::cout << "Press any key to exit." << std::endl;
        cv::waitKey();  // wait for any key
    } else {
        auto t0 = std::chrono::high_resolution_clock::now();
        double wait_time = 10.0;
        while (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() < wait_time) {
            int key = cv::waitKey(10);
            if (key == 27) { break; }  // ESC
            if (key == 'q') { break; }
        }
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
            ("use-gazebo-data", po::bool_switch(&g_options.use_gazebo_room_2d)->default_value(g_options.use_gazebo_room_2d), "Use Gazebo data")
            ("use-house-expo-data", po::bool_switch(&g_options.use_house_expo_lidar_2d)->default_value(g_options.use_house_expo_lidar_2d), "Use HouseExpo data")
            ("use-ros-bag-data", po::bool_switch(&g_options.use_ucsd_fah_2d)->default_value(g_options.use_ucsd_fah_2d), "Use ROS bag data")
            ("stride", po::value<int>(&g_options.stride)->default_value(g_options.stride), "stride for running the sequence")
            ("map-resolution", po::value<double>(&g_options.map_resolution)->default_value(g_options.map_resolution), "Map resolution")
            ("surf-normal-scale", po::value<double>(&g_options.surf_normal_scale)->default_value(g_options.surf_normal_scale), "Surface normal scale")
            ("init-frame", po::value<int>(&g_options.init_frame)->default_value(g_options.init_frame), "Initial frame index")
            ("visualize", po::bool_switch(&g_options.visualize)->default_value(g_options.visualize), "Visualize the mapping")
            ("test-io", po::bool_switch(&g_options.test_io)->default_value(g_options.test_io), "Test IO")
            ("output-dir", po::value<std::string>(&g_options.output_dir)->default_value(g_options.output_dir)->value_name("dir"), "Output directory")
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
                "ros-bag-dat-file",
                po::value<std::string>(&g_options.ucsd_fah_2d_file)->default_value(g_options.ucsd_fah_2d_file)->value_name("file"),
                "ROS bag dat file"
            )(
                "surface-mapping-config-file",
                po::value<std::string>(&g_options.surface_mapping_config_file)->default_value(g_options.surface_mapping_config_file)->value_name("file"),
                "Surface mapping config file");
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
    return RUN_ALL_TESTS();
}
