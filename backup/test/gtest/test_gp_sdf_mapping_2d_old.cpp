#include "erl_common/pangolin_plotter_curve_2d.hpp"
#include "erl_common/pangolin_plotter_image.hpp"
#include "erl_common/progress_bar.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_geometry/gazebo_room_2d.hpp"
#include "erl_geometry/house_expo_map_lidar_2d.hpp"
#include "erl_geometry/lidar_2d.hpp"
#include "erl_geometry/occupancy_quadtree_drawer.hpp"
#include "erl_geometry/ucsd_fah_2d.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping_2d.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_2d.hpp"

#include <boost/program_options.hpp>

#include <filesystem>

static std::string g_window_name = "ERL_SDF_MAPPING";
const std::filesystem::path kProjectRootDir = ERL_SDF_MAPPING_ROOT_DIR;

struct Options {
    std::string gazebo_train_file = kProjectRootDir / "data" / "gazebo_train.dat";
    std::string house_expo_map_file = kProjectRootDir / "data" / "house_expo_room_1451.json";
    std::string house_expo_traj_file = kProjectRootDir / "data" / "house_expo_room_1451.csv";
    std::string ucsd_fah_2d_file = kProjectRootDir / "data" / "ucsd_fah_2d.dat";
    std::string sdf_mapping_config_file = kProjectRootDir / "config" / "sdf_mapping_2d.yaml";
    std::string output_dir;
    bool use_gazebo_room_2d = false;
    bool use_house_expo_lidar_2d = false;
    bool use_ucsd_fah_2d = false;
    bool visualize = false;
    bool test_io = false;
    bool hold = false;
    bool interactive = false;
    bool save_video = false;
    int stride = 1;
    double map_resolution = 0.025;
    double surf_normal_scale = 0.35;
    int init_frame = 0;
};

static Options g_options;
using SurfaceData = erl::sdf_mapping::SurfaceDataManager<2>::SurfaceData;
using Gp = erl::sdf_mapping::SdfGaussianProcess<2, SurfaceData>;

void
DrawGp(
    cv::Mat &img,
    const std::shared_ptr<Gp> &gp1,
    const std::shared_ptr<erl::common::GridMapInfo2D> &grid_map_info,
    const cv::Scalar &data_color = {0, 255, 125, 255},
    const cv::Scalar &pos_color = {125, 255, 0, 255},
    const cv::Scalar &rect_color = {125, 255, 0, 255}) {

    if (gp1 == nullptr) { return; }

    Eigen::Vector2i gp1_position_px = grid_map_info->MeterToPixelForPoints(gp1->position);
    cv::drawMarker(img, cv::Point(gp1_position_px[0], gp1_position_px[1]), pos_color, cv::MARKER_STAR, 10, 1);
    const Eigen::Vector2d gp1_area_min = gp1->position.array() - gp1->half_size;
    const Eigen::Vector2d gp1_area_max = gp1->position.array() + gp1->half_size;
    Eigen::Vector2i gp1_area_min_px = grid_map_info->MeterToPixelForPoints(gp1_area_min);
    Eigen::Vector2i gp1_area_max_px = grid_map_info->MeterToPixelForPoints(gp1_area_max);
    cv::rectangle(img, cv::Point(gp1_area_min_px[0], gp1_area_min_px[1]), cv::Point(gp1_area_max_px[0], gp1_area_max_px[1]), rect_color, 2);

    const Eigen::Matrix2Xd used_surface_points = gp1->edf_gp->GetTrainInputSamplesBuffer().block(0, 0, 2, gp1->edf_gp->GetNumTrainSamples());
    Eigen::Matrix2Xi used_surface_points_px = grid_map_info->MeterToPixelForPoints(used_surface_points);
    for (long j = 0; j < used_surface_points.cols(); j++) {
        cv::circle(img, cv::Point(used_surface_points_px(0, j), used_surface_points_px(1, j)), 3, data_color, -1);
    }
}

struct OpenCvUserData {
    std::shared_ptr<erl::common::GridMapInfo2D> grid_map_info;
    erl::sdf_mapping::GpSdfMapping2D *sdf_mapping = nullptr;
    cv::Mat img;
};

void
OpenCvMouseCallback(const int event, const int x, const int y, int /*flags*/, void *userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        auto *data = static_cast<OpenCvUserData *>(userdata);
        Eigen::Vector2d position = data->grid_map_info->PixelToMeterForPoints(Eigen::Vector2i(x, y));
        ERL_INFO("Clicked at [{:f}, {:f}].", position.x(), position.y());
        Eigen::VectorXd distance(1);
        Eigen::Matrix2Xd gradient(2, 1);
        Eigen::Matrix3Xd variances(3, 1);
        Eigen::Matrix3Xd covariances(3, 1);
        if (data->sdf_mapping->Test(position, distance, gradient, variances, covariances)) {
            ERL_INFO(
                "SDF at [{:f}, {:f}]: {:f}, grad: [{:f}, {:f}], var: {}, cov: {}.",
                position.x(),
                position.y(),
                distance[0],
                gradient(0, 0),
                gradient(1, 0),
                variances.col(0).transpose(),
                covariances.col(0).transpose());
            cv::Mat img = data->img.clone();

            // draw sdf
            auto radius = static_cast<int>(std::abs(distance[0]) / data->grid_map_info->Resolution(0));
            cv::Mat circle_layer(img.rows, img.cols, CV_8UC4, cv::Scalar(0));
            cv::Mat circle_mask(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
            cv::circle(circle_mask, cv::Point2i(x, y), radius, cv::Scalar(255), cv::FILLED);
            cv::circle(circle_layer, cv::Point2i(x, y), radius, cv::Scalar(0, 255, 0, 25), cv::FILLED);
            cv::add(img * 0.5, circle_layer * 0.5, img, circle_mask);

            // draw sdf variance
            radius = static_cast<int>((std::sqrt(variances(0, 0)) + std::abs(distance[0])) / data->grid_map_info->Resolution(0));
            cv::circle(img, cv::Point2i(x, y), radius, cv::Scalar(0, 0, 255, 25), 1);

            // draw sdf gradient
            Eigen::VectorXi grad_pixel = data->grid_map_info->MeterToPixelForVectors(gradient.col(0));
            cv::arrowedLine(img, cv::Point(x, y), cv::Point(x + grad_pixel[0], y + grad_pixel[1]), cv::Scalar(255, 0, 0, 255), 1);

            auto &[gp1, gp2] = data->sdf_mapping->GetUsedGps()[0];
            if (gp1 != nullptr) { DrawGp(img, gp1, data->grid_map_info); }
            if (gp2 != nullptr) { DrawGp(img, gp2, data->grid_map_info); }

            cv::putText(
                img,
                fmt::format(
                    "SDF: {:.2f}, Var: {:.6f} | grad: [{:.6f}, {:.6f}], Var: [{:.6f}, {:.6f}], Std(theta): {:.6f}",
                    distance[0],
                    variances(0, 0),
                    gradient(0, 0),
                    gradient(1, 0),
                    variances(1, 0),
                    variances(2, 0),
                    std::sqrt(variances(1, 0) + variances(2, 0)) * 180.0 / M_PI),
                cv::Point(10, 20),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255, 255),
                1);

            cv::imshow(g_window_name, img);
        } else {
            ERL_WARN("Failed to test SDF estimation at [{:f}, {:f}].", position.x(), position.y());
        }
    }
}

TEST(GpSdfMapping2D, Build_Save_Load) {
    GTEST_PREPARE_OUTPUT_DIR();

    g_options.output_dir = (test_output_dir / "results").string();

    ASSERT_TRUE(g_options.use_gazebo_room_2d || g_options.use_house_expo_lidar_2d || g_options.use_ucsd_fah_2d)
        << "Please specify one of --use-gazebo-room-2d, --use-house-expo-lidar-2d, --use-ucsd-fah-2d.";
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
    double tic = 0.2;

    auto sdf_mapping_setting = std::make_shared<erl::sdf_mapping::GpSdfMapping2D::Setting>();
    ERL_ASSERTM(sdf_mapping_setting->FromYamlFile(g_options.sdf_mapping_config_file), "Failed to load config file: {}", g_options.sdf_mapping_config_file);
    sdf_mapping_setting->test_query->compute_covariance = true;

    if (g_options.use_gazebo_room_2d) {
        auto train_data_loader = erl::geometry::GazeboRoom2D::TrainDataLoader(g_options.gazebo_train_file);
        map_min = erl::geometry::GazeboRoom2D::kMapMin;
        map_max = erl::geometry::GazeboRoom2D::kMapMax;
        // prepare buffer
        max_update_cnt = static_cast<long>(train_data_loader.size() - g_options.init_frame) / g_options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        // load data into buffer
        auto bar_setting = std::make_shared<erl::common::ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = erl::common::ProgressBar::Open(bar_setting, std::cout);
        int cnt = 0;
        for (int i = g_options.init_frame; i < static_cast<int>(train_data_loader.size()); i += g_options.stride, ++cnt) {
            auto [rotation, translation, angles, ranges] = train_data_loader[i];
            cur_traj.col(cnt) << translation;
            train_angles.push_back(angles);
            train_ranges.push_back(ranges);
            train_poses.emplace_back(std::move(rotation), std::move(translation));
            bar->Update();
        }
        map_resolution.array() = 0.02;
        map_padding.setZero();
        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else if (g_options.use_house_expo_lidar_2d) {
        auto lidar_setting = std::make_shared<erl::geometry::Lidar2D::Setting>();
        lidar_setting->num_lines = 720;
        erl::geometry::HouseExpoMapLidar2D house_expo_map(g_options.house_expo_map_file, g_options.house_expo_traj_file, 0.2, lidar_setting, true, 0.01);
        map_min = house_expo_map.GetMapMin();
        map_max = house_expo_map.GetMapMax();
        // prepare buffer
        max_update_cnt = (house_expo_map.Size() - g_options.init_frame) / g_options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        // load data into buffer
        auto bar_setting = std::make_shared<erl::common::ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = erl::common::ProgressBar::Open(bar_setting, std::cout);
        int cnt = 0;
        for (long i = g_options.init_frame; i < house_expo_map.Size(); i += g_options.stride, ++cnt) {
            auto [rotation, translation, angles, ranges] = house_expo_map[i];
            cur_traj.col(cnt) << translation;
            train_angles.push_back(angles);
            train_ranges.push_back(ranges);
            train_poses.emplace_back(std::move(rotation), std::move(translation));
            bar->Update();
        }
        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else if (g_options.use_ucsd_fah_2d) {
        erl::geometry::UcsdFah2D ucsd_fah(g_options.ucsd_fah_2d_file);
        tic = ucsd_fah.GetTimeStep();
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
        std::cerr << "Please specify one of --use-gazebo-room-2d, --use-house-expo-lidar-2d, --use-ucsd-fah-2d." << std::endl;
        return;
    }
    max_update_cnt = cur_traj.cols();

    ERL_INFO("map_min: {}, map_max: {}", map_min.transpose(), map_max.transpose());

    auto surface_mapping_setting = std::dynamic_pointer_cast<erl::sdf_mapping::GpOccSurfaceMapping2D::Setting>(sdf_mapping_setting->surface_mapping);
    surface_mapping_setting->sensor_gp->lidar_frame->angle_min = train_angles[0].minCoeff();
    surface_mapping_setting->sensor_gp->lidar_frame->angle_max = train_angles[0].maxCoeff();
    surface_mapping_setting->sensor_gp->lidar_frame->num_rays = train_ranges[0].size();

    std::cout << sdf_mapping_setting->AsYamlString() << std::endl;

    erl::sdf_mapping::GpSdfMapping2D sdf_mapping(sdf_mapping_setting);
    auto surface_mapping = std::dynamic_pointer_cast<erl::sdf_mapping::GpOccSurfaceMapping2D>(sdf_mapping.GetSurfaceMapping());

    using OccupancyQuadtreeDrawer = erl::geometry::OccupancyQuadtreeDrawer<erl::sdf_mapping::SurfaceMappingQuadtree>;
    auto drawer_setting = std::make_shared<OccupancyQuadtreeDrawer::Setting>();
    drawer_setting->area_min = map_min;
    drawer_setting->area_max = map_max;
    drawer_setting->resolution = map_resolution[0];
    drawer_setting->padding = map_padding[0];
    drawer_setting->border_color = cv::Scalar(255, 0, 0, 255);
    std::cout << "Quadtree Drawer Setting:" << std::endl << *drawer_setting << std::endl;
    auto drawer = std::make_shared<OccupancyQuadtreeDrawer>(drawer_setting);
    std::vector<std::pair<cv::Point, cv::Point>> arrowed_lines;
    auto &surface_data_manager = surface_mapping->GetSurfaceDataManager();
    drawer->SetDrawTreeCallback([&](const OccupancyQuadtreeDrawer *self, cv::Mat &img, erl::sdf_mapping::SurfaceMappingQuadtree::TreeIterator &it) {
        const uint32_t cluster_depth = surface_mapping->GetQuadtree()->GetTreeDepth() - surface_mapping->GetClusterLevel();
        const auto grid_map_info = self->GetGridMapInfo();
        if (it->GetDepth() == cluster_depth) {
            Eigen::Vector2i position_px = grid_map_info->MeterToPixelForPoints(Eigen::Vector2d(it.GetX(), it.GetY()));
            const cv::Point position_px_cv(position_px[0], position_px[1]);
            cv::circle(img, position_px_cv, 2, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
            return;
        }
        if (!it->HasSurfaceData()) { return; }
        Eigen::Vector2i position_px = grid_map_info->MeterToPixelForPoints(surface_data_manager[it->surface_data_index].position);
        cv::Point position_px_cv(position_px[0], position_px[1]);
        cv::circle(img, cv::Point(position_px[0], position_px[1]), 1, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
        Eigen::Vector2i normal_px = grid_map_info->MeterToPixelForVectors(surface_data_manager[it->surface_data_index].normal * g_options.surf_normal_scale);
        cv::Point arrow_end_px(position_px[0] + normal_px[0], position_px[1] + normal_px[1]);
        arrowed_lines.emplace_back(position_px_cv, arrow_end_px);
    });
    drawer->SetDrawLeafCallback([&](const OccupancyQuadtreeDrawer *self, cv::Mat &img, erl::sdf_mapping::SurfaceMappingQuadtree::LeafIterator &it) {
        if (!it->HasSurfaceData()) { return; }
        const auto grid_map_info = self->GetGridMapInfo();
        Eigen::Vector2i position_px = grid_map_info->MeterToPixelForPoints(surface_data_manager[it->surface_data_index].position);
        cv::Point position_px_cv(position_px[0], position_px[1]);
        cv::circle(img, position_px_cv, 1, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
        Eigen::Vector2i normal_px = grid_map_info->MeterToPixelForVectors(surface_data_manager[it->surface_data_index].normal * g_options.surf_normal_scale);
        cv::Point arrow_end_px(position_px[0] + normal_px[0], position_px[1] + normal_px[1]);
        arrowed_lines.emplace_back(position_px_cv, arrow_end_px);
    });

    cv::Scalar trajectory_color(0, 0, 0, 255);
    cv::Mat img;
    bool drawer_connected = false;
    bool update_occupancy = surface_mapping->GetSetting()->update_occupancy;
    if (g_options.visualize) {
        if (update_occupancy) {
            drawer->DrawLeaves(img);
        } else {
            drawer->DrawTree(img);
        }
    }
    if (g_options.hold) {
        std::cout << "Press any key to start" << std::endl;
        while (!pangolin::ShouldQuit()) {}  // wait for any key
    }
    auto grid_map_info = drawer->GetGridMapInfo();

    // pangolin
    std::shared_ptr<erl::common::PangolinWindow> pangolin_plotter_window;
    std::shared_ptr<erl::common::PangolinPlotterCurve2D> plotter_sdf;
    std::shared_ptr<erl::common::PangolinPlotterCurve2D> plotter_grad;
    std::shared_ptr<erl::common::PangolinWindow> pangolin_map_window;
    std::shared_ptr<erl::common::PangolinPlotterImage> pangolin_plotter_map;
    if (g_options.visualize) {
        pangolin_plotter_window = std::make_shared<erl::common::PangolinWindow>(g_window_name + ": curves", 1280, 960);
        pangolin_plotter_window->GetDisplay("main").SetLayout(pangolin::LayoutEqualVertical);
        plotter_sdf = std::make_shared<erl::common::PangolinPlotterCurve2D>(
            pangolin_plotter_window,
            "SDF",
            std::vector<std::string>{"t", "SDF", "EDF", "var(SDF)"},
            500,                              // number of points in the plot window
            0.0f,                             // t0
            static_cast<float>(tic) / 10.0f,  // dt
            0.05f);                           // dy

        plotter_grad = std::make_shared<erl::common::PangolinPlotterCurve2D>(
            pangolin_plotter_window,
            "Gradient",
            std::vector<std::string>{"t", "gradX", "gradY", "var(gradX)", "var(gradY)"},
            500,                                      // number of points in the plot window
            0.0f,                                     // t0
            static_cast<float>(tic) / 10.0f,          // dt
            0.05f,                                    // dy
            pangolin::Colour{0.1f, 0.1f, 0.1f, 1.0f}  // bg_color
        );

        pangolin_map_window = std::make_shared<erl::common::PangolinWindow>(g_window_name + ": map", img.cols, img.rows);
        pangolin_plotter_map = std::make_shared<erl::common::PangolinPlotterImage>(pangolin_map_window, img.rows, img.cols, GL_RGBA, GL_UNSIGNED_BYTE);
    }

    // save video
    std::shared_ptr<cv::VideoWriter> video_writer = nullptr;
    std::string video_path = (std::filesystem::path(g_options.output_dir) / "sdf_mapping.avi").string();
    cv::Mat video_frame;
    if (g_options.save_video) {
        cv::Size size(img.cols + 1280, std::max(img.rows, 960));
        video_writer = std::make_shared<cv::VideoWriter>(video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0, size);
        video_frame = cv::Mat(std::max(img.rows, 960), img.cols + 1280, CV_8UC3, cv::Scalar(0));
    }
    const std::string bin_file = g_options.output_dir + "/gp_sdf_mapping_2d.bin";
    double t = 0;
    for (long i = 0; i < max_update_cnt; i++) {
        const auto &[rotation, translation] = train_poses[i];
        const Eigen::VectorXd &ranges = train_ranges[i];
        auto t0 = std::chrono::high_resolution_clock::now();
        EXPECT_TRUE(sdf_mapping.Update(rotation, translation, ranges));
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("Update time: {:f} ms.", dt);
        t += tic;

        if (g_options.visualize) {
            bool pixel_based = true;
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

            // Test SDF Estimation
            Eigen::Vector2d position = cur_traj.col(i);
            Eigen::VectorXd distance(1);
            Eigen::Matrix2Xd gradient(2, 1);
            Eigen::Matrix3Xd variances(3, 1);
            Eigen::Matrix3Xd covariances(3, 1);
            EXPECT_TRUE(sdf_mapping.Test(position, distance, gradient, variances, covariances));

            // draw sdf
            cv::Point position_px(
                grid_map_info->MeterToGridForValue(cur_traj(0, i), 0),
                grid_map_info->Shape(1) - grid_map_info->MeterToGridForValue(cur_traj(1, i), 1));
            auto radius = static_cast<int>(std::abs(distance[0]) / grid_map_info->Resolution(0));
            cv::Mat circle_layer(img.rows, img.cols, CV_8UC4, cv::Scalar(0));
            cv::Mat circle_mask(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
            cv::circle(circle_mask, position_px, radius, cv::Scalar(255), cv::FILLED);
            cv::circle(circle_layer, position_px, radius, cv::Scalar(0, 255, 0, 25), cv::FILLED);
            cv::add(img * 0.5, circle_layer * 0.5, img, circle_mask);

            // draw sdf gradient
            Eigen::Vector2i gradient_px = grid_map_info->MeterToPixelForVectors(gradient);
            cv::arrowedLine(
                img,
                position_px,
                cv::Point(position_px.x + gradient_px.x(), position_px.y + gradient_px.y()),
                cv::Scalar(255, 0, 0, 255),
                2,
                8,
                0,
                0.15);

            // draw used surface points
            auto &[gp1, gp2] = sdf_mapping.GetUsedGps()[0];
            if (gp1 != nullptr) {
                DrawGp(img, gp1, grid_map_info, {0, 125, 255, 255}, {125, 255, 0, 255}, {255, 125, 0, 255});
                ERL_INFO("GP1 at [{:f}, {:f}] has {} data points.", gp1->position.x(), gp1->position.y(), gp1->edf_gp->GetNumTrainSamples());
            }
            if (gp2 != nullptr) {
                DrawGp(img, gp2, grid_map_info, {125, 125, 255, 255}, {125, 255, 125, 255}, {255, 125, 0, 255});
                ERL_INFO("GP2 has {} data points.", gp2->edf_gp->GetNumTrainSamples());
            }

            // draw trajectory
            erl::common::DrawTrajectoryInplace(img, cur_traj.leftCols(i), grid_map_info, trajectory_color, 2, pixel_based);

            // draw fps
            cv::putText(img, std::to_string(1000.0 / dt), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0, 255), 2);

            if (pangolin_map_window != nullptr && pangolin_plotter_map != nullptr) {
                pangolin_plotter_map->Update(img);

                // pangolin_plotter_window->Activate(true);
                plotter_sdf->Append(
                    static_cast<float>(t),
                    {static_cast<float>(distance[0]), static_cast<float>(std::abs(distance[0])), static_cast<float>(variances(0, 0))});
                plotter_grad->Append(
                    static_cast<float>(t),
                    {static_cast<float>(gradient(0, 0)),
                     static_cast<float>(gradient(1, 0)),
                     static_cast<float>(variances(1, 0)),
                     static_cast<float>(variances(2, 0))});
                if (g_options.save_video) {
                    pangolin::TypedImage buffer = pangolin::ReadFramebuffer(pangolin_plotter_window->GetDisplay("main").v, "BGR24");
                    cv::Mat tmp(static_cast<int>(buffer.h), static_cast<int>(buffer.w), CV_8UC3, buffer.ptr);
                    cv::flip(tmp, tmp, 0);
                    int offset = (video_frame.rows - tmp.rows) / 2;
                    tmp.copyTo(video_frame(cv::Rect(img.cols, offset, tmp.cols, tmp.rows)));
                    cv::cvtColor(img, tmp, cv::COLOR_BGRA2BGR);
                    tmp.copyTo(video_frame(cv::Rect(0, 0, tmp.cols, tmp.rows)));
                    video_writer->write(video_frame);
                }
                pangolin_plotter_window->Deactivate();
            } else {
                ERL_WARN("Pangolin window is not initialized.");
            }
        }
        std::cout << "=====================================" << std::endl;

        if (g_options.test_io && (i == 0 || i == max_update_cnt - 1)) {  // test io
            ASSERT_TRUE(sdf_mapping.Write(bin_file)) << "Failed to write to " << bin_file;
            erl::sdf_mapping::GpSdfMapping2D sdf_mapping_load(std::make_shared<erl::sdf_mapping::GpSdfMapping2D::Setting>());
            ASSERT_TRUE(sdf_mapping_load.Read(bin_file)) << "Failed to read from " << bin_file;
            ASSERT_TRUE(sdf_mapping == sdf_mapping_load) << "Loaded SDF mapping is not equal to the original one.";
        }
    }

    ERL_INFO("Average update time: {:f} ms.", t / static_cast<double>(max_update_cnt));
    if (g_options.save_video) {
        video_writer->release();
        ERL_INFO("Saved surface mapping video to {}.", video_path.c_str());
    }

    // Test SDF Estimation
    constexpr bool c_stride = true;
    Eigen::Matrix2Xd positions_in = grid_map_info->GenerateMeterCoordinates(c_stride);
    Eigen::VectorXd distances_out(positions_in.cols());
    Eigen::Matrix2Xd gradients_out(2, positions_in.cols());
    Eigen::Matrix3Xd variances_out(3, positions_in.cols());
    Eigen::Matrix3Xd covariances_out;
    auto t0 = std::chrono::high_resolution_clock::now();
    bool success = sdf_mapping.Test(positions_in, distances_out, gradients_out, variances_out, covariances_out);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double t_per_point = dt / static_cast<double>(positions_in.cols()) * 1000;  // us
    ERL_INFO("Test time: {:f} ms for {} points, {:f} us per point.", dt, positions_in.cols(), t_per_point);

    EXPECT_TRUE(success) << "Failed to test SDF estimation at the end.";
    ERL_ASSERT(surface_mapping->GetQuadtree()->WriteBinary("tree.bt"));
    ERL_ASSERT(surface_mapping->GetQuadtree()->Write("tree.ot"));

    std::shared_ptr<erl::common::PangolinWindow> pangolin_sdf_image_window;
    std::shared_ptr<erl::common::PangolinWindow> pangolin_sdf_sign_image_window;
    std::shared_ptr<erl::common::PangolinWindow> pangolin_sdf_var_image_window;
    std::shared_ptr<erl::common::PangolinPlotterImage> pangolin_plotter_sdf_image;
    std::shared_ptr<erl::common::PangolinPlotterImage> pangolin_plotter_sdf_sign_image;
    std::shared_ptr<erl::common::PangolinPlotterImage> pangolin_plotter_sdf_var_image;
    if (success && g_options.visualize) {
        Eigen::MatrixXd sdf_out_mat = distances_out.reshaped(grid_map_info->Height(), grid_map_info->Width());
        // erl::common::SaveEigenMatrixToTextFile<double>("sdf_out.txt", sdf_out_mat);
        double min_distance = distances_out.minCoeff();
        double max_distance = distances_out.maxCoeff();
        ERL_INFO("min distance: {:f}, max distance: {:f}.", min_distance, max_distance);

        img.setTo(cv::Scalar(128, 128, 128, 255));
        drawer_setting->border_color = drawer_setting->occupied_color;
        arrowed_lines.clear();
        if (update_occupancy) {
            drawer->DrawLeaves(img);
        } else {
            drawer->DrawTree(img);
        }

        max_distance = sdf_out_mat.maxCoeff();
        Eigen::MatrixX8U distances_out_mat_normalized = ((sdf_out_mat.array() - min_distance) / (max_distance - min_distance) * 255).cast<uint8_t>();
        cv::Mat src, dst;
        cv::eigen2cv(distances_out_mat_normalized, src);
        cv::flip(src, src, 0);  // flip along y axis
        cv::applyColorMap(src, dst, cv::COLORMAP_JET);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2BGRA);
        cv::addWeighted(dst, 0.5, img, 0.5, 0.0, dst);

        pangolin_sdf_image_window = std::make_shared<erl::common::PangolinWindow>(g_window_name + ": sdf", dst.cols, dst.rows);
        pangolin_plotter_sdf_image =
            std::make_shared<erl::common::PangolinPlotterImage>(pangolin_sdf_image_window, dst.rows, dst.cols, GL_RGBA, GL_UNSIGNED_BYTE);
        pangolin_plotter_sdf_image->Update(dst);

        Eigen::MatrixX8U sdf_sign_mat(sdf_out_mat.rows(), sdf_out_mat.cols());
        uint8_t *sdf_sign_mat_ptr = sdf_sign_mat.data();
        const double *sdf_out_mat_ptr = sdf_out_mat.data();
        for (int i = 0; i < sdf_out_mat.size(); ++i) { sdf_sign_mat_ptr[i] = sdf_out_mat_ptr[i] > 0 ? 255 : 0; }
        cv::eigen2cv(sdf_sign_mat, src);
        cv::flip(src, src, 0);  // flip along y axis
        cv::cvtColor(src, dst, cv::COLOR_GRAY2BGRA);
        pangolin_sdf_sign_image_window = std::make_shared<erl::common::PangolinWindow>(g_window_name + ": sdf_sign", src.cols, src.rows);
        pangolin_plotter_sdf_sign_image =
            std::make_shared<erl::common::PangolinPlotterImage>(pangolin_sdf_sign_image_window, src.rows, src.cols, GL_RGBA, GL_UNSIGNED_BYTE);
        pangolin_plotter_sdf_sign_image->Update(dst);

        Eigen::MatrixXd sdf_variances_mat = variances_out.row(0).reshaped(grid_map_info->Height(), grid_map_info->Width());
        double min_variance = sdf_variances_mat.minCoeff();
        double max_variance = sdf_variances_mat.maxCoeff();
        Eigen::MatrixX8U variances_out_mat_normalized = ((sdf_variances_mat.array() - min_variance) / (max_variance - min_variance) * 255).cast<uint8_t>();
        cv::eigen2cv(variances_out_mat_normalized, src);
        cv::flip(src, src, 0);  // flip along y axis
        cv::applyColorMap(src, dst, cv::COLORMAP_JET);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2BGRA);
        cv::addWeighted(dst, 0.5, img, 0.5, 0.0, dst);

        pangolin_sdf_var_image_window = std::make_shared<erl::common::PangolinWindow>(g_window_name + ": sdf_variances", dst.cols, dst.rows);
        pangolin_plotter_sdf_var_image =
            std::make_shared<erl::common::PangolinPlotterImage>(pangolin_sdf_var_image_window, dst.rows, dst.cols, GL_RGBA, GL_UNSIGNED_BYTE);
        pangolin_plotter_sdf_var_image->Update(dst);
    }

    auto process_event = [&]() {
        if (pangolin_plotter_window != nullptr) { pangolin_plotter_window->GetWindow().ProcessEvents(); }
        if (pangolin_map_window != nullptr) { pangolin_plotter_map->Render(); }
        if (pangolin_sdf_image_window != nullptr) { pangolin_plotter_sdf_image->Render(); }
        if (pangolin_sdf_sign_image_window != nullptr) { pangolin_plotter_sdf_sign_image->Render(); }
        if (pangolin_sdf_var_image_window != nullptr) { pangolin_plotter_sdf_var_image->Render(); }
    };

    if (g_options.visualize && g_options.hold) {
        std::cout << "Press any key to exit." << std::endl;
        while (!pangolin::ShouldQuit()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            process_event();
        }
    } else {
        t0 = std::chrono::high_resolution_clock::now();
        double wait_time = 10.0;
        while (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() < wait_time && !pangolin::ShouldQuit()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            process_event();
        }
    }

    pangolin::QuitAll();

    if (g_options.interactive) {

        if (!drawer_connected) { drawer->SetQuadtree(surface_mapping->GetQuadtree()); }
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

        OpenCvUserData data;
        data.img = img;
        data.grid_map_info = grid_map_info;
        data.sdf_mapping = &sdf_mapping;

        cv::imshow(g_window_name, img);
        cv::setMouseCallback(g_window_name, OpenCvMouseCallback, &data);
        while (cv::waitKey(0) != 27) {}  // wait for ESC key
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
            ("use-gazebo-room-2d", po::bool_switch(&g_options.use_gazebo_room_2d)->default_value(g_options.use_gazebo_room_2d), "Use Gazebo data")
            ("use-house-expo-lidar-2d", po::bool_switch(&g_options.use_house_expo_lidar_2d)->default_value(g_options.use_house_expo_lidar_2d), "Use HouseExpo data")
            ("use-ucsd-fah-2d", po::bool_switch(&g_options.use_ucsd_fah_2d)->default_value(g_options.use_ucsd_fah_2d), "Use ROS bag data")
            ("stride", po::value<int>(&g_options.stride)->default_value(g_options.stride), "stride for running the sequence")
            ("map-resolution", po::value<double>(&g_options.map_resolution)->default_value(g_options.map_resolution), "Map resolution")
            ("surf-normal-scale", po::value<double>(&g_options.surf_normal_scale)->default_value(g_options.surf_normal_scale), "Surface normal scale")
            ("init-frame", po::value<int>(&g_options.init_frame)->default_value(g_options.init_frame), "Initial frame index")
            ("visualize", po::bool_switch(&g_options.visualize)->default_value(g_options.visualize), "Visualize the mapping")
            ("test-io", po::bool_switch(&g_options.test_io)->default_value(g_options.test_io), "Test IO")
            ("save-video", po::bool_switch(&g_options.save_video)->default_value(g_options.save_video), "Save the mapping video")
            ("output-dir", po::value<std::string>(&g_options.output_dir)->default_value(g_options.output_dir)->value_name("dir"), "Output directory")
            ("hold", po::bool_switch(&g_options.hold)->default_value(g_options.hold), "Hold the test until a key is pressed")
            ("interactive", po::bool_switch(&g_options.interactive)->default_value(g_options.interactive), "Interactive mode")
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
                "ucsd-fah-2d-file",
                po::value<std::string>(&g_options.ucsd_fah_2d_file)->default_value(g_options.ucsd_fah_2d_file)->value_name("file"),
                "ROS bag dat file"
            )(
                "sdf-mapping-config-file",
                po::value<std::string>(&g_options.sdf_mapping_config_file)->default_value(g_options.sdf_mapping_config_file)->value_name("file"),
                "SDF mapping config file");
        // clang-format on

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        if (vm.count("help")) {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl << desc << std::endl;
            return 0;
        }
        po::notify(vm);
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    g_window_name = argv[0];
    if (g_options.save_video) { g_options.visualize = true; }
    return RUN_ALL_TESTS();
}
