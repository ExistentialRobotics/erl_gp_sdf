#include "erl_common/csv.hpp"
#include "erl_common/pangolin_plotter_time_series_2d.hpp"
#include "erl_common/progress_bar.hpp"
#include "erl_common/random.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_geometry/gazebo_room.hpp"
#include "erl_geometry/house_expo_map.hpp"
#include "erl_geometry/lidar_2d.hpp"
#include "erl_geometry/occupancy_quadtree_drawer.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping_2d.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_2d.hpp"

#include <boost/program_options.hpp>

#include <filesystem>

static std::filesystem::path g_file_path = __FILE__;
static std::filesystem::path g_dir_path = g_file_path.parent_path();
static std::filesystem::path g_config_dir_path = g_dir_path.parent_path().parent_path() / "config";
static std::string g_window_name = "ERL_SDF_MAPPING";

struct Options {
    std::string gazebo_train_file = (g_dir_path / "gazebo_train.dat").string();
    std::string house_expo_map_file = (g_dir_path / "house_expo_room_1451.json").string();
    std::string house_expo_traj_file = (g_dir_path / "house_expo_room_1451.csv").string();
    std::string ros_bag_dat_file = (g_dir_path / "ros_bag.dat").string();
    std::string surface_mapping_config_file = (g_config_dir_path / "surface_mapping.yaml").string();
    std::string sdf_mapping_config_file = (g_config_dir_path / "sdf_mapping.yaml").string();
    std::string output_dir = (g_dir_path / "results").string();
    bool use_gazebo_data = false;
    bool use_house_expo_data = false;
    bool use_ros_bag_data = false;
    bool visualize = false;
    bool hold = false;
    bool save_video = false;
    int stride = 1;
    double map_resolution = 0.025;
    double surf_normal_scale = 0.35;
    int init_frame = 0;
};

static Options g_options;

void
DrawGp(
    cv::Mat &img,
    const std::shared_ptr<erl::sdf_mapping::GpSdfMapping2D::Gp> &gp1,
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

    const Eigen::Matrix2Xd used_surface_points = gp1->gp->GetTrainInputSamplesBuffer().block(0, 0, 2, gp1->gp->GetNumTrainSamples());
    Eigen::Matrix2Xi used_surface_points_px = grid_map_info->MeterToPixelForPoints(used_surface_points);
    for (long j = 0; j < used_surface_points.cols(); j++) {
        cv::circle(img, cv::Point(used_surface_points_px(0, j), used_surface_points_px(1, j)), 3, data_color, -1);
    }
}

TEST(ERL_SDF_MAPPING, GpSdfMapping2D) {
    long max_update_cnt;
    std::vector<Eigen::VectorXd> train_angles;
    std::vector<Eigen::VectorXd> train_ranges;
    std::vector<Eigen::Matrix23d> train_poses;
    Eigen::Vector2d map_min(0, 0);
    Eigen::Vector2d map_max(0, 0);
    Eigen::Vector2d map_resolution(g_options.map_resolution, g_options.map_resolution);
    Eigen::Vector2i map_padding(10, 10);
    Eigen::Matrix2Xd cur_traj;
    double tic = 0.05;

    if (g_options.use_gazebo_data) {
        auto train_data_loader = erl::geometry::GazeboRoom::TrainDataLoader(g_options.gazebo_train_file.c_str());
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
            train_ranges.push_back(df.distances);
            train_poses.emplace_back(df.pose_numpy);
            cur_traj.col(cnt) << df.pose_numpy(0, 2), df.pose_numpy(1, 2);
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
    } else if (g_options.use_house_expo_data) {
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
            Eigen::Matrix23d pose;
            pose.leftCols<2>() << rotation;
            pose.rightCols<1>() << translation;
            train_poses.push_back(std::move(pose));
            bar->Update();
        }
        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else if (g_options.use_ros_bag_data) {
        Eigen::MatrixXd ros_bag_data = erl::common::LoadEigenMatrixFromBinaryFile<double, Eigen::Dynamic, Eigen::Dynamic>(g_options.ros_bag_dat_file);
        tic = ros_bag_data(1, 0) - ros_bag_data(0, 0);
        // prepare buffer
        max_update_cnt = static_cast<long>(ros_bag_data.rows() - g_options.init_frame) / g_options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        // load data into buffer
        long num_rays = (ros_bag_data.cols() - 7) / 2;
        auto bar_setting = std::make_shared<erl::common::ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = erl::common::ProgressBar::Open(bar_setting, std::cout);
        long cnt = 0;
        for (long i = g_options.init_frame; i < ros_bag_data.rows(); i += g_options.stride, cnt++) {
            Eigen::Matrix23d pose = ros_bag_data.row(i).segment(1, 6).reshaped(3, 2).transpose();
            cur_traj.col(cnt) << pose(0, 2), pose(1, 2);
            train_angles.emplace_back(ros_bag_data.row(i).segment(7, num_rays));
            train_ranges.emplace_back(ros_bag_data.row(i).segment(7 + num_rays, num_rays));
            train_poses.push_back(pose);
            const auto &angles = train_angles.back();
            const Eigen::VectorXd &ranges = train_ranges.back();
            for (long k = 0; k < num_rays; k++) {
                if (std::isnan(ranges[k]) || std::isinf(ranges[k])) { continue; }
                double angle = angles[k];
                double range = ranges[k];
                Eigen::Vector2d point = pose * Eigen::Vector3d(range * std::cos(angle), range * std::sin(angle), 1);
                if (point[0] < map_min[0]) { map_min[0] = point[0]; }
                if (point[0] > map_max[0]) { map_max[0] = point[0]; }
                if (point[1] < map_min[1]) { map_min[1] = point[1]; }
                if (point[1] > map_max[1]) { map_max[1] = point[1]; }
            }
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

    auto surface_mapping_setting = std::make_shared<erl::sdf_mapping::GpOccSurfaceMapping2D::Setting>();
    surface_mapping_setting->FromYamlFile(g_options.surface_mapping_config_file);
    auto surface_mapping = std::make_shared<erl::sdf_mapping::GpOccSurfaceMapping2D>(surface_mapping_setting);
    auto sdf_mapping_setting = std::make_shared<erl::sdf_mapping::GpSdfMapping2D::Setting>();
    sdf_mapping_setting->FromYamlFile(g_options.sdf_mapping_config_file);
    erl::sdf_mapping::GpSdfMapping2D sdf_mapping(surface_mapping, sdf_mapping_setting);
    sdf_mapping.GetSetting()->test_query->compute_covariance = true;
    std::cout << "Surface Mapping Setting:" << std::endl
              << *surface_mapping->GetSetting() << std::endl
              << "Sdf Mapping Setting:" << std::endl
              << *sdf_mapping.GetSetting() << std::endl;
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
    drawer->SetDrawTreeCallback([&](const OccupancyQuadtreeDrawer *self, cv::Mat &img, erl::sdf_mapping::SurfaceMappingQuadtree::TreeIterator &it) {
        const uint32_t cluster_depth = surface_mapping->GetQuadtree()->GetTreeDepth() - surface_mapping->GetClusterLevel();
        const auto grid_map_info = self->GetGridMapInfo();
        if (it->GetDepth() == cluster_depth) {
            Eigen::Vector2i position_px = grid_map_info->MeterToPixelForPoints(Eigen::Vector2d(it.GetX(), it.GetY()));
            const cv::Point position_px_cv(position_px[0], position_px[1]);
            cv::circle(img, position_px_cv, 2, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
            return;
        }
        if (it->GetSurfaceData() == nullptr) { return; }
        Eigen::Vector2i position_px = grid_map_info->MeterToPixelForPoints(it->GetSurfaceData()->position);
        cv::Point position_px_cv(position_px[0], position_px[1]);
        cv::circle(img, cv::Point(position_px[0], position_px[1]), 1, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
        Eigen::Vector2i normal_px = grid_map_info->MeterToPixelForVectors(it->GetSurfaceData()->normal * g_options.surf_normal_scale);
        cv::Point arrow_end_px(position_px[0] + normal_px[0], position_px[1] + normal_px[1]);
        arrowed_lines.emplace_back(position_px_cv, arrow_end_px);
    });
    drawer->SetDrawLeafCallback([&](const OccupancyQuadtreeDrawer *self, cv::Mat &img, erl::sdf_mapping::SurfaceMappingQuadtree::LeafIterator &it) {
        if (it->GetSurfaceData() == nullptr) { return; }
        const auto grid_map_info = self->GetGridMapInfo();
        Eigen::Vector2i position_px = grid_map_info->MeterToPixelForPoints(it->GetSurfaceData()->position);
        cv::Point position_px_cv(position_px[0], position_px[1]);
        cv::circle(img, position_px_cv, 1, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
        Eigen::Vector2i normal_px = grid_map_info->MeterToPixelForVectors(it->GetSurfaceData()->normal * g_options.surf_normal_scale);
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
    // pangolin::WindowInterface *pangolin_plotter_window = nullptr;
    // pangolin::View *pangolin_plotter_view = nullptr;
    // std::shared_ptr<pangolin::DataLog> pangolin_sdf_log = nullptr;
    // std::shared_ptr<pangolin::Plotter> pangolin_sdf_plotter;
    // // pangolin::View *pangolin_sdf_plotter_view = nullptr;
    // std::shared_ptr<pangolin::DataLog> pangolin_grad_log = nullptr;
    // std::shared_ptr<pangolin::Plotter> pangolin_grad_plotter;
    // pangolin::View *pangolin_grad_plotter_view = nullptr;

    std::shared_ptr<erl::common::PangolinWindow> pangolin_plotter_window;
    std::shared_ptr<erl::common::PangolinPlotterTimeSeries2D> plotter_sdf;
    std::shared_ptr<erl::common::PangolinPlotterTimeSeries2D> plotter_grad;

    pangolin::WindowInterface *pangolin_map_window = nullptr;
    pangolin::View *pangolin_map_view = nullptr;
    std::shared_ptr<pangolin::GlTexture> pangolin_texture = nullptr;
    if (g_options.visualize) {
        pangolin_plotter_window = std::make_shared<erl::common::PangolinWindow>(g_window_name + ": curves", 1280, 960);
        pangolin_plotter_window->GetDisplay("main").SetLayout(pangolin::LayoutEqualVertical);
        plotter_sdf = std::make_shared<erl::common::PangolinPlotterTimeSeries2D>(
            pangolin_plotter_window,
            "SDF",
            std::vector<std::string>{"t", "SDF", "EDF", "var(SDF)"},
            600,                      // number of points in the plot window
            0.0f,                     // t0
            static_cast<float>(tic),  // dt
            0.05f);                   // dy

        plotter_grad = std::make_shared<erl::common::PangolinPlotterTimeSeries2D>(
            pangolin_plotter_window,
            "Gradient",
            std::vector<std::string>{"t", "gradX", "gradY", "var(gradX)", "var(gradY)"},
            600,                                      // number of points in the plot window
            0.0f,                                     // t0
            static_cast<float>(tic),                  // dt
            0.05f,                                    // dy
            pangolin::Colour{0.1f, 0.1f, 0.1f, 1.0f}  // bg_color
        );

        pangolin_map_window = &pangolin::CreateWindowAndBind(g_window_name + ": map", img.cols, img.rows);
        glEnable(GL_DEPTH_TEST);
        pangolin_map_view = &(pangolin::Display("cam").SetBounds(0, 1.0, 0, 1.0, static_cast<float>(img.cols) / static_cast<float>(img.rows)));
        pangolin_texture = std::make_shared<pangolin::GlTexture>(img.cols, img.rows, GL_RGBA, false, 0, GL_RGBA, GL_UNSIGNED_BYTE);
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

    double t_ms = 0;
    for (long i = 0; i < max_update_cnt; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        sdf_mapping.Update(train_angles[i], train_ranges[i], train_poses[i]);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("Update time: {:f} ms.", dt);
        t_ms += dt;

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
            sdf_mapping.Test(position, distance, gradient, variances, covariances);

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
            DrawGp(img, gp1, grid_map_info, {0, 125, 255, 255}, {125, 255, 0, 255}, {255, 125, 0, 255});
            DrawGp(img, gp2, grid_map_info, {125, 125, 255, 255}, {125, 255, 125, 255}, {255, 125, 0, 255});
            ERL_INFO("GP1 at [{:f}, {:f}] has {} data points.", gp1->position.x(), gp1->position.y(), gp1->gp->GetNumTrainSamples());
            if (gp2 != nullptr) { ERL_INFO("GP2 has {} data points.", gp2->gp->GetNumTrainSamples()); }

            // draw trajectory
            erl::common::DrawTrajectoryInplace(img, cur_traj.block(0, 0, 2, i), grid_map_info, trajectory_color, 2, pixel_based);

            // draw fps
            cv::putText(img, std::to_string(1000.0 / dt), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0, 255), 2);

            if (pangolin_plotter_window != nullptr && pangolin_map_window != nullptr && pangolin_map_view != nullptr) {
                pangolin_map_window->MakeCurrent();                  // make current context
                pangolin::BindToContext(g_window_name + ": map");    // bind context
                pangolin_map_view->Activate();                       // activate view to draw in this area
                pangolin_map_window->ProcessEvents();                // process events like mouse and keyboard inputs
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // clear entire window
                // glColor4f(1.0, 1.0, 1.0, 1.0);
                pangolin_texture->Upload(img.data, GL_BGRA, GL_UNSIGNED_BYTE);
                pangolin_texture->RenderToViewportFlipY();
                pangolin::FinishFrame();
                pangolin_map_window->RemoveCurrent();

                pangolin_plotter_window->Activate(true);
                auto t = static_cast<float>(t_ms / 1000.0);
                plotter_sdf->Append(t, {static_cast<float>(distance[0]), static_cast<float>(std::abs(distance[0])), static_cast<float>(variances(0, 0))});
                plotter_grad->Append(
                    t,
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
    }

    ERL_INFO("Average update time: {:f} ms.", t_ms / static_cast<double>(max_update_cnt));
    if (g_options.save_video) {
        video_writer->release();
        ERL_INFO("Saved surface mapping video to {}.", video_path.c_str());
    }

    // Test SDF Estimation
    bool c_stride = true;
    Eigen::Matrix2Xd positions_in = grid_map_info->GenerateMeterCoordinates(c_stride);
    Eigen::VectorXd distances_out(positions_in.cols());
    Eigen::Matrix2Xd gradients_out(2, positions_in.cols());
    Eigen::Matrix3Xd variances_out(3, positions_in.cols());
    Eigen::Matrix3Xd covariances_out;
    auto t0 = std::chrono::high_resolution_clock::now();
    sdf_mapping.Test(positions_in, distances_out, gradients_out, variances_out, covariances_out);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double t_per_point = dt / static_cast<double>(positions_in.cols()) * 1000;  // us
    ERL_INFO("Test time: {:f} ms for {} points, {:f} us per point.", dt, positions_in.cols(), t_per_point);

    if (g_options.visualize) {
        Eigen::MatrixXd sdf_out_mat = distances_out.reshaped(grid_map_info->Height(), grid_map_info->Width());
        double min_distance = distances_out.minCoeff();
        double max_distance = distances_out.maxCoeff();
        Eigen::MatrixX8U distances_out_mat_normalized = ((sdf_out_mat.array() - min_distance) / (max_distance - min_distance) * 255).cast<uint8_t>();
        cv::Mat src, dst;
        cv::eigen2cv(distances_out_mat_normalized, src);
        cv::flip(src, src, 0);  // flip along y axis
        cv::applyColorMap(src, dst, cv::COLORMAP_JET);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2BGRA);
        cv::addWeighted(dst, 0.5, img, 0.5, 0.0, dst);

        pangolin::CreateWindowAndBind(g_window_name + ": sdf", dst.cols, dst.rows);
        glEnable(GL_DEPTH_TEST);
        pangolin::ImageView sdf_img_view("sdf");
        pangolin::DisplayBase().AddDisplay(sdf_img_view.SetBounds(0, 1.0, 0, 1.0, static_cast<float>(dst.cols) / static_cast<float>(dst.rows)));
        pangolin::TypedImage sdf_img(dst.cols, dst.rows, pangolin::PixelFormatFromString("BGRA32"));
        std::copy_n(dst.data, dst.cols * dst.rows * 4, sdf_img.ptr);
        sdf_img_view.SetImage(sdf_img);
        pangolin::FinishFrame();

        Eigen::MatrixXd sdf_variances_mat = variances_out.row(0).reshaped(grid_map_info->Height(), grid_map_info->Width());
        double min_variance = sdf_variances_mat.minCoeff();
        double max_variance = sdf_variances_mat.maxCoeff();
        Eigen::MatrixX8U variances_out_mat_normalized = ((sdf_variances_mat.array() - min_variance) / (max_variance - min_variance) * 255).cast<uint8_t>();
        cv::eigen2cv(variances_out_mat_normalized, src);
        cv::flip(src, src, 0);  // flip along y axis
        cv::applyColorMap(src, dst, cv::COLORMAP_JET);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2BGRA);
        cv::addWeighted(dst, 0.5, img, 0.5, 0.0, dst);

        pangolin::CreateWindowAndBind(g_window_name + ": sdf_variances", dst.cols, dst.rows);
        glEnable(GL_DEPTH_TEST);
        pangolin::ImageView sdf_variances_img_view("sdf_variances");
        pangolin::DisplayBase().AddDisplay(sdf_variances_img_view.SetBounds(0, 1.0, 0, 1.0, static_cast<float>(dst.cols) / static_cast<float>(dst.rows)));
        pangolin::TypedImage variances_img(dst.cols, dst.rows, pangolin::PixelFormatFromString("BGRA32"));
        std::copy_n(dst.data, dst.cols * dst.rows * 4, variances_img.ptr);
        sdf_variances_img_view.SetImage(variances_img);
        pangolin::FinishFrame();
    }

    surface_mapping->GetQuadtree()->WriteBinary("tree.bt");
    ERL_ASSERT(surface_mapping->GetQuadtree()->Write("tree.ot"));
    if (g_options.hold) {
        std::cout << "Press any key to exit." << std::endl;
        while (!pangolin::ShouldQuit()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            pangolin_plotter_window->Activate();
        }
    } else {
        t0 = std::chrono::high_resolution_clock::now();
        double wait_time = 10.0;
        while (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() < wait_time && !pangolin::ShouldQuit()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
            ("use-gazebo-data", po::bool_switch(&g_options.use_gazebo_data)->default_value(g_options.use_gazebo_data), "Use Gazebo data")
            ("use-house-expo-data", po::bool_switch(&g_options.use_house_expo_data)->default_value(g_options.use_house_expo_data), "Use HouseExpo data")
            ("use-ros-bag-data", po::bool_switch(&g_options.use_ros_bag_data)->default_value(g_options.use_ros_bag_data), "Use ROS bag data")
            ("stride", po::value<int>(&g_options.stride)->default_value(g_options.stride), "stride for running the sequence")
            ("map-resolution", po::value<double>(&g_options.map_resolution)->default_value(g_options.map_resolution), "Map resolution")
            ("surf-normal-scale", po::value<double>(&g_options.surf_normal_scale)->default_value(g_options.surf_normal_scale), "Surface normal scale")
            ("init-frame", po::value<int>(&g_options.init_frame)->default_value(g_options.init_frame), "Initial frame index")
            ("visualize", po::bool_switch(&g_options.visualize)->default_value(g_options.visualize), "Visualize the mapping")
            ("save-video", po::bool_switch(&g_options.save_video)->default_value(g_options.save_video), "Save the mapping video")
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
                po::value<std::string>(&g_options.ros_bag_dat_file)->default_value(g_options.ros_bag_dat_file)->value_name("file"),
                "ROS bag dat file"
            )(
                "surface-mapping-config-file",
                po::value<std::string>(&g_options.surface_mapping_config_file)->default_value(g_options.surface_mapping_config_file)->value_name("file"),
                "Surface mapping config file"
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
        if (g_options.use_gazebo_data + g_options.use_house_expo_data + g_options.use_ros_bag_data != 1) {
            std::cerr << "Please specify one of --use-gazebo-data, --use-house-expo-data, --use-ros-bag-data." << std::endl;
            return 0;
        }
        if (g_options.use_gazebo_data && !std::filesystem::exists(g_options.gazebo_train_file)) {
            std::cerr << "Gazebo train data file " << g_options.gazebo_train_file << " does not exist." << std::endl;
            return 0;
        }
        if (g_options.use_house_expo_data &&
            (!std::filesystem::exists(g_options.house_expo_map_file) || !std::filesystem::exists(g_options.house_expo_traj_file))) {
            std::cerr << "HouseExpo map file " << g_options.house_expo_map_file << " or trajectory file " << g_options.house_expo_traj_file
                      << " does not exist." << std::endl;
            return 0;
        }
        if (g_options.use_ros_bag_data && !std::filesystem::exists(g_options.ros_bag_dat_file)) {
            std::cerr << "ROS bag dat file " << g_options.ros_bag_dat_file << " does not exist." << std::endl;
            return 0;
        }
    } catch (std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    g_window_name = argv[0];
    if (g_options.save_video) { g_options.visualize = true; }
    std::filesystem::create_directories(g_options.output_dir);
    return RUN_ALL_TESTS();
}
