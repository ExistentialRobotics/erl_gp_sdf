#include "erl_geometry/house_expo.hpp"
#include "erl_geometry/lidar_2d.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping_2d.hpp"
#include "erl_geometry/occupancy_quadtree_drawer.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_2d.hpp"
#include "erl_common/csv.hpp"
#include "erl_geometry/gazebo_room.hpp"
#include "erl_common/random.hpp"
#include "erl_common/progress_bar.hpp"
#include <boost/program_options.hpp>
#include <gtest/gtest.h>
#include <filesystem>
#include <pangolin/display/display.h>
#include <pangolin/display/image_view.h>
#include <pangolin/plot/plotter.h>

static std::filesystem::path g_file_path = __FILE__;
static std::filesystem::path g_dir_path = g_file_path.parent_path();
static std::filesystem::path g_config_dir_path = g_dir_path.parent_path().parent_path() / "config";
static std::string g_window_name = "ERL_SDF_MAPPING";

struct Options {
    std::string gazebo_train_file = (g_dir_path / "gazebo_train.dat").string();
    std::string gazebo_test_file = (g_dir_path / "gazebo_test.dat").string();
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
    const std::shared_ptr<erl::sdf_mapping::GpSdfMapping2D::GP> &gp1,
    const std::shared_ptr<erl::common::GridMapInfo2D> &grid_map_info,
    const cv::Scalar &data_color = {0, 255, 125, 255},
    const cv::Scalar &pos_color = {125, 255, 0, 255},
    const cv::Scalar &rect_color = {125, 255, 0, 255}) {

    if (gp1 == nullptr) { return; }

    Eigen::Vector2i gp1_position_px = grid_map_info->MeterToPixelForPoints(gp1->position);
    cv::drawMarker(img, cv::Point(gp1_position_px[0], gp1_position_px[1]), pos_color, cv::MARKER_STAR, 10, 1);
    Eigen::Vector2d gp1_area_min = gp1->position.array() - gp1->half_size;
    Eigen::Vector2d gp1_area_max = gp1->position.array() + gp1->half_size;
    Eigen::Vector2i gp1_area_min_px = grid_map_info->MeterToPixelForPoints(gp1_area_min);
    Eigen::Vector2i gp1_area_max_px = grid_map_info->MeterToPixelForPoints(gp1_area_max);
    cv::rectangle(img, cv::Point(gp1_area_min_px[0], gp1_area_min_px[1]), cv::Point(gp1_area_max_px[0], gp1_area_max_px[1]), rect_color, 2);

    Eigen::Matrix2Xd used_surface_points = gp1->gp->GetTrainInputSamplesBuffer().block(0, 0, 2, gp1->gp->GetNumTrainSamples());
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
        max_update_cnt = long(train_data_loader.size() - g_options.init_frame) / g_options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        erl::common::ProgressBar bar(int(max_update_cnt), true, std::cout);
        int cnt = 0;
        for (int i = g_options.init_frame; i < int(train_data_loader.size()); i += g_options.stride, ++cnt) {
            auto &df = train_data_loader[i];
            train_angles.push_back(df.angles);
            train_ranges.push_back(df.distances);
            train_poses.emplace_back(df.pose_numpy);
            cur_traj.col(cnt) << df.pose_numpy(0, 2), df.pose_numpy(1, 2);
            double &x = cur_traj(0, cnt);
            double &y = cur_traj(1, cnt);
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
            bar.Update();
        }
        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else if (g_options.use_house_expo_data) {
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
        max_update_cnt = long(trajectory.size() - g_options.init_frame) / g_options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        erl::common::ProgressBar bar(int(max_update_cnt), true, std::cout);
        int cnt = 0;
        for (std::size_t i = g_options.init_frame; i < trajectory.size(); i += g_options.stride, cnt++) {
            std::vector<double> &waypoint = trajectory[i];
            cur_traj.col(long(cnt)) << waypoint[0], waypoint[1];
            lidar.SetTranslation(cur_traj.col(long(cnt)));
            lidar.SetRotation(waypoint[2]);
            auto lidar_ranges = lidar.Scan(scan_in_parallel);
            lidar_ranges += erl::common::GenerateGaussianNoise(lidar_ranges.size(), 0.0, 0.01);
            train_angles.push_back(lidar.GetAngles());
            train_ranges.push_back(lidar_ranges);
            train_poses.emplace_back(lidar.GetPose().topRows<2>());
            bar.Update();
        }
        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else if (g_options.use_ros_bag_data) {
        Eigen::MatrixXd ros_bag_data = erl::common::LoadEigenMatrixFromBinaryFile<double, Eigen::Dynamic, Eigen::Dynamic>(g_options.ros_bag_dat_file);
        tic = ros_bag_data(1, 0) - ros_bag_data(0, 0);
        // prepare buffer
        max_update_cnt = long(ros_bag_data.rows() - g_options.init_frame) / g_options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        // load data into buffer
        long num_rays = (ros_bag_data.cols() - 7) / 2;
        erl::common::ProgressBar bar(int(max_update_cnt), true, std::cout);
        long cnt = 0;
        for (long i = g_options.init_frame; i < ros_bag_data.rows(); i += g_options.stride, cnt++) {
            Eigen::Matrix23d pose = ros_bag_data.row(i).segment(1, 6).reshaped(3, 2).transpose();
            cur_traj.col(cnt) << pose(0, 2), pose(1, 2);
            train_angles.emplace_back(ros_bag_data.row(i).segment(7, num_rays));
            train_ranges.emplace_back(ros_bag_data.row(i).segment(7 + num_rays, num_rays));
            train_poses.push_back(pose);
            Eigen::VectorXd &angles = train_angles.back();
            Eigen::VectorXd &ranges = train_ranges.back();
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
            bar.Update();
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
        Eigen::Vector2i normal_px = grid_map_info->MeterToPixelForVectors(it->GetSurfaceData()->normal * g_options.surf_normal_scale);
        cv::Point arrow_end_px(position_px[0] + normal_px[0], position_px[1] + normal_px[1]);
        arrowed_lines.emplace_back(position_px_cv, arrow_end_px);
    });
    drawer->SetDrawLeafCallback([&](const OccupancyQuadtreeDrawer *self, cv::Mat &img, erl::sdf_mapping::SurfaceMappingQuadtree::LeafIterator &it) {
        if (it->GetSurfaceData() == nullptr) { return; }
        auto grid_map_info = self->GetGridMapInfo();
        Eigen::Vector2i position_px = grid_map_info->MeterToPixelForPoints(it->GetSurfaceData()->position);
        cv::Point position_px_cv(position_px[0], position_px[1]);
        cv::circle(img, position_px_cv, 1, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
        Eigen::Vector2i normal_px = grid_map_info->MeterToPixelForVectors(it->GetSurfaceData()->normal * g_options.surf_normal_scale);
        cv::Point arrow_end_px(position_px[0] + normal_px[0], position_px[1] + normal_px[1]);
        arrowed_lines.emplace_back(position_px_cv, arrow_end_px);
    });

    bool pixel_based = true;
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
    pangolin::WindowInterface *pangolin_plotter_window = nullptr;
    std::shared_ptr<pangolin::DataLog> pangolin_log = nullptr;
    std::shared_ptr<pangolin::Plotter> pangolin_plotter;
    pangolin::View *pangolin_plotter_view = nullptr;
    pangolin::WindowInterface *pangolin_map_window = nullptr;
    pangolin::View* pangolin_map_view = nullptr;
    std::shared_ptr<pangolin::GlTexture> pangolin_texture = nullptr;
    if (g_options.visualize) {
        pangolin_plotter_window = &pangolin::CreateWindowAndBind(g_window_name + ": curves", 1280, 960);
        glEnable(GL_DEPTH_TEST);
        pangolin_log = std::make_shared<pangolin::DataLog>();
        std::vector<std::string> pangolin_labels = {"SDF", "EDF", "var(SDF)", "var(gradX)", "var(gradY)"};
        pangolin_log->SetLabels(pangolin_labels);
        float pangolin_bound_left = 0.0f;
        float pangolin_bound_right = 600.0f;
        float pangolin_bound_bottom = -1.0f;
        float pangolin_bound_top = 3.0f;
        pangolin_plotter = std::make_shared<pangolin::Plotter>(
            pangolin_log.get(),
            pangolin_bound_left,
            pangolin_bound_right,
            pangolin_bound_bottom,
            pangolin_bound_top,
            float(tic),
            0.05f);
        pangolin_plotter_view = &pangolin_plotter->SetBounds(0.0, 1.0, 0.0, 1.0);
        pangolin_plotter->Track("$i");
        // pangolin_plotter->AddMarker(pangolin::Marker::Vertical, -1000, pangolin::Marker::LessThan, pangolin::Colour::Blue().WithAlpha(0.2f));
        // pangolin_plotter->AddMarker(pangolin::Marker::Horizontal, 100, pangolin::Marker::GreaterThan, pangolin::Colour::Red().WithAlpha(0.2f));
        // pangolin_plotter->AddMarker(pangolin::Marker::Horizontal, 10, pangolin::Marker::Equal, pangolin::Colour::Green().WithAlpha(0.2f));
        pangolin::DisplayBase().AddDisplay(*pangolin_plotter);

        pangolin_map_window = &pangolin::CreateWindowAndBind(g_window_name + ": map", img.cols, img.rows);
        glEnable(GL_DEPTH_TEST);
        pangolin_map_view = &(pangolin::Display("cam").SetBounds(0, 1.0, 0, 1.0, float(img.cols) / float(img.rows)));
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

    double t = 0;
    for (long i = 0; i < max_update_cnt; i++) {
        Eigen::VectorXd noise = erl::common::GenerateGaussianNoise(train_ranges[i].size(), 0.0, 0.01);

        auto t0 = std::chrono::high_resolution_clock::now();
        sdf_mapping.Update(train_angles[i], train_ranges[i], train_poses[i]);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("Update time: %f ms.", dt);
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
            auto radius = int(std::abs(distance[0]) / grid_map_info->Resolution(0));
            cv::Mat circle_layer(img.rows, img.cols, CV_8UC4, cv::Scalar(0));
            cv::Mat circle_mask(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
            cv::circle(circle_mask, position_px, radius, cv::Scalar(255), cv::FILLED);
            cv::circle(circle_layer, position_px, radius, cv::Scalar(0, 255, 0, 25), cv::FILLED);
            cv::add(img * 0.5, circle_layer * 0.5, img, circle_mask);

            // draw sdf gradient
            Eigen::Vector2i gradient_px = grid_map_info->MeterToPixelForVectors(gradient);
            cv::arrowedLine(img, position_px, cv::Point(position_px.x + gradient_px.x(), position_px.y + gradient_px.y()), cv::Scalar(255, 0, 0, 255), 2, 8, 0, 0.15);

            // draw used surface points
            auto &[gp1, gp2] = sdf_mapping.GetUsedGps()[0];
            DrawGp(img, gp1, grid_map_info, {0, 125, 255, 255}, {125, 255, 0, 255}, {255, 125, 0, 255});
            DrawGp(img, gp2, grid_map_info, {125, 125, 255, 255}, {125, 255, 125, 255}, {255, 125, 0, 255});
            ERL_INFO("GP1 at [%f, %f] has %ld data points.", gp1->position.x(), gp1->position.y(), gp1->gp->GetNumTrainSamples());
            if (gp2 != nullptr) { ERL_INFO("GP2 has %ld data points.", gp2->gp->GetNumTrainSamples()); }

            // draw trajectory
            erl::common::DrawTrajectoryInplace(img, cur_traj.block(0, 0, 2, i), grid_map_info, trajectory_color, 2, pixel_based);

            // draw fps
            cv::putText(img, std::to_string(1000.0 / dt), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0, 255), 2);

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

            pangolin_plotter_window->MakeCurrent();
            pangolin::BindToContext(g_window_name + ": curves");
            pangolin_plotter_view->Activate();
            pangolin_plotter_window->ProcessEvents();            // process events like mouse and keyboard inputs
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // clear entire window
            // pangolin_log->Log(float(std::abs(distance[0])));
            pangolin_log->Log(float(distance[0]), float(std::abs(distance[0])), float(variances(0, 0)), float(variances(1, 0)), float(variances(2, 0)));
            pangolin::FinishFrame();
            if (g_options.save_video) {
                pangolin::TypedImage buffer = pangolin::ReadFramebuffer(pangolin_plotter_view->v, "BGR24");
                cv::Mat tmp(int(buffer.h), int(buffer.w), CV_8UC3, buffer.ptr);
                cv::flip(tmp, tmp, 0);
                int offset = (video_frame.rows - tmp.rows) / 2;
                tmp.copyTo(video_frame(cv::Rect(img.cols, offset, tmp.cols, tmp.rows)));
                cv::cvtColor(img, tmp, cv::COLOR_BGRA2BGR);
                tmp.copyTo(video_frame(cv::Rect(0, 0, tmp.cols, tmp.rows)));
                video_writer->write(video_frame);
            }
            pangolin_plotter_window->RemoveCurrent();
        }
        std::cout << "=====================================" << std::endl;
    }

    ERL_INFO("Average update time: %f ms.", t / double(max_update_cnt));
    if (g_options.save_video) {
        video_writer->release();
        ERL_INFO("Saved surface mapping video to %s.", video_path.c_str());
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
    double t_per_point = double(dt) / (double) positions_in.cols() * 1000;  // us
    ERL_INFO("Test time: %f ms for %ld points, %f us per point.", dt, positions_in.cols(), t_per_point);

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
        pangolin::DisplayBase().AddDisplay(sdf_img_view.SetBounds(0, 1.0, 0, 1.0, float(dst.cols) / float(dst.rows)));
        pangolin::TypedImage sdf_img(dst.cols, dst.rows, pangolin::PixelFormatFromString("BGRA32"));
        std::copy(dst.data, dst.data + dst.cols * dst.rows * 4, sdf_img.ptr);
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
        pangolin::DisplayBase().AddDisplay(sdf_variances_img_view.SetBounds(0, 1.0, 0, 1.0, float(dst.cols) / float(dst.rows)));
        pangolin::TypedImage variances_img(dst.cols, dst.rows, pangolin::PixelFormatFromString("BGRA32"));
        std::copy(dst.data, dst.data + dst.cols * dst.rows * 4, variances_img.ptr);
        sdf_variances_img_view.SetImage(variances_img);
        pangolin::FinishFrame();
    }

    surface_mapping->GetQuadtree()->WriteBinary("tree.bt");
    ERL_ASSERT(surface_mapping->GetQuadtree()->Write("tree.ot"));
    if (g_options.hold) {
        std::cout << "Press any key to exit." << std::endl;
        while (!pangolin::ShouldQuit()) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
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
                "gazebo-test-file",
                po::value<std::string>(&g_options.gazebo_test_file)->default_value(g_options.gazebo_test_file)->value_name("file"),
                "Gazebo test data file"
            )(
                "ros-bag-csv-file",
                po::value<std::string>(&g_options.ros_bag_dat_file)->default_value(g_options.ros_bag_dat_file)->value_name("file"),
                "ROS bag csv file"
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
    } catch (std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    g_window_name = argv[0];
    if (g_options.save_video) { g_options.visualize = true; }
    std::filesystem::create_directories(g_options.output_dir);
    return RUN_ALL_TESTS();
}
