#include "erl_common/csv.hpp"
#include "erl_common/plplot_fig.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_geometry/gazebo_room_2d.hpp"
#include "erl_geometry/house_expo_map.hpp"
#include "erl_geometry/lidar_2d.hpp"
#include "erl_geometry/occupancy_quadtree_drawer.hpp"
#include "erl_geometry/trajectory.hpp"
#include "erl_geometry/ucsd_fah_2d.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping.hpp"

#include <boost/program_options.hpp>

const std::filesystem::path kProjectRootDir = ERL_SDF_MAPPING_ROOT_DIR;
int g_argc = 0;
char **g_argv = nullptr;

template<typename Dtype, typename Gp, typename Drawer>
void
DrawGp(
    cv::Mat &img,
    const std::shared_ptr<Gp> &gp,
    const Drawer &drawer,
    const cv::Scalar &data_color = {0, 255, 125, 255},
    const cv::Scalar &pos_color = {125, 255, 0, 255},
    const cv::Scalar &rect_color = {125, 255, 0, 255}) {

    if (gp == nullptr) { return; }

    Eigen::Vector2i gp1_position_px = drawer.GetPixelCoordsForPositions(gp->position, true);
    cv::drawMarker(img, cv::Point(gp1_position_px[0], gp1_position_px[1]), pos_color, cv::MARKER_STAR, 10, 1);
    const Eigen::Vector2<Dtype> gp1_area_min = gp->position.array() - gp->half_size;
    const Eigen::Vector2<Dtype> gp1_area_max = gp->position.array() + gp->half_size;
    Eigen::Vector2i gp1_area_min_px = drawer.GetPixelCoordsForPositions(gp1_area_min, true);
    Eigen::Vector2i gp1_area_max_px = drawer.GetPixelCoordsForPositions(gp1_area_max, true);
    cv::rectangle(img, cv::Point(gp1_area_min_px[0], gp1_area_min_px[1]), cv::Point(gp1_area_max_px[0], gp1_area_max_px[1]), rect_color, 2);

    typename erl::gaussian_process::NoisyInputGaussianProcess<Dtype>::TrainSet &train_set = gp->edf_gp->GetTrainSet();
    const Eigen::Matrix2X<Dtype> used_surface_points = train_set.x.block(0, 0, 2, train_set.num_samples);
    Eigen::Matrix2Xi used_surface_points_px = drawer.GetPixelCoordsForPositions(used_surface_points, true);
    for (long j = 0; j < used_surface_points.cols(); j++) {
        cv::circle(img, cv::Point(used_surface_points_px(0, j), used_surface_points_px(1, j)), 3, data_color, -1);
    }
}

template<typename Dtype, typename Drawer>
struct OpenCvUserData {
    std::string window_name;
    Drawer *drawer = nullptr;
    erl::sdf_mapping::GpSdfMapping<Dtype, 2, erl::sdf_mapping::GpOccSurfaceMapping<Dtype, 2>> *sdf_mapping = nullptr;
    cv::Mat img;
};

template<typename Dtype, typename Drawer>
void
OpenCvMouseCallback(const int event, const int x, const int y, int /*flags*/, void *userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        auto *data = static_cast<OpenCvUserData<Dtype, Drawer> *>(userdata);
        Eigen::Vector2<Dtype> position = data->drawer->GetMeterCoordsForPositions(Eigen::Vector2i(x, y), false);
        ERL_INFO("Clicked at [{:f}, {:f}].", position.x(), position.y());
        Eigen::VectorX<Dtype> distance(1);
        Eigen::Matrix2X<Dtype> gradient(2, 1);
        Eigen::Matrix3X<Dtype> variances(3, 1);
        Eigen::Matrix3X<Dtype> covariances(3, 1);
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
            const Dtype resolution = data->drawer->GetGridMapInfo()->Resolution(0);
            auto radius = static_cast<int>(std::abs(distance[0]) / resolution);
            cv::Mat circle_layer(img.rows, img.cols, CV_8UC4, cv::Scalar(0));
            cv::Mat circle_mask(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
            cv::circle(circle_mask, cv::Point2i(x, y), radius, cv::Scalar(255), cv::FILLED);
            cv::circle(circle_layer, cv::Point2i(x, y), radius, cv::Scalar(0, 255, 0, 25), cv::FILLED);
            cv::add(img * 0.5, circle_layer * 0.5, img, circle_mask);

            // draw sdf variance
            radius = static_cast<int>((std::sqrt(variances(0, 0)) + std::abs(distance[0])) / resolution);
            cv::circle(img, cv::Point2i(x, y), radius, cv::Scalar(0, 0, 255, 25), 1);

            // draw sdf gradient
            Eigen::VectorXi grad_pixel = data->drawer->GetPixelCoordsForVectors(gradient.col(0));
            cv::arrowedLine(img, cv::Point(x, y), cv::Point(x + grad_pixel[0], y + grad_pixel[1]), cv::Scalar(255, 0, 0, 255), 1);

            auto &[gp1, gp2] = data->sdf_mapping->GetUsedGps()[0];
            DrawGp<Dtype>(img, gp1, *data->drawer);
            DrawGp<Dtype>(img, gp2, *data->drawer);

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

            cv::imshow(data->window_name, img);
        } else {
            ERL_WARN("Failed to test SDF estimation at [{:f}, {:f}].", position.x(), position.y());
        }
    }
}

cv::Point
EigenToOpenCV(const Eigen::Vector2i &p) {
    return {p.x(), p.y()};
}

template<typename Dtype>
void
TestImpl2D() {
    GTEST_PREPARE_OUTPUT_DIR();
    using namespace erl::common;

    using SurfaceMapping = erl::sdf_mapping::GpOccSurfaceMapping<Dtype, 2>;
    using SdfMapping = erl::sdf_mapping::GpSdfMapping<Dtype, 2, SurfaceMapping>;
    using SurfaceMappingQuadtree = erl::sdf_mapping::SurfaceMappingQuadtree<Dtype>;
    using QuadtreeDrawer = erl::geometry::OccupancyQuadtreeDrawer<SurfaceMappingQuadtree>;
    using Lidar2D = erl::geometry::Lidar2D;

    using Matrix2 = Eigen::Matrix2<Dtype>;
    using Matrix2X = Eigen::Matrix2X<Dtype>;
    using Matrix3X = Eigen::Matrix3X<Dtype>;
    using MatrixX = Eigen::MatrixX<Dtype>;
    using MatrixX8U = Eigen::MatrixX<uint8_t>;
    using Vector2 = Eigen::Vector2<Dtype>;
    using VectorX = Eigen::VectorX<Dtype>;

    struct Options {
        std::string gazebo_train_file = kProjectRootDir / "data" / "gazebo";
        std::string house_expo_map_file = kProjectRootDir / "data" / "house_expo_room_1451.json";
        std::string house_expo_traj_file = kProjectRootDir / "data" / "house_expo_room_1451.csv";
        std::string ucsd_fah_2d_file = kProjectRootDir / "data" / "ucsd_fah_2d.dat";
        std::string surface_mapping_config_file = kProjectRootDir / "config" / "surface_mapping_2d.yaml";
        std::string sdf_mapping_config_file = kProjectRootDir / "config" / "sdf_mapping_2d.yaml";
        bool use_gazebo_room_2d = false;
        bool use_house_expo_lidar_2d = false;
        bool use_ucsd_fah_2d = false;
        bool visualize = false;
        bool test_io = false;
        bool hold = false;
        bool interactive = false;
        bool save_video = false;
        int init_frame = 0;
        int stride = 1;
        Dtype map_resolution = 0.025;
        Dtype surf_normal_scale = 0.35;
    };

    Options options;
    bool options_parsed = false;
    try {
        namespace po = boost::program_options;
        po::options_description desc;
        // clang-format off
        desc.add_options()
            ("help", "produce help message")
            ("use-gazebo-room-2d", po::bool_switch(&options.use_gazebo_room_2d)->default_value(options.use_gazebo_room_2d), "Use Gazebo data")
            ("use-house-expo-lidar-2d", po::bool_switch(&options.use_house_expo_lidar_2d)->default_value(options.use_house_expo_lidar_2d), "Use HouseExpo data")
            ("use-ucsd-fah-2d", po::bool_switch(&options.use_ucsd_fah_2d)->default_value(options.use_ucsd_fah_2d), "Use ROS bag data")
            ("visualize", po::bool_switch(&options.visualize)->default_value(options.visualize), "Visualize the mapping")
            ("test-io", po::bool_switch(&options.test_io)->default_value(options.test_io), "Test IO")
            ("hold", po::bool_switch(&options.hold)->default_value(options.hold), "Hold the test until a key is pressed")
            ("interactive", po::bool_switch(&options.interactive)->default_value(options.interactive), "Interactive mode")
            ("save-video", po::bool_switch(&options.save_video)->default_value(options.save_video), "Save the mapping video")
            ("init-frame", po::value<int>(&options.init_frame)->default_value(options.init_frame), "Initial frame index")
            ("stride", po::value<int>(&options.stride)->default_value(options.stride), "stride for running the sequence")
            ("map-resolution", po::value<Dtype>(&options.map_resolution)->default_value(options.map_resolution), "Map resolution")
            ("surf-normal-scale", po::value<Dtype>(&options.surf_normal_scale)->default_value(options.surf_normal_scale), "Surface normal scale")
            (
                "gazebo-train-file",
                po::value<std::string>(&options.gazebo_train_file)->default_value(options.gazebo_train_file)->value_name("file"),
                "Gazebo train data file"
            )(
                "house-expo-map-file",
                po::value<std::string>(&options.house_expo_map_file)->default_value(options.house_expo_map_file)->value_name("file"),
                "HouseExpo map file"
            )(
                "house-expo-traj-file",
                po::value<std::string>(&options.house_expo_traj_file)->default_value(options.house_expo_traj_file)->value_name("file"),
                "HouseExpo trajectory file"
            )(
                "ucsd-fah-2d-file",
                po::value<std::string>(&options.ucsd_fah_2d_file)->default_value(options.ucsd_fah_2d_file)->value_name("file"),
                "ROS bag dat file"
            )(
                "surface-mapping-config-file",
                po::value<std::string>(&options.surface_mapping_config_file)->default_value(options.surface_mapping_config_file)->value_name("file"),
                "surface mapping config file"
            )(
                "sdf-mapping-config-file",
                po::value<std::string>(&options.sdf_mapping_config_file)->default_value(options.sdf_mapping_config_file)->value_name("file"),
                "SDF mapping config file");
        // clang-format on

        po::variables_map vm;
        po::store(po::command_line_parser(g_argc, g_argv).options(desc).run(), vm);
        if (vm.count("help")) {
            std::cout << "Usage: " << g_argv[0] << " [options]" << std::endl << desc << std::endl;
            return;
        }
        po::notify(vm);
        options_parsed = true;
    } catch (std::exception &e) { std::cerr << e.what() << std::endl; }
    ASSERT_TRUE(options_parsed);

    ASSERT_TRUE(options.use_gazebo_room_2d || options.use_house_expo_lidar_2d || options.use_ucsd_fah_2d)
        << "Please specify one of --use-gazebo-room-2d, --use-house-expo-lidar-2d, --use-ucsd-fah-2d.";
    if (options.use_gazebo_room_2d) {
        ASSERT_TRUE(std::filesystem::exists(options.gazebo_train_file)) << "Gazebo train data file " << options.gazebo_train_file << " does not exist.";
    }
    if (options.use_house_expo_lidar_2d) {
        ASSERT_TRUE(std::filesystem::exists(options.house_expo_map_file)) << "HouseExpo map file " << options.house_expo_map_file << " does not exist.";
        ASSERT_TRUE(std::filesystem::exists(options.house_expo_traj_file))
            << "HouseExpo trajectory file " << options.house_expo_traj_file << " does not exist.";
    }
    if (options.use_ucsd_fah_2d) {
        ASSERT_TRUE(std::filesystem::exists(options.ucsd_fah_2d_file)) << "ROS bag dat file " << options.ucsd_fah_2d_file << " does not exist.";
    }

    // load the scene
    long max_update_cnt;
    std::vector<VectorX> train_angles;
    std::vector<VectorX> train_ranges;
    std::vector<std::pair<Matrix2, Vector2>> train_poses;
    Vector2 map_min(0, 0);
    Vector2 map_max(0, 0);
    Vector2 map_resolution(options.map_resolution, options.map_resolution);
    Eigen::Vector2i map_padding(10, 10);
    Matrix2X cur_traj;
    double tic = 0.2;

    if (options.use_gazebo_room_2d) {
        auto train_data_loader = erl::geometry::GazeboRoom2D::TrainDataLoader(options.gazebo_train_file);
        max_update_cnt = static_cast<long>(train_data_loader.size() - options.init_frame) / options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        auto bar_setting = std::make_shared<ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = ProgressBar::Open(bar_setting, std::cout);
        int cnt = 0;
        for (int i = options.init_frame; i < static_cast<int>(train_data_loader.size()); i += options.stride, ++cnt) {
            auto &df = train_data_loader[i];
            train_angles.push_back(df.angles.cast<Dtype>());
            train_ranges.push_back(df.ranges.cast<Dtype>());
            train_poses.emplace_back(df.rotation.cast<Dtype>(), df.translation.cast<Dtype>());
            cur_traj.col(cnt) << df.translation.cast<Dtype>();
            bar->Update();
        }
        map_min = erl::geometry::GazeboRoom2D::kMapMin.cast<Dtype>();
        map_max = erl::geometry::GazeboRoom2D::kMapMax.cast<Dtype>();
        map_padding.setZero();

        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else if (options.use_house_expo_lidar_2d) {
        erl::geometry::HouseExpoMap house_expo_map(options.house_expo_map_file, 0.2);
        map_min = house_expo_map.GetMeterSpace()->GetSurface()->vertices.rowwise().minCoeff().cast<Dtype>();
        map_max = house_expo_map.GetMeterSpace()->GetSurface()->vertices.rowwise().maxCoeff().cast<Dtype>();
        auto lidar_setting = std::make_shared<Lidar2D::Setting>();
        lidar_setting->num_lines = 720;
        Lidar2D lidar(lidar_setting, house_expo_map.GetMeterSpace());
        auto trajectory = LoadAndCastCsvFile<double>(options.house_expo_traj_file.c_str(), [](const std::string &str) -> double { return std::stod(str); });
        max_update_cnt = static_cast<long>(trajectory.size() - options.init_frame) / options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        auto bar_setting = std::make_shared<ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = ProgressBar::Open(bar_setting, std::cout);
        int cnt = 0;
        for (std::size_t i = options.init_frame; i < trajectory.size(); i += options.stride, cnt++) {
            bool scan_in_parallel = true;
            std::vector<double> &waypoint = trajectory[i];
            cur_traj.col(cnt) << static_cast<Dtype>(waypoint[0]), static_cast<Dtype>(waypoint[1]);

            Eigen::Matrix2d rotation = Eigen::Rotation2Dd(waypoint[2]).toRotationMatrix();
            Eigen::Vector2d translation(waypoint[0], waypoint[1]);
            VectorX lidar_ranges = lidar.Scan(rotation, translation, scan_in_parallel).cast<Dtype>();
            lidar_ranges += GenerateGaussianNoise<Dtype>(lidar_ranges.size(), 0.0, 0.01);
            train_angles.push_back(lidar.GetAngles().cast<Dtype>());
            train_ranges.push_back(lidar_ranges);
            train_poses.emplace_back(rotation.cast<Dtype>(), translation.cast<Dtype>());
            bar->Update();
        }
        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else if (options.use_ucsd_fah_2d) {
        erl::geometry::UcsdFah2D ucsd_fah(options.ucsd_fah_2d_file);
        tic = ucsd_fah.GetTimeStep();
        map_min = erl::geometry::UcsdFah2D::kMapMin.cast<Dtype>();
        map_max = erl::geometry::UcsdFah2D::kMapMax.cast<Dtype>();
        // prepare buffer
        max_update_cnt = (ucsd_fah.Size() - options.init_frame) / options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        // load data into buffer
        auto bar_setting = std::make_shared<ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = ProgressBar::Open(bar_setting, std::cout);
        long cnt = 0;
        for (long i = options.init_frame; i < ucsd_fah.Size(); i += options.stride, ++cnt) {
            auto [sequence_number, timestamp, header_timestamp, rotation, translation, angles, ranges] = ucsd_fah[i];
            cur_traj.col(cnt) << translation.cast<Dtype>();
            train_angles.emplace_back(angles.cast<Dtype>());
            train_ranges.emplace_back(ranges.cast<Dtype>());
            train_poses.emplace_back(rotation.cast<Dtype>(), translation.cast<Dtype>());
            bar->Update();
        }
        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else {
        return;
    }
    max_update_cnt = cur_traj.cols();

    // load setting
    const auto surface_mapping_setting = std::make_shared<typename SurfaceMapping::Setting>();
    ERL_ASSERTM(
        surface_mapping_setting->FromYamlFile(options.surface_mapping_config_file),
        "Failed to load surface_mapping_config_file: {}",
        options.surface_mapping_config_file);

    ASSERT_TRUE(surface_mapping_setting->FromYamlFile(options.surface_mapping_config_file));
    surface_mapping_setting->sensor_gp->sensor_frame->angle_min = train_angles[0].minCoeff();
    surface_mapping_setting->sensor_gp->sensor_frame->angle_max = train_angles[0].maxCoeff();
    surface_mapping_setting->sensor_gp->sensor_frame->num_rays = train_angles[0].size();
    std::shared_ptr<SurfaceMapping> surface_mapping = std::make_shared<SurfaceMapping>(surface_mapping_setting);

    const auto sdf_mapping_setting = std::make_shared<typename SdfMapping::Setting>();
    ERL_ASSERTM(
        sdf_mapping_setting->FromYamlFile(options.sdf_mapping_config_file),
        "Failed to load sdf_mapping_config_file: {}",
        options.sdf_mapping_config_file);
    sdf_mapping_setting->test_query.compute_covariance = true;
    sdf_mapping_setting->test_query.use_global_buffer = true;
    SdfMapping sdf_mapping(sdf_mapping_setting, surface_mapping);

    // prepare the visualizer
    auto drawer_setting = std::make_shared<typename QuadtreeDrawer::Setting>();
    drawer_setting->area_min = map_min;
    drawer_setting->area_max = map_max;
    drawer_setting->resolution = map_resolution[0];
    drawer_setting->scaling = surface_mapping_setting->scaling;
    drawer_setting->padding = map_padding[0];
    drawer_setting->border_color = cv::Scalar(255, 0, 0, 255);
    QuadtreeDrawer drawer(drawer_setting);

    std::vector<std::pair<cv::Point, cv::Point>> arrowed_lines;
    auto &surface_data_manager = surface_mapping->GetSurfaceDataManager();
    drawer.SetDrawTreeCallback([&](const QuadtreeDrawer *, cv::Mat &img, typename SurfaceMappingQuadtree::TreeIterator &it) {
        if (it->GetDepth() == surface_mapping_setting->cluster_depth) {
            Eigen::Vector2i position_px = drawer.GetPixelCoordsForPositions(Vector2(it.GetX(), it.GetY()), true);
            const cv::Point position_px_cv(position_px[0], position_px[1]);
            cv::circle(img, position_px_cv, 2, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
            return;
        }
        if (!it->HasSurfaceData()) { return; }
        auto &surface_data = surface_data_manager[it->surface_data_index];
        Eigen::Vector2i position_px = drawer.GetPixelCoordsForPositions(surface_data.position, true);
        cv::Point position_px_cv(position_px[0], position_px[1]);
        cv::circle(img, cv::Point(position_px[0], position_px[1]), 1, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
        Eigen::Vector2i normal_px = drawer.GetPixelCoordsForVectors(surface_data.normal * options.surf_normal_scale);
        cv::Point arrow_end_px(position_px[0] + normal_px[0], position_px[1] + normal_px[1]);
        arrowed_lines.emplace_back(position_px_cv, arrow_end_px);
    });
    drawer.SetDrawLeafCallback([&](const QuadtreeDrawer *, cv::Mat &img, typename SurfaceMappingQuadtree::LeafIterator &it) {
        if (!it->HasSurfaceData()) { return; }
        auto &surface_data = surface_data_manager[it->surface_data_index];
        Eigen::Vector2i position_px = drawer.GetPixelCoordsForPositions(surface_data.position, true);
        cv::Point position_px_cv(position_px[0], position_px[1]);
        cv::circle(img, position_px_cv, 2, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
        Eigen::Vector2i normal_px = drawer.GetPixelCoordsForVectors(surface_data.normal * options.surf_normal_scale);
        cv::Point arrow_end_px(position_px[0] + normal_px[0], position_px[1] + normal_px[1]);
        arrowed_lines.emplace_back(position_px_cv, arrow_end_px);
    });

    cv::Scalar trajectory_color(0, 0, 0, 255);
    cv::Mat img;
    bool drawer_connected = false;
    const bool update_occupancy = surface_mapping_setting->update_occupancy;
    if (options.visualize) {
        if (update_occupancy) {
            drawer.DrawLeaves(img);
        } else {
            drawer.DrawTree(img);
        }
    }
    if (options.hold) {
        std::cout << "Press any key to start" << std::endl;
        cv::waitKey();  // wait for any key
    }

    auto grid_map_info = drawer.GetGridMapInfo();
    PlplotFig fig_sdf(1280, 480, true);
    PlplotFig fig_grad(1280, 480, true);
    PlplotFig::LegendOpt legend_opt_sdf(3, {"SDF", "EDF", "Variance"});
    legend_opt_sdf.SetTextColors({PlplotFig::Color0::Red, PlplotFig::Color0::Yellow, PlplotFig::Color0::Green})
        .SetStyles({PL_LEGEND_LINE, PL_LEGEND_LINE, PL_LEGEND_LINE})
        .SetLineColors(legend_opt_sdf.text_colors)
        .SetLineStyles({1, 1, 1})
        .SetLineWidths({1.0, 1.0, 1.0});
    PlplotFig::LegendOpt legend_opt_grad(4, {"grad_x", "grad_y", "var_grad_x", "var_grad_y"});
    legend_opt_grad.SetTextColors({PlplotFig::Color0::Red, PlplotFig::Color0::Yellow, PlplotFig::Color0::Green, PlplotFig::Color0::Aquamarine})
        .SetStyles({PL_LEGEND_LINE, PL_LEGEND_LINE, PL_LEGEND_LINE, PL_LEGEND_LINE})
        .SetLineColors(legend_opt_grad.text_colors)
        .SetLineStyles({1, 1, 1, 1})
        .SetLineWidths({1.0, 1.0, 1.0, 1.0});

    std::string window_name = test_info->name();
    const double tspan = 500.0 * tic;
    auto draw_curve = [&](PlplotFig &fig, const PlplotFig::Color0 color_idx, const int n, const double *ts, const double *vs) {
        fig.SetCurrentColor(color_idx).SetPenWidth(2).DrawLine(n, ts, vs).SetPenWidth(1);
    };

    // save video
    std::shared_ptr<cv::VideoWriter> video_writer = nullptr;
    std::string video_path = (test_output_dir / "sdf_mapping.avi").string();
    cv::Size frame_size(img.cols + 1280, std::max(img.rows, 960));
    cv::Mat frame(frame_size.height, frame_size.width, CV_8UC3, cv::Scalar(0));
    if (options.save_video) { video_writer = std::make_shared<cv::VideoWriter>(video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0, frame_size); }

    // start the mapping
    const std::string bin_file = test_output_dir / fmt::format("sdf_mapping_2d_{}.bin", type_name<Dtype>());
    double t_ms = 0;
    double traj_t = 0;
    std::vector<double> timestamps_second;
    std::vector<double> sdf_values;
    std::vector<double> edf_values;
    std::vector<double> var_sdf_values;
    std::vector<double> grad_x_values;
    std::vector<double> grad_y_values;
    std::vector<double> var_grad_x_values;
    std::vector<double> var_grad_y_values;
    timestamps_second.reserve(max_update_cnt);
    sdf_values.reserve(max_update_cnt);
    edf_values.reserve(max_update_cnt);
    var_sdf_values.reserve(max_update_cnt);
    grad_x_values.reserve(max_update_cnt);
    grad_y_values.reserve(max_update_cnt);
    var_grad_x_values.reserve(max_update_cnt);
    var_grad_y_values.reserve(max_update_cnt);
    for (long i = 0; i < max_update_cnt; i++) {
        const auto &[rotation, translation] = train_poses[i];
        const VectorX &ranges = train_ranges[i];
        auto t0 = std::chrono::high_resolution_clock::now();
        EXPECT_TRUE(sdf_mapping.Update(rotation, translation, ranges));
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("Update time: {:f} ms.", dt);
        t_ms += dt;
        traj_t += tic;

        if (options.visualize) {
            bool pixel_based = true;
            if (!drawer_connected) {
                drawer.SetQuadtree(surface_mapping->GetTree());
                drawer_connected = true;
            }
            img.setTo(cv::Scalar(128, 128, 128, 255));
            arrowed_lines.clear();
            if (update_occupancy) {
                drawer.DrawLeaves(img);
            } else {
                drawer.DrawTree(img);
            }
            for (auto &[position_px_cv, arrow_end_px]: arrowed_lines) {
                cv::arrowedLine(img, position_px_cv, arrow_end_px, cv::Scalar(0, 0, 255, 255), 1, 8, 0, 0.1);
            }

            // Test SDF Estimation
            Vector2 position = cur_traj.col(i);
            VectorX distance(1);
            Matrix2X gradient(2, 1);
            Matrix3X variances(3, 1);
            Matrix3X covariances(3, 1);
            EXPECT_TRUE(sdf_mapping.Test(position, distance, gradient, variances, covariances));

            // draw sdf
            cv::Point position_px = EigenToOpenCV(drawer.GetGridMapInfo()->MeterToPixelForPoints(cur_traj.col(i)));
            auto radius = static_cast<int>(std::abs(distance[0]) / drawer_setting->resolution);
            cv::Mat circle_layer(img.rows, img.cols, CV_8UC4, cv::Scalar(0));
            cv::Mat circle_mask(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
            cv::circle(circle_mask, position_px, radius, cv::Scalar(255), cv::FILLED);
            cv::circle(circle_layer, position_px, radius, cv::Scalar(0, 255, 0, 25), cv::FILLED);
            cv::add(img * 0.5, circle_layer * 0.5, img, circle_mask);

            // draw sdf gradient
            Eigen::Vector2i gradient_px = drawer.GetPixelCoordsForVectors(gradient);
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
            auto &gps = sdf_mapping.GetUsedGps()[0];
            auto gp1 = gps[0];
            auto gp2 = gps[1];
            if (gp1 != nullptr) {
                DrawGp<Dtype>(img, gp1, drawer, {0, 125, 255, 255}, {125, 255, 0, 255}, {255, 125, 0, 255});
                typename erl::gaussian_process::NoisyInputGaussianProcess<Dtype>::TrainSet &train_set = gp1->edf_gp->GetTrainSet();
                ERL_INFO("GP1 at [{:f}, {:f}] has {} data points.", gp1->position.x(), gp1->position.y(), train_set.num_samples);
            }
            if (gp2 != nullptr) {
                DrawGp<Dtype>(img, gp2, drawer, {125, 125, 255, 255}, {125, 255, 125, 255}, {255, 125, 0, 255});
                typename erl::gaussian_process::NoisyInputGaussianProcess<Dtype>::TrainSet &train_set = gp2->edf_gp->GetTrainSet();
                ERL_INFO("GP2 has {} data points.", train_set.num_samples);
            }

            // draw trajectory
            DrawTrajectoryInplace<Dtype>(img, cur_traj.block(0, 0, 2, i), drawer.GetGridMapInfo(), trajectory_color, 2, pixel_based);

            // draw fps
            cv::putText(img, std::to_string(1000.0 / dt), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0, 255), 2);

            timestamps_second.push_back(traj_t);
            sdf_values.push_back(distance[0]);
            edf_values.push_back(std::abs(distance[0]));
            var_sdf_values.push_back(variances(0, 0));
            grad_x_values.push_back(gradient(0, 0));
            grad_y_values.push_back(gradient(1, 0));
            var_grad_x_values.push_back(variances(1, 0));
            var_grad_y_values.push_back(variances(2, 0));
            const double t_min = traj_t - tspan;
            int n = 0;
            for (; n < static_cast<int>(timestamps_second.size()) && timestamps_second[n] < t_min; ++n) {}  // skip the first n points
            if (n > 0) {
                timestamps_second.erase(timestamps_second.begin(), timestamps_second.begin() + n);
                sdf_values.erase(sdf_values.begin(), sdf_values.begin() + n);
                edf_values.erase(edf_values.begin(), edf_values.begin() + n);
                var_sdf_values.erase(var_sdf_values.begin(), var_sdf_values.begin() + n);
                grad_x_values.erase(grad_x_values.begin(), grad_x_values.begin() + n);
                grad_y_values.erase(grad_y_values.begin(), grad_y_values.begin() + n);
                var_grad_x_values.erase(var_grad_x_values.begin(), var_grad_x_values.begin() + n);
                var_grad_y_values.erase(var_grad_y_values.begin(), var_grad_y_values.begin() + n);
            }
            n = static_cast<int>(timestamps_second.size());

            if (!timestamps_second.empty()) {
                auto minmax = std::minmax_element(sdf_values.begin(), sdf_values.end());
                double fig_sdf_y_min = *minmax.first;
                double fig_sdf_y_max = *minmax.second;

                // render fig_sdf
                minmax = std::minmax_element(edf_values.begin(), edf_values.end());
                fig_sdf_y_min = std::min(fig_sdf_y_min, *minmax.first) - 0.1;
                fig_sdf_y_max = std::max(fig_sdf_y_max, *minmax.second) + 0.1;

                fig_sdf.Clear()
                    .SetMargin(0.15, 0.85, 0.15, 0.9)
                    .SetAxisLimits(traj_t - tspan, traj_t, fig_sdf_y_min, fig_sdf_y_max)
                    .SetCurrentColor(PlplotFig::Color0::White)
                    .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
                    .SetAxisLabelX("time (sec)")
                    .SetCurrentColor(PlplotFig::Color0::Yellow)
                    .SetAxisLabelY("SDF/EDF (meter)");

                draw_curve(fig_sdf, PlplotFig::Color0::Red, n, timestamps_second.data(), sdf_values.data());
                draw_curve(fig_sdf, PlplotFig::Color0::Yellow, n, timestamps_second.data(), edf_values.data());

                minmax = std::minmax_element(var_sdf_values.begin(), var_sdf_values.end());
                fig_sdf.SetCurrentColor(PlplotFig::Color0::White)
                    .SetAxisLimits(traj_t - tspan, traj_t, *minmax.first - 0.01, *minmax.second + 0.01)
                    .DrawAxesBox(
                        PlplotFig::AxisOpt::Off(),
                        PlplotFig::AxisOpt::Off().DrawTopRightEdge().DrawTickMajor().DrawTickMinor().DrawTopRightTickLabels().DrawPerpendicularTickLabels())
                    .SetCurrentColor(PlplotFig::Color0::Green)
                    .SetAxisLabelY("Variance", true);
                draw_curve(fig_sdf, PlplotFig::Color0::Green, n, timestamps_second.data(), var_sdf_values.data());
                fig_sdf.Legend(legend_opt_sdf);

                // render fig_grad
                minmax = std::minmax_element(grad_x_values.begin(), grad_x_values.end());
                double fig_grad_y_min = *minmax.first;
                double fig_grad_y_max = *minmax.second;

                minmax = std::minmax_element(grad_y_values.begin(), grad_y_values.end());
                fig_grad_y_min = std::min(fig_grad_y_min, *minmax.first) - 0.1;
                fig_grad_y_max = std::max(fig_grad_y_max, *minmax.second) + 0.1;

                fig_grad.Clear()
                    .SetMargin(0.15, 0.85, 0.15, 0.9)
                    .SetAxisLimits(traj_t - tspan, traj_t, fig_grad_y_min, fig_grad_y_max)
                    .SetCurrentColor(PlplotFig::Color0::White)
                    .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
                    .SetAxisLabelX("time (sec)")
                    .SetCurrentColor(PlplotFig::Color0::Yellow)
                    .SetAxisLabelY("Gradient (meter)");

                draw_curve(fig_grad, PlplotFig::Color0::Red, n, timestamps_second.data(), grad_x_values.data());
                draw_curve(fig_grad, PlplotFig::Color0::Yellow, n, timestamps_second.data(), grad_y_values.data());

                minmax = std::minmax_element(var_grad_x_values.begin(), var_grad_x_values.end());
                fig_grad_y_min = *minmax.first;
                fig_grad_y_max = *minmax.second;

                minmax = std::minmax_element(var_grad_y_values.begin(), var_grad_y_values.end());
                fig_grad_y_min = std::min(fig_grad_y_min, *minmax.first) - 0.01;
                fig_grad_y_max = std::max(fig_grad_y_max, *minmax.second) + 0.01;

                fig_grad.SetCurrentColor(PlplotFig::Color0::White)
                    .SetAxisLimits(traj_t - tspan, traj_t, fig_grad_y_min, fig_grad_y_max)
                    .DrawAxesBox(
                        PlplotFig::AxisOpt::Off(),
                        PlplotFig::AxisOpt::Off().DrawTopRightEdge().DrawTickMajor().DrawTickMinor().DrawTopRightTickLabels().DrawPerpendicularTickLabels())
                    .SetCurrentColor(PlplotFig::Color0::Green)
                    .SetAxisLabelY("Variance", true);
                draw_curve(fig_grad, PlplotFig::Color0::Green, n, timestamps_second.data(), var_grad_x_values.data());
                draw_curve(fig_grad, PlplotFig::Color0::Aquamarine, n, timestamps_second.data(), var_grad_y_values.data());
                fig_grad.Legend(legend_opt_grad);
            }
            cv::Mat tmp(frame.rows, frame.cols, CV_8UC4, cv::Scalar(0));
            if (img.rows == frame.rows) {
                const int offset = (frame.rows - fig_sdf.Height() * 2) / 2;
                img.copyTo(tmp(cv::Rect(0, 0, img.cols, img.rows)));
                fig_sdf.ToCvMat().copyTo(tmp(cv::Rect(img.cols, offset, fig_sdf.Width(), fig_sdf.Height())));
                fig_grad.ToCvMat().copyTo(tmp(cv::Rect(img.cols, offset + fig_sdf.Height(), fig_grad.Width(), fig_grad.Height())));
            } else {
                const int offset = (frame.rows - img.rows) / 2;
                img.copyTo(tmp(cv::Rect(0, offset, img.cols, img.rows)));
                fig_sdf.ToCvMat().copyTo(tmp(cv::Rect(img.cols, 0, fig_sdf.Width(), fig_sdf.Height())));
                fig_grad.ToCvMat().copyTo(tmp(cv::Rect(img.cols, fig_sdf.Height(), fig_grad.Width(), fig_grad.Height())));
            }
            cv::cvtColor(tmp, frame, cv::COLOR_BGRA2BGR);

            if (options.save_video) { video_writer->write(frame); }

            cv::imshow(window_name, frame);
            cv::waitKey(1);
        }
        double avg_dt = t_ms / static_cast<double>(i + 1);
        double fps = 1000.0 / avg_dt;
        ERL_INFO("Average update time: {:f} ms, Average fps: {:f}", avg_dt, fps);
        std::cout << "=====================================" << std::endl;

        if (options.test_io && (i == 0 || i == max_update_cnt - 1)) {  // test io
            // TODO: test io
            // ASSERT_TRUE(sdf_mapping.Write(bin_file)) << "Failed to write to " << bin_file;
            // SdfMapping sdf_mapping_load(std::make_shared<erl::sdf_mapping::GpSdfMapping2D::Setting>());
            // ASSERT_TRUE(sdf_mapping_load.Read(bin_file)) << "Failed to read from " << bin_file;
            // ASSERT_TRUE(sdf_mapping == sdf_mapping_load) << "Loaded SDF mapping is not equal to the original one.";
        }
    }

    ERL_INFO("Average update time: {:f} ms.", t_ms / static_cast<double>(max_update_cnt));
    if (options.save_video) {
        video_writer->release();
        ERL_INFO("Saved surface mapping video to {}.", video_path.c_str());
    }

    // Test SDF Estimation
    constexpr bool c_stride = true;
    Matrix2X positions_in = grid_map_info->GenerateMeterCoordinates(c_stride);
    VectorX distances_out(positions_in.cols());
    Matrix2X gradients_out(2, positions_in.cols());
    Matrix3X variances_out(3, positions_in.cols());
    Matrix3X covariances_out;
    auto t0 = std::chrono::high_resolution_clock::now();
    bool success = sdf_mapping.Test(positions_in, distances_out, gradients_out, variances_out, covariances_out);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double t_per_point = dt / static_cast<double>(positions_in.cols()) * 1000;  // us
    ERL_INFO("Test time: {:f} ms for {} points, {:f} us per point.", dt, positions_in.cols(), t_per_point);

    EXPECT_TRUE(success) << "Failed to test SDF estimation at the end.";
    ERL_ASSERT(surface_mapping->GetTree()->WriteBinary("tree.bt"));
    ERL_ASSERT(surface_mapping->GetTree()->Write("tree.ot"));

    if (success && options.visualize) {
        MatrixX sdf_out_mat = distances_out.reshaped(grid_map_info->Height(), grid_map_info->Width());
        Dtype min_distance = distances_out.minCoeff();
        Dtype max_distance = distances_out.maxCoeff();
        ERL_INFO("min distance: {:f}, max distance: {:f}.", min_distance, max_distance);

        img.setTo(cv::Scalar(128, 128, 128, 255));
        drawer_setting->border_color = drawer_setting->occupied_color;
        arrowed_lines.clear();
        if (update_occupancy) {
            drawer.DrawLeaves(img);
        } else {
            drawer.DrawTree(img);
        }

        max_distance = sdf_out_mat.maxCoeff();
        MatrixX8U distances_out_mat_normalized = ((sdf_out_mat.array() - min_distance) / (max_distance - min_distance) * 255).template cast<uint8_t>();
        cv::Mat src, dst;
        cv::eigen2cv(distances_out_mat_normalized, src);
        cv::flip(src, src, 0);  // flip along y axis
        cv::applyColorMap(src, dst, cv::COLORMAP_JET);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2BGRA);
        cv::addWeighted(dst, 0.5, img, 0.5, 0.0, dst);
        cv::imshow(window_name + ": sdf", dst);

        MatrixX8U sdf_sign_mat(sdf_out_mat.rows(), sdf_out_mat.cols());
        uint8_t *sdf_sign_mat_ptr = sdf_sign_mat.data();
        const Dtype *sdf_out_mat_ptr = sdf_out_mat.data();
        for (int i = 0; i < sdf_out_mat.size(); ++i) { sdf_sign_mat_ptr[i] = sdf_out_mat_ptr[i] >= 0 ? 255 : 0; }
        cv::eigen2cv(sdf_sign_mat, src);
        cv::flip(src, src, 0);  // flip along y axis
        cv::imshow(window_name + ": sdf sign", src);

        MatrixX sdf_variances_mat = variances_out.row(0).reshaped(grid_map_info->Height(), grid_map_info->Width());
        Dtype min_variance = sdf_variances_mat.minCoeff();
        Dtype max_variance = sdf_variances_mat.maxCoeff();
        MatrixX8U variances_out_mat_normalized = ((sdf_variances_mat.array() - min_variance) / (max_variance - min_variance) * 255).template cast<uint8_t>();
        cv::eigen2cv(variances_out_mat_normalized, src);
        cv::flip(src, src, 0);  // flip along y axis
        cv::applyColorMap(src, dst, cv::COLORMAP_JET);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2BGRA);
        cv::addWeighted(dst, 0.5, img, 0.5, 0.0, dst);
        cv::imshow(window_name + ": sdf variances", dst);
        cv::waitKey(1);
    }

    if (options.visualize && options.hold) {
        std::cout << "Press any key to exit." << std::endl;
        cv::waitKey(0);
    } else {
        constexpr double wait_time = 10.0;
        cv::waitKey(wait_time * 1000);  // wait for 10 seconds
    }

    if (options.interactive) {

        if (!drawer_connected) { drawer.SetQuadtree(surface_mapping->GetTree()); }
        img.setTo(cv::Scalar(128, 128, 128, 255));
        arrowed_lines.clear();
        if (update_occupancy) {
            drawer.DrawLeaves(img);
        } else {
            drawer.DrawTree(img);
        }
        for (auto &[position_px_cv, arrow_end_px]: arrowed_lines) {
            cv::arrowedLine(img, position_px_cv, arrow_end_px, cv::Scalar(0, 0, 255, 255), 1, 8, 0, 0.1);
        }

        OpenCvUserData<Dtype, QuadtreeDrawer> data;
        data.window_name = window_name;
        data.img = img;
        data.drawer = &drawer;
        data.sdf_mapping = &sdf_mapping;

        cv::imshow(window_name, img);
        cv::setMouseCallback(window_name, OpenCvMouseCallback<Dtype, QuadtreeDrawer>, &data);
        while (cv::waitKey(0) != 27) {}  // wait for ESC key
    }
}

TEST(GpSdfMapping, 2Dd) { TestImpl2D<double>(); }

TEST(GpSdfMapping, 2Df) { TestImpl2D<float>(); }

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
