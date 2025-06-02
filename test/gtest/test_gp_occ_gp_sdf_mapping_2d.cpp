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
const std::filesystem::path kDataDir = kProjectRootDir / "data";
const std::filesystem::path kConfigDir = kProjectRootDir / "config";
int g_argc = 0;
char **g_argv = nullptr;

template<typename Dtype, typename Drawer, typename SurfaceMapping>
void
DrawSurfaceData(
    cv::Mat &img,
    Drawer &drawer,
    SurfaceMapping &surface_mapping,
    const Dtype surf_normal_scale,
    const bool draw_normals) {
    for (auto it = surface_mapping.BeginSurfaceData(), end = surface_mapping.EndSurfaceData();
         it != end;
         ++it) {
        Eigen::Vector2i position_px = drawer.GetPixelCoordsForPositions(it->position, true);
        cv::Point position_px_cv(position_px[0], position_px[1]);
        cv::circle(img, position_px_cv, 2, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
        if (!draw_normals) { continue; }
        Eigen::Vector2i normal_px = drawer.GetPixelCoordsForVectors(it->normal * surf_normal_scale);
        const cv::Point arrow_end_px(position_px[0] + normal_px[0], position_px[1] + normal_px[1]);
        cv::arrowedLine(
            img,
            position_px_cv,
            arrow_end_px,
            cv::Scalar(0, 0, 255, 255),
            1,
            cv::LINE_8,
            0,
            0.1);
    }
}

template<typename Dtype>
void
DrawSdf(cv::Mat &img, const int x, const int y, Dtype sdf, Dtype resolution) {
    const auto radius = static_cast<int>(std::abs(sdf) / resolution);
    cv::Mat circle_layer(img.rows, img.cols, CV_8UC4, cv::Scalar(0));
    cv::Mat circle_mask(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
    cv::circle(circle_mask, cv::Point2i(x, y), radius, cv::Scalar(255), cv::FILLED);
    cv::circle(circle_layer, cv::Point2i(x, y), radius, cv::Scalar(0, 255, 0, 25), cv::FILLED);
    cv::add(img * 0.5, circle_layer * 0.5, img, circle_mask);
}

template<typename Dtype>
void
DrawSdfVariance(
    cv::Mat &img,
    const int x,
    const int y,
    const Dtype sdf,
    const Dtype sdf_variance,
    const Dtype resolution) {
    const auto radius = static_cast<int>((std::sqrt(sdf_variance) + std::abs(sdf)) / resolution);
    cv::circle(img, cv::Point2i(x, y), radius, cv::Scalar(0, 0, 255, 25), 1);
}

template<typename Dtype, typename Drawer>
void
DrawSdfGradient(
    cv::Mat &img,
    Drawer *drawer,
    const int x,
    const int y,
    const Eigen::Vector2<Dtype> &gradient) {
    const Eigen::VectorXi grad_pixel = drawer->GetPixelCoordsForVectors(gradient);
    cv::arrowedLine(
        img,
        cv::Point(x, y),
        cv::Point(x + grad_pixel[0], y + grad_pixel[1]),
        cv::Scalar(255, 0, 0, 255),
        2);
}

template<typename Dtype, typename Gp, typename Drawer>
void
DrawGp(
    cv::Mat &img,
    const std::shared_ptr<Gp> &gp,
    const Drawer &drawer,
    const cv::Scalar &data_color = {0, 255, 125, 255},
    const cv::Scalar &pos_color = {255, 125, 0, 255},
    const cv::Scalar &rect_color = {125, 255, 0, 255},
    const bool draw_data = true,
    const bool draw_pos = true,
    const bool draw_rect = true) {
    if (gp == nullptr) { return; }

    if (draw_pos) {
        Eigen::Vector2i gp_position_px = drawer.GetPixelCoordsForPositions(gp->position, true);
        cv::drawMarker(
            img,
            cv::Point(gp_position_px[0], gp_position_px[1]),
            pos_color,
            cv::MARKER_STAR,
            10,
            1);
    }

    if (draw_rect) {
        const Eigen::Vector2<Dtype> gp_area_min = gp->position.array() - gp->half_size;
        const Eigen::Vector2<Dtype> gp_area_max = gp->position.array() + gp->half_size;
        Eigen::Vector2i gp_area_min_px = drawer.GetPixelCoordsForPositions(gp_area_min, true);
        Eigen::Vector2i gp_area_max_px = drawer.GetPixelCoordsForPositions(gp_area_max, true);
        cv::rectangle(
            img,
            cv::Point(gp_area_min_px[0], gp_area_min_px[1]),
            cv::Point(gp_area_max_px[0], gp_area_max_px[1]),
            rect_color,
            2);
    }

    if (!draw_data) { return; }

    typename erl::gaussian_process::NoisyInputGaussianProcess<Dtype>::TrainSet &train_set =
        gp->edf_gp->GetTrainSet();
    Eigen::Matrix2X<Dtype> used_surface_points = train_set.x.block(0, 0, 2, train_set.num_samples);
    Eigen::Matrix2Xi used_surface_points_px =
        drawer.GetPixelCoordsForPositions(used_surface_points, true);
    for (long j = 0; j < used_surface_points.cols(); j++) {
        cv::circle(
            img,
            cv::Point(used_surface_points_px(0, j), used_surface_points_px(1, j)),
            3,
            data_color,
            -1);
    }
}

template<typename Dtype, typename Drawer>
struct OpenCvUserData {
    using SurfaceMapping = erl::sdf_mapping::GpOccSurfaceMapping<Dtype, 2>;
    using SdfMapping = erl::sdf_mapping::GpSdfMapping<Dtype, 2, SurfaceMapping>;

    std::string window_name;
    Drawer *drawer = nullptr;
    SurfaceMapping *surface_mapping = nullptr;
    SdfMapping *sdf_mapping = nullptr;
    cv::Mat img;
};

template<typename Dtype, typename Drawer>
void
OpenCvMouseCallback(const int event, const int x, const int y, int /*flags*/, void *userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        auto *data = static_cast<OpenCvUserData<Dtype, Drawer> *>(userdata);
        Eigen::Vector2<Dtype> position =
            data->drawer->GetMeterCoordsForPositions(Eigen::Vector2i(x, y), false);
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

            auto gp = data->sdf_mapping->GetUsedGps()[0][0];
            ERL_INFO("pick {}", reinterpret_cast<uint64_t>(gp.get()));
            ERL_INFO("position: {}, half_size: {}", gp->position.transpose(), gp->half_size);
            erl::geometry::Aabb<Dtype, 2> aabb(gp->position, gp->half_size);
            std::vector<std::pair<Dtype, std::size_t> > distances_indices;
            data->surface_mapping->CollectSurfaceDataInAabb(aabb, distances_indices);
            ERL_INFO("Found {} surface data points in the area.", distances_indices.size());

            cv::Mat img = data->img.clone();

            // draw sdf
            const Dtype resolution = data->drawer->GetGridMapInfo()->Resolution(0);
            DrawSdf(img, x, y, distance[0], resolution);
            // draw sdf variance
            DrawSdfVariance(img, x, y, distance[0], variances(0, 0), resolution);
            // draw sdf gradient
            DrawSdfGradient<Dtype, Drawer>(img, data->drawer, x, y, gradient.col(0));

            auto &[gp1, gp2] = data->sdf_mapping->GetUsedGps()[0];
            DrawGp<Dtype>(img, gp1, *data->drawer, {0, 125, 255, 255});
            DrawGp<Dtype>(img, gp2, *data->drawer, {125, 125, 255, 255});

            cv::putText(
                img,
                fmt::format(
                    "SDF: {:.2f}, Var: {:.6f} | grad: [{:.6f}, {:.6f}], Var: [{:.6f}, {:.6f}], "
                    "Std(theta): {:.6f}",
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
    using Quadtree = typename SurfaceMapping::Tree;
    using QuadtreeDrawer = erl::geometry::OccupancyQuadtreeDrawer<Quadtree>;
    using Lidar2D = erl::geometry::Lidar2D;

    using Matrix2X = Eigen::Matrix2X<Dtype>;
    using Matrix3X = Eigen::Matrix3X<Dtype>;
    using Vector2 = Eigen::Vector2<Dtype>;
    using VectorX = Eigen::VectorX<Dtype>;

#pragma region options

    struct Options {
        std::string gazebo_dir = kDataDir / "gazebo";
        std::string house_expo_map_file = kDataDir / "house_expo_room_1451.json";
        std::string house_expo_traj_file = kDataDir / "house_expo_room_1451.csv";
        std::string ucsd_fah_2d_file = kDataDir / "ucsd_fah_2d.dat";
        std::string surface_mapping_config_file =
            kConfigDir / "template" / fmt::format("gp_occ_mapping_2d_{}.yaml", type_name<Dtype>());
        std::string sdf_mapping_config_file =
            kConfigDir / "template" / fmt::format("sdf_mapping_2d_{}.yaml", type_name<Dtype>());
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

#pragma endregion

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
                po::value<std::string>(&options.gazebo_dir)->default_value(options.gazebo_dir)->value_name("folder"),
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

    ASSERT_TRUE(
        options.use_gazebo_room_2d || options.use_house_expo_lidar_2d || options.use_ucsd_fah_2d)
        << "Please specify one of --use-gazebo-room-2d, --use-house-expo-lidar-2d, "
           "--use-ucsd-fah-2d.";
    if (options.use_gazebo_room_2d) {
        ASSERT_TRUE(std::filesystem::exists(options.gazebo_dir))
            << "Gazebo data folder " << options.gazebo_dir << " does not exist.";
    }
    if (options.use_house_expo_lidar_2d) {
        ASSERT_TRUE(std::filesystem::exists(options.house_expo_map_file))
            << "HouseExpo map file " << options.house_expo_map_file << " does not exist.";
        ASSERT_TRUE(std::filesystem::exists(options.house_expo_traj_file))
            << "HouseExpo trajectory file " << options.house_expo_traj_file << " does not exist.";
    }
    if (options.use_ucsd_fah_2d) {
        ASSERT_TRUE(std::filesystem::exists(options.ucsd_fah_2d_file))
            << "ROS bag dat file " << options.ucsd_fah_2d_file << " does not exist.";
    }

    // load the scene
    long max_update_cnt;
    std::vector<VectorX> train_angles;
    std::vector<Eigen::VectorXd> train_ranges;
    std::vector<std::pair<Eigen::Matrix2d, Eigen::Vector2d> > train_poses;
    Vector2 map_min(0, 0);
    Vector2 map_max(0, 0);
    Vector2 map_resolution(options.map_resolution, options.map_resolution);
    Eigen::Vector2i map_padding(10, 10);
    Matrix2X cur_traj;
    double tic = 0.2;

    using namespace erl::geometry;

    if (options.use_gazebo_room_2d) {
        auto train_data_loader = GazeboRoom2D::TrainDataLoader(options.gazebo_dir);
        max_update_cnt =
            static_cast<long>(train_data_loader.size() - options.init_frame) / options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        auto bar_setting = std::make_shared<ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = ProgressBar::Open(bar_setting, std::cout);
        int cnt = 0;
        for (int i = options.init_frame; i < static_cast<int>(train_data_loader.size());
             i += options.stride, ++cnt) {
            auto &df = train_data_loader[i];
            train_angles.push_back(df.angles.cast<Dtype>());
            train_ranges.push_back(df.ranges);
            train_poses.emplace_back(df.rotation, df.translation);
            cur_traj.col(cnt) << df.translation.cast<Dtype>();
            bar->Update();
        }
        map_min = GazeboRoom2D::kMapMin.cast<Dtype>();
        map_max = GazeboRoom2D::kMapMax.cast<Dtype>();
        map_padding.setZero();

        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else if (options.use_house_expo_lidar_2d) {
        HouseExpoMap house_expo_map(options.house_expo_map_file, 0.2);
        map_min = house_expo_map.GetMeterSpace()
                      ->GetSurface()
                      ->vertices.rowwise()
                      .minCoeff()
                      .cast<Dtype>();
        map_max = house_expo_map.GetMeterSpace()
                      ->GetSurface()
                      ->vertices.rowwise()
                      .maxCoeff()
                      .cast<Dtype>();
        auto lidar_setting = std::make_shared<Lidar2D::Setting>();
        lidar_setting->num_lines = 720;
        Lidar2D lidar(lidar_setting, house_expo_map.GetMeterSpace());
        auto trajectory = LoadAndCastCsvFile<double>(
            options.house_expo_traj_file.c_str(),
            [](const std::string &str) -> double { return std::stod(str); });
        max_update_cnt =
            static_cast<long>(trajectory.size() - options.init_frame) / options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        auto bar_setting = std::make_shared<ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = ProgressBar::Open(bar_setting, std::cout);
        int cnt = 0;
        for (std::size_t i = options.init_frame; i < trajectory.size();
             i += options.stride, cnt++) {
            bool scan_in_parallel = true;
            std::vector<double> &waypoint = trajectory[i];
            cur_traj.col(cnt) << static_cast<Dtype>(waypoint[0]), static_cast<Dtype>(waypoint[1]);

            Eigen::Matrix2d rotation = Eigen::Rotation2Dd(waypoint[2]).toRotationMatrix();
            Eigen::Vector2d translation(waypoint[0], waypoint[1]);
            Eigen::VectorXd lidar_ranges = lidar.Scan(rotation, translation, scan_in_parallel);
            lidar_ranges += GenerateGaussianNoise<double>(lidar_ranges.size(), 0.0, 0.01);
            train_angles.push_back(lidar.GetAngles().cast<Dtype>());
            train_ranges.push_back(lidar_ranges);
            train_poses.emplace_back(rotation, translation);
            bar->Update();
        }
        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else if (options.use_ucsd_fah_2d) {
        UcsdFah2D ucsd_fah(options.ucsd_fah_2d_file);
        tic = ucsd_fah.GetTimeStep();
        map_min = UcsdFah2D::kMapMin.cast<Dtype>();
        map_max = UcsdFah2D::kMapMax.cast<Dtype>();
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
            auto
                [sequence_number,
                 timestamp,
                 header_timestamp,
                 rotation,
                 translation,
                 angles,
                 ranges] = ucsd_fah[i];
            cur_traj.col(cnt) << translation.cast<Dtype>();
            train_angles.emplace_back(angles.cast<Dtype>());
            train_ranges.emplace_back(ranges);
            train_poses.emplace_back(rotation, translation);
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
    auto surface_mapping = std::make_shared<SurfaceMapping>(surface_mapping_setting);

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
    legend_opt_sdf
        .SetTextColors({PlplotFig::Color0::Red, PlplotFig::Color0::Blue, PlplotFig::Color0::Green})
        .SetStyles({PL_LEGEND_LINE, PL_LEGEND_LINE, PL_LEGEND_LINE})
        .SetLineColors(legend_opt_sdf.text_colors)
        .SetLineStyles({1, 1, 1})
        .SetLineWidths({1.0, 1.0, 1.0})
        .SetPosition(PL_POSITION_LEFT | PL_POSITION_TOP)
        .SetBoxStyle(PL_LEGEND_BOUNDING_BOX | PL_LEGEND_BACKGROUND)
        .SetLegendBoxLineColor0(PlplotFig::Color0::Black)
        .SetBgColor0(PlplotFig::Color0::Gray)
        .SetTextScale(1.1);
    PlplotFig::LegendOpt legend_opt_grad(4, {"grad_x", "grad_y", "var_grad_x", "var_grad_y"});
    legend_opt_grad
        .SetTextColors(
            {PlplotFig::Color0::Red,
             PlplotFig::Color0::Blue,
             PlplotFig::Color0::Green,
             PlplotFig::Color0::Brown})
        .SetStyles({PL_LEGEND_LINE, PL_LEGEND_LINE, PL_LEGEND_LINE, PL_LEGEND_LINE})
        .SetLineColors(legend_opt_grad.text_colors)
        .SetLineStyles({1, 1, 1, 1})
        .SetLineWidths({1.0, 1.0, 1.0, 1.0})
        .SetPosition(PL_POSITION_LEFT | PL_POSITION_TOP)
        .SetBoxStyle(PL_LEGEND_BOUNDING_BOX | PL_LEGEND_BACKGROUND)
        .SetLegendBoxLineColor0(PlplotFig::Color0::Black)
        .SetBgColor0(PlplotFig::Color0::Gray);

    std::string window_name = test_info->name();
    const double tspan = 500.0 * tic;
    auto draw_curve = [&](PlplotFig &fig,
                          const PlplotFig::Color0 color_idx,
                          const int n,
                          const double *ts,
                          const double *vs) {
        fig.SetCurrentColor(color_idx).SetPenWidth(2).DrawLine(n, ts, vs).SetPenWidth(1);
    };

    // save video
    std::shared_ptr<cv::VideoWriter> video_writer = nullptr;
    std::string video_path = (test_output_dir / "sdf_mapping.avi").string();
    cv::Size frame_size(img.cols + 1280, std::max(img.rows, 960));
    cv::Mat frame(frame_size.height, frame_size.width, CV_8UC3, cv::Scalar(0));
    if (options.save_video) {
        video_writer = std::make_shared<cv::VideoWriter>(
            video_path,
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            30.0,
            frame_size);
    }

    // start the mapping
    auto bin_file = test_output_dir / fmt::format("sdf_mapping_2d_{}.bin", type_name<Dtype>());
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
        const Eigen::VectorXd &ranges = train_ranges[i];
        auto t0 = std::chrono::high_resolution_clock::now();
        EXPECT_TRUE(sdf_mapping.Update(rotation, translation, ranges, false, false));
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
            if (update_occupancy) {
                drawer.DrawLeaves(img);
            } else {
                drawer.DrawTree(img);
            }
            DrawSurfaceData(img, drawer, *surface_mapping, options.surf_normal_scale, true);

            // Test SDF Estimation
            Vector2 position = cur_traj.col(i);
            VectorX distance(1);
            Matrix2X gradient(2, 1);
            Matrix3X variances(3, 1);
            Matrix3X covariances(3, 1);
            EXPECT_TRUE(sdf_mapping.Test(position, distance, gradient, variances, covariances));

            // Visualize the results
            cv::Point position_px =
                EigenToOpenCV(drawer.GetGridMapInfo()->MeterToPixelForPoints(position));
            DrawSdf(img, position_px.x, position_px.y, distance[0], drawer_setting->resolution);
            DrawSdfVariance(
                img,
                position_px.x,
                position_px.y,
                distance[0],
                variances(0, 0),
                drawer_setting->resolution);
            DrawSdfGradient(
                img,
                &drawer,
                position_px.x,
                position_px.y,
                Eigen::Vector2<Dtype>(gradient.col(0)));

            // draw used surface points
            auto &gps = sdf_mapping.GetUsedGps()[0];
            auto gp1 = gps[0];
            auto gp2 = gps[1];
            if (gp1 != nullptr) {
                DrawGp<Dtype>(img, gp1, drawer, {0, 125, 255, 255}, {255, 125, 0, 255});
                typename erl::gaussian_process::NoisyInputGaussianProcess<Dtype>::TrainSet
                    &train_set = gp1->edf_gp->GetTrainSet();
                ERL_INFO(
                    "GP1 at [{:f}, {:f}] has {} data points.",
                    gp1->position.x(),
                    gp1->position.y(),
                    train_set.num_samples);
            }
            if (gp2 != nullptr) {
                DrawGp<Dtype>(img, gp2, drawer, {125, 125, 255, 255}, {125, 255, 125, 255});
                typename erl::gaussian_process::NoisyInputGaussianProcess<Dtype>::TrainSet
                    &train_set = gp2->edf_gp->GetTrainSet();
                ERL_INFO("GP2 has {} data points.", train_set.num_samples);
            }

            // draw trajectory
            DrawTrajectoryInplace<Dtype>(
                img,
                cur_traj.block(0, 0, 2, i),
                drawer.GetGridMapInfo(),
                trajectory_color,
                2,
                pixel_based);

            // draw fps
            cv::putText(
                img,
                fmt::format("{:.2f} fps", 1000.0 / dt),
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 255, 0, 255),
                2);

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
            for (; n < static_cast<int>(timestamps_second.size()) && timestamps_second[n] < t_min;
                 ++n) {}  // skip the first n points
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
                    .SetMargin(0.15, 0.85, 0.15, 0.85)
                    .SetAxisLimits(traj_t - tspan, traj_t, fig_sdf_y_min, fig_sdf_y_max)
                    .SetCurrentColor(PlplotFig::Color0::Black)
                    .DrawAxesBox(
                        PlplotFig::AxisOpt().DrawTopRightEdge(),
                        PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
                    .SetAxisLabelX("time (sec)")
                    .SetCurrentColor(PlplotFig::Color0::Black)
                    .SetAxisLabelY("SDF/EDF (meter)");

                draw_curve(
                    fig_sdf,
                    PlplotFig::Color0::Red,
                    n,
                    timestamps_second.data(),
                    sdf_values.data());
                draw_curve(
                    fig_sdf,
                    PlplotFig::Color0::Blue,
                    n,
                    timestamps_second.data(),
                    edf_values.data());

                minmax = std::minmax_element(var_sdf_values.begin(), var_sdf_values.end());
                fig_sdf.SetCurrentColor(PlplotFig::Color0::Black)
                    .SetAxisLimits(
                        traj_t - tspan,
                        traj_t,
                        *minmax.first - 0.001,
                        *minmax.second + 0.001)
                    .DrawAxesBox(
                        PlplotFig::AxisOpt::Off(),
                        PlplotFig::AxisOpt::Off()
                            .DrawTopRightEdge()
                            .DrawTickMajor()
                            .DrawTickMinor()
                            .DrawTopRightTickLabels()
                            .DrawPerpendicularTickLabels())
                    .SetCurrentColor(PlplotFig::Color0::Black)
                    .SetAxisLabelY("Variance", true)
                    .SetTitle(
                        fmt::format("sdf: {:.2f}, var_sdf: {:.2e}", distance[0], variances(0, 0))
                            .c_str());
                draw_curve(
                    fig_sdf,
                    PlplotFig::Color0::Green,
                    n,
                    timestamps_second.data(),
                    var_sdf_values.data());
                fig_sdf.SetFontSize(0.0, 0.8).Legend(legend_opt_sdf).SetFontSize();

                // render fig_grad
                minmax = std::minmax_element(grad_x_values.begin(), grad_x_values.end());
                double fig_grad_y_min = *minmax.first;
                double fig_grad_y_max = *minmax.second;

                minmax = std::minmax_element(grad_y_values.begin(), grad_y_values.end());
                fig_grad_y_min = std::min(fig_grad_y_min, *minmax.first) - 0.1;
                fig_grad_y_max = std::max(fig_grad_y_max, *minmax.second) + 0.1;

                fig_grad.Clear()
                    .SetMargin(0.15, 0.85, 0.15, 0.85)
                    .SetAxisLimits(traj_t - tspan, traj_t, fig_grad_y_min, fig_grad_y_max)
                    .SetCurrentColor(PlplotFig::Color0::Black)
                    .DrawAxesBox(
                        PlplotFig::AxisOpt().DrawTopRightEdge(),
                        PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
                    .SetAxisLabelX("time (sec)")
                    .SetCurrentColor(PlplotFig::Color0::Black)
                    .SetAxisLabelY("Gradient (meter)");

                draw_curve(
                    fig_grad,
                    PlplotFig::Color0::Red,
                    n,
                    timestamps_second.data(),
                    grad_x_values.data());
                draw_curve(
                    fig_grad,
                    PlplotFig::Color0::Blue,
                    n,
                    timestamps_second.data(),
                    grad_y_values.data());

                minmax = std::minmax_element(var_grad_x_values.begin(), var_grad_x_values.end());
                fig_grad_y_min = *minmax.first;
                fig_grad_y_max = *minmax.second;

                minmax = std::minmax_element(var_grad_y_values.begin(), var_grad_y_values.end());
                fig_grad_y_min = std::min(fig_grad_y_min, *minmax.first) - 0.001;
                fig_grad_y_max = std::max(fig_grad_y_max, *minmax.second) + 0.001;

                fig_grad.SetCurrentColor(PlplotFig::Color0::Black)
                    .SetAxisLimits(traj_t - tspan, traj_t, fig_grad_y_min, fig_grad_y_max)
                    .DrawAxesBox(
                        PlplotFig::AxisOpt::Off(),
                        PlplotFig::AxisOpt::Off()
                            .DrawTopRightEdge()
                            .DrawTickMajor()
                            .DrawTickMinor()
                            .DrawTopRightTickLabels()
                            .DrawPerpendicularTickLabels())
                    .SetCurrentColor(PlplotFig::Color0::Black)
                    .SetAxisLabelY("Variance", true)
                    .SetTitle(fmt::format(  //
                                  "grad: [{:.2f}, {:.2f}], var_grad: [{:.2e}, {:.2e}]",
                                  gradient(0, 0),
                                  gradient(1, 0),
                                  variances(1, 0),
                                  variances(2, 0))
                                  .c_str());
                draw_curve(
                    fig_grad,
                    PlplotFig::Color0::Green,
                    n,
                    timestamps_second.data(),
                    var_grad_x_values.data());
                draw_curve(
                    fig_grad,
                    PlplotFig::Color0::Brown,
                    n,
                    timestamps_second.data(),
                    var_grad_y_values.data());
                fig_grad.SetFontSize(0.0, 0.8).Legend(legend_opt_grad).SetFontSize();
            }
            cv::Mat tmp(frame.rows, frame.cols, CV_8UC4, cv::Scalar(255, 255, 255, 255));
            if (img.rows == frame.rows) {
                const int offset = (frame.rows - fig_sdf.Height() * 2) / 2;
                img.copyTo(tmp(cv::Rect(0, 0, img.cols, img.rows)));
                fig_sdf.ToCvMat().copyTo(
                    tmp(cv::Rect(img.cols, offset, fig_sdf.Width(), fig_sdf.Height())));
                fig_grad.ToCvMat().copyTo(tmp(cv::Rect(
                    img.cols,
                    offset + fig_sdf.Height(),
                    fig_grad.Width(),
                    fig_grad.Height())));
            } else {
                const int offset = (frame.rows - img.rows) / 2;
                img.copyTo(tmp(cv::Rect(0, offset, img.cols, img.rows)));
                fig_sdf.ToCvMat().copyTo(
                    tmp(cv::Rect(img.cols, 0, fig_sdf.Width(), fig_sdf.Height())));
                fig_grad.ToCvMat().copyTo(
                    tmp(cv::Rect(img.cols, fig_sdf.Height(), fig_grad.Width(), fig_grad.Height())));
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

        if (options.test_io && (i == 0 || i == max_update_cnt - 1)) {
            ASSERT_TRUE(Serialization<SdfMapping>::Write(bin_file, &sdf_mapping));
            auto surface_mapping_read = std::make_shared<SurfaceMapping>(
                std::make_shared<typename SurfaceMapping::Setting>());
            SdfMapping sdf_mapping_read(
                std::make_shared<typename SdfMapping::Setting>(),
                surface_mapping_read);
            ASSERT_TRUE(Serialization<SdfMapping>::Read(bin_file, &sdf_mapping_read));
            ASSERT_TRUE(sdf_mapping == sdf_mapping_read)
                << "Loaded SDF mapping is not equal to the original one.";
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
    bool success =
        sdf_mapping
            .Test(positions_in, distances_out, gradients_out, variances_out, covariances_out);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double t_per_point = dt / static_cast<double>(positions_in.cols()) * 1000;  // us
    ERL_INFO(
        "Test time: {:f} ms for {} points, {:f} us per point.",
        dt,
        positions_in.cols(),
        t_per_point);

    EXPECT_TRUE(success) << "Failed to test SDF estimation at the end.";
    ASSERT_TRUE(Serialization<typename SurfaceMapping::Tree>::Write(
        test_output_dir / "tree.bt",
        [&](std::ostream &s) -> bool { return surface_mapping->GetTree()->WriteBinary(s); }));
    ASSERT_TRUE(Serialization<typename SurfaceMapping::Tree>::Write(
        test_output_dir / "tree.ot",
        surface_mapping->GetTree()));

    if (success && options.visualize) {
        Dtype min_distance = distances_out.minCoeff();
        Dtype max_distance = distances_out.maxCoeff();
        ERL_INFO("min distance: {:f}, max distance: {:f}.", min_distance, max_distance);

        img.setTo(cv::Scalar(128, 128, 128, 255));
        // drawer_setting->border_color = drawer_setting->occupied_color;
        if (update_occupancy) {
            drawer.DrawLeaves(img);
        } else {
            drawer.DrawTree(img);
        }

        cv::Mat img_gp = img.clone();
        int gp_cnt = 0;
        for (const auto &[key, sdf_gp]: sdf_mapping.GetGpMap()) {
            if (sdf_gp == nullptr) { continue; }
            ++gp_cnt;
            DrawGp<Dtype>(
                img_gp,
                sdf_gp,
                drawer,
                {0, 125, 255, 255},
                {255, 125, 0, 255},
                {0, 0, 0, 255},
                false,
                true,
                gp_cnt < 100);  // draw only the first 100 GPs' bounding boxes
        }
        cv::imshow(window_name + ": GPs", img_gp);
        cv::imwrite(test_output_dir / "gps.png", img_gp);

        cv::Mat img_sdf(
            grid_map_info->Width(),
            grid_map_info->Height(),
            sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1,
            distances_out.data());
        img_sdf = img_sdf.t();

        cv::normalize(img_sdf, img_sdf, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::flip(img_sdf, img_sdf, 0);
        cv::applyColorMap(img_sdf, img_sdf, cv::COLORMAP_JET);
        cv::cvtColor(img_sdf, img_sdf, cv::COLOR_BGR2BGRA);
        cv::addWeighted(img_sdf, 0.5, img, 0.5, 0.0, img_sdf);
        cv::imshow(window_name + ": sdf", img_sdf);
        cv::imwrite(test_output_dir / "sdf.png", img_sdf);

        // convert to binary image: 0 for negative, 255 for positive

        Eigen::VectorXi sdf_sign = (distances_out.array() >= 0).template cast<int>();
        cv::Mat img_sign(
            grid_map_info->Width(),
            grid_map_info->Height(),
            CV_32SC1,
            sdf_sign.data());
        img_sign = img_sign.t();
        cv::normalize(img_sign, img_sign, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::flip(img_sign, img_sign, 0);  // flip along y axis
        cv::imshow(window_name + ": sdf sign", img_sign);
        cv::imwrite(test_output_dir / "sdf_sign.png", img_sign);

        VectorX sdf_variances = variances_out.row(0).transpose();
        cv::Mat img_sdf_variance(
            grid_map_info->Width(),
            grid_map_info->Height(),
            sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1,
            sdf_variances.data());
        img_sdf_variance = img_sdf_variance.t();
        cv::normalize(img_sdf_variance, img_sdf_variance, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::flip(img_sdf_variance, img_sdf_variance, 0);
        cv::applyColorMap(img_sdf_variance, img_sdf_variance, cv::COLORMAP_JET);
        cv::cvtColor(img_sdf_variance, img_sdf_variance, cv::COLOR_BGR2BGRA);
        cv::addWeighted(img_sdf_variance, 0.5, img, 0.5, 0.0, img_sdf_variance);
        cv::imshow(window_name + ": sdf variance", img_sdf_variance);
        cv::imwrite(test_output_dir / "sdf_variance.png", img_sdf_variance);
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
        if (update_occupancy) {
            drawer.DrawLeaves(img);
        } else {
            drawer.DrawTree(img);
        }
        DrawSurfaceData(img, drawer, *surface_mapping, options.surf_normal_scale, false);

        OpenCvUserData<Dtype, QuadtreeDrawer> data;
        data.window_name = window_name + ": interactive";
        data.img = img;
        data.drawer = &drawer;
        data.surface_mapping = surface_mapping.get();
        data.sdf_mapping = &sdf_mapping;

        cv::imshow(data.window_name, img);
        cv::setMouseCallback(data.window_name, OpenCvMouseCallback<Dtype, QuadtreeDrawer>, &data);
        while (cv::waitKey(0) != 27) {}  // wait for the ESC key
    }
}

TEST(GpSdfMapping, GpOcc2Dd) { TestImpl2D<double>(); }

TEST(GpSdfMapping, GpOcc2Df) { TestImpl2D<float>(); }

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
