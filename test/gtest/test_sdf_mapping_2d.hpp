#pragma once

#include "erl_common/block_timer.hpp"
#include "erl_common/csv.hpp"
#include "erl_common/macros.hpp"
#include "erl_common/plplot_fig.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_common/yaml.hpp"
#include "erl_geometry/gazebo_room_2d.hpp"
#include "erl_geometry/house_expo_map.hpp"
#include "erl_geometry/lidar_2d.hpp"
#include "erl_geometry/occupancy_quadtree_drawer.hpp"
#include "erl_geometry/ucsd_fah_2d.hpp"
#include "erl_gp_sdf/gp_sdf_mapping.hpp"

enum class DataSetType {
    GazeboRoom2D = 1,
    HouseExpoLidar2D = 2,
    UcsdFah2D = 3,
};

template<typename Dtype>
struct Options : public erl::common::Yamlable<Options<Dtype>> {
    inline static const std::filesystem::path kProjectRootDir = ERL_GP_SDF_ROOT_DIR;
    inline static const std::filesystem::path kDataDir = kProjectRootDir / "data";
    inline static const std::filesystem::path kConfigDir = kProjectRootDir / "config";

    std::string dataset_name = "gazebo_room_2d";
    std::string gazebo_dir = kDataDir / "gazebo";
    std::string house_expo_map_file = kDataDir / "house_expo_room_1451.json";
    std::string house_expo_traj_file = kDataDir / "house_expo_room_1451.csv";
    std::string ucsd_fah_2d_file = kDataDir / "ucsd_fah_2d.dat";
    std::string surface_mapping_config_file;
    std::string sdf_mapping_config_file;
    bool load_sdf_mapping_bin = false;
    bool visualize = false;
    bool test_io = false;
    bool hold = false;
    bool interactive = false;
    bool save_video = false;
    long start_wp_idx = 0;
    long end_wp_idx = -1;
    long seq_stride = 1;
    long vis_stride = 1;
    Dtype map_resolution = 0.025;
    Dtype surf_normal_scale = 0.35;

    ERL_REFLECT_SCHEMA(
        Options,
        ERL_REFLECT_MEMBER(Options, dataset_name),
        ERL_REFLECT_MEMBER(Options, gazebo_dir),
        ERL_REFLECT_MEMBER(Options, house_expo_map_file),
        ERL_REFLECT_MEMBER(Options, house_expo_traj_file),
        ERL_REFLECT_MEMBER(Options, ucsd_fah_2d_file),
        ERL_REFLECT_MEMBER(Options, surface_mapping_config_file),
        ERL_REFLECT_MEMBER(Options, sdf_mapping_config_file),
        ERL_REFLECT_MEMBER(Options, load_sdf_mapping_bin),
        ERL_REFLECT_MEMBER(Options, visualize),
        ERL_REFLECT_MEMBER(Options, test_io),
        ERL_REFLECT_MEMBER(Options, hold),
        ERL_REFLECT_MEMBER(Options, interactive),
        ERL_REFLECT_MEMBER(Options, save_video),
        ERL_REFLECT_MEMBER(Options, start_wp_idx),
        ERL_REFLECT_MEMBER(Options, end_wp_idx),
        ERL_REFLECT_MEMBER(Options, seq_stride),
        ERL_REFLECT_MEMBER(Options, vis_stride),
        ERL_REFLECT_MEMBER(Options, map_resolution),
        ERL_REFLECT_MEMBER(Options, surf_normal_scale));
};

inline cv::Point
EigenToOpenCV(const Eigen::Vector2i &p) {
    return {p.x(), p.y()};
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
    const Drawer &drawer,
    const int x,
    const int y,
    const Eigen::Vector2<Dtype> &gradient) {
    const Eigen::VectorXi grad_pixel = drawer.template GetPixelCoordsForVectors<Dtype>(gradient);
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
        Eigen::Vector2i gp_position_px =
            drawer.template GetPixelCoordsForPositions<Dtype>(gp->position, true);
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
        Eigen::Vector2i gp_area_min_px =
            drawer.template GetPixelCoordsForPositions<Dtype>(gp_area_min, true);
        Eigen::Vector2i gp_area_max_px =
            drawer.template GetPixelCoordsForPositions<Dtype>(gp_area_max, true);
        cv::rectangle(
            img,
            cv::Point(gp_area_min_px[0], gp_area_min_px[1]),
            cv::Point(gp_area_max_px[0], gp_area_max_px[1]),
            rect_color,
            2);
    }

    if (!draw_data) { return; }

    typename erl::gaussian_process::NoisyInputGaussianProcess<Dtype>::TrainBuf &train_buf =
        gp->edf_gp->GetTrainBuffer();
    Eigen::Matrix2X<Dtype> used_surface_points = train_buf.x.block(0, 0, 2, train_buf.num_samples);
    Eigen::Matrix2Xi used_surface_points_px =
        drawer.template GetPixelCoordsForPositions<Dtype>(used_surface_points, true);
    for (long j = 0; j < used_surface_points.cols(); j++) {
        cv::circle(
            img,
            cv::Point(used_surface_points_px(0, j), used_surface_points_px(1, j)),
            3,
            data_color,
            -1);
    }
}

template<typename SurfaceMapping, typename SdfMapping, typename Drawer>
struct OpenCvUserData {
    std::string window_name;
    SurfaceMapping *surf_map = nullptr;
    SdfMapping *sdf_map = nullptr;
    Drawer *drawer = nullptr;
    cv::Mat img;
};

template<typename Dtype, typename SurfaceMapping, typename SdfMapping, typename Drawer>
void
OpenCvMouseCallback(const int event, const int x, const int y, int /*flags*/, void *userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        auto *data = static_cast<OpenCvUserData<SurfaceMapping, SdfMapping, Drawer> *>(userdata);
        Eigen::Vector2<Dtype> position =
            data->drawer->template GetMeterCoordsForPositions<Dtype>(Eigen::Vector2i(x, y), false);
        ERL_INFO("Clicked at [{:f}, {:f}].", position.x(), position.y());
        Eigen::VectorX<Dtype> distance(1);
        Eigen::Matrix2X<Dtype> gradient(2, 1);
        Eigen::Matrix3X<Dtype> variances(3, 1);
        Eigen::Matrix3X<Dtype> covariances(3, 1);
        if (data->sdf_map->Test(position, distance, gradient, variances, covariances)) {
            ERL_INFO(
                "SDF at [{:f}, {:f}]: {:f}, grad: [{:f}, {:f}], var: {}, cov: {}.",
                position.x(),
                position.y(),
                distance[0],
                gradient(0, 0),
                gradient(1, 0),
                variances.col(0).transpose(),
                covariances.col(0).transpose());

            auto gp = data->sdf_map->GetUsedGps()[0][0];
            if (gp == nullptr) { return; }
            ERL_INFO("pick {}", reinterpret_cast<uint64_t>(gp.get()));
            ERL_INFO("position: {}, half_size: {}", gp->position.transpose(), gp->half_size);
            erl::geometry::Aabb<Dtype, 2> aabb(gp->position, gp->half_size);
            std::vector<std::pair<Dtype, std::size_t>> distances_indices;
            data->surf_map->CollectSurfaceDataInAabb(aabb, distances_indices);
            ERL_INFO("Found {} surface data points in the area.", distances_indices.size());

            cv::Mat img = data->img.clone();

            // draw sdf
            const Dtype resolution = data->drawer->GetGridMapInfo()->Resolution(0);
            DrawSdf(img, x, y, distance[0], resolution);
            // draw sdf variance
            DrawSdfVariance(img, x, y, distance[0], variances(0, 0), resolution);
            // draw sdf gradient
            DrawSdfGradient<Dtype, Drawer>(img, *data->drawer, x, y, gradient.col(0));

            auto &[gp1, gp2] = data->sdf_map->GetUsedGps()[0];
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

template<typename Dtype, typename SurfaceMappingType>
struct TestSdfMapping2D {
    using SurfaceMapping = SurfaceMappingType;
    using SdfMapping = erl::gp_sdf::GpSdfMapping<Dtype, 2>;
    using SurfaceMappingSetting = typename SurfaceMapping::Setting;
    using SdfMappingSetting = typename SdfMapping::Setting;

    using GazeboRoom2D = erl::geometry::GazeboRoom2D;
    using HouseExpoMap = erl::geometry::HouseExpoMap;
    using UcsdFah2D = erl::geometry::UcsdFah2D;

    using Quadtree = typename SurfaceMapping::Tree;
    using QuadtreeDrawer = erl::geometry::OccupancyQuadtreeDrawer<Quadtree>;
    using QuadtreeDrawerSetting = typename QuadtreeDrawer::Setting;
    using GridMapInfo2D = erl::common::GridMapInfo2D<Dtype>;
    using Lidar2D = erl::geometry::Lidar2D;

    using PlplotFig = erl::common::PlplotFig;

    using VectorX = Eigen::VectorX<Dtype>;
    using Vector2 = Eigen::Vector2<Dtype>;
    using Matrix2 = Eigen::Matrix2<Dtype>;
    using Matrix3 = Eigen::Matrix3<Dtype>;
    using MatrixX = Eigen::MatrixX<Dtype>;
    using Matrix2X = Eigen::Matrix2X<Dtype>;
    using Matrix3X = Eigen::Matrix3X<Dtype>;

    Options<Dtype> options;

    std::shared_ptr<SurfaceMappingSetting> surf_map_setting = nullptr;
    std::shared_ptr<SdfMappingSetting> sdf_map_setting = nullptr;
    std::shared_ptr<SurfaceMapping> surf_map = nullptr;
    std::shared_ptr<SdfMapping> sdf_map = nullptr;

    // dataset

    DataSetType dataset_type;
    std::shared_ptr<GazeboRoom2D::TrainDataLoader> gazebo_room_2d = nullptr;
    std::shared_ptr<HouseExpoMap> house_expo_map = nullptr;
    std::shared_ptr<UcsdFah2D> ucsd_fah_2d = nullptr;
    std::vector<std::vector<Dtype>> trajectory;
    long max_wp_idx = 0;
    long wp_idx = 0;
    bool mapping_uses_points = false;  // should be set externally
    VectorX train_angles;
    VectorX train_ranges;
    Matrix2 rotation;
    Vector2 translation;

    // sensor

    std::shared_ptr<Lidar2D> lidar = nullptr;

    // visualization

    std::shared_ptr<QuadtreeDrawer> quadtree_drawer = nullptr;
    std::shared_ptr<GridMapInfo2D> grid_map_info = nullptr;
    PlplotFig fig_sdf{1280, 480, true};
    PlplotFig fig_grad{1280, 480, true};
    PlplotFig::LegendOpt legend_opt_sdf{3, {"SDF", "EDF", "Variance"}};
    PlplotFig::LegendOpt legend_opt_grad{4, {"grad_x", "grad_y", "var_grad_x", "var_grad_y"}};
    double t_ms = 0;
    double traj_t = 0;
    double t_span = 100;
    std::vector<double> timestamps_sec;
    std::vector<double> sdf_values;
    std::vector<double> edf_values;
    std::vector<double> var_sdf_values;
    std::vector<double> grad_x_values;
    std::vector<double> grad_y_values;
    std::vector<double> var_grad_x_values;
    std::vector<double> var_grad_y_values;
    cv::Mat img_scene;
    cv::Mat img_canvas;
    cv::Scalar color_trajectory{0, 0, 0, 255};
    cv::Scalar color_surf_point{0, 255, 125, 255};
    cv::Scalar color_normal_vec{0, 0, 255, 255};
    cv::Scalar color_text{0, 255, 0, 255};
    std::shared_ptr<cv::VideoWriter> video_writer = nullptr;
    std::string window_name;

    // test data
    Vector2 map_min, map_max;
    VectorX sdf_pred{1};
    Matrix2X sdf_gradient_pred{2, 1};
    Matrix3X sdf_var_pred{3, 1};
    Matrix3X sdf_covariances_pred{3, 1};

    // logging

    bool surf_map_updated = false;
    bool sdf_map_updated = false;
    bool test_success = false;
    Eigen::Matrix4Xd fps_data;
    double surf_map_update_dt = 0;
    double sdf_map_update_dt = 0;
    double test_dt = 0;
    double gui_dt = 0;
    double surf_map_update_fps = 0;
    double sdf_map_update_fps = 0;
    double test_fps = 0;
    double gui_fps = 0;
    Matrix2X grid_points;
    // Eigen::VectorXl gp_indices;
    // absl::flat_hash_map<uint64_t, long> gp_index_map;

    // output folders

    std::filesystem::path test_output_folder;
    std::filesystem::path img_dir;
    std::filesystem::path video_path;

    TestSdfMapping2D(const int argc, char *argv[]) {
        ParseOptions(argc, argv);
        LoadSetting();
        PrepareDataset();
        PrepareOutputFolders();
        PrepareVisualizer();
    }

    void
    Run() {
        if (options.load_sdf_mapping_bin) {
            ReadSdfMappingBin(*sdf_map);
        } else {
            if (options.test_io) { TestIo(); }

            for (; wp_idx < max_wp_idx; wp_idx += options.seq_stride) {
                ERL_INFO("wp_idx: {}", wp_idx);
                ERL_BLOCK_TIMER_MSG_TIME("gui_update", gui_dt);
                if (UpdateSdfMap()) { PredSdfFollow(); }
                if (wp_idx % options.vis_stride == 0) { UpdateVisualization(); }
            }
            ERL_INFO("gui_update (fps): {:.2f}", 1000.0 / gui_dt);

            if (options.save_video) {
                video_writer->release();
                ERL_INFO("Saved surface mapping video to {}.", video_path.c_str());
            }

            if (options.test_io) { TestIo(); }
        }

        ShowFinalResults();

        if (options.visualize && options.hold) {
            std::cout << "Press any key to exit." << std::endl;
            cv::waitKey(0);
        } else {
            constexpr double wait_time = 10.0;
            cv::waitKey(wait_time * 1000);  // wait for 10 seconds
        }

        if (options.interactive) { Interactive(); }
    }

    virtual ~TestSdfMapping2D() = default;

protected:
    // initialization

    void
    ParseOptions(int argc, char **argv) {
        options.FromCommandLine(argc, argv);

        if (options.dataset_name == "gazebo_room_2d") {
            dataset_type = DataSetType::GazeboRoom2D;
            ERL_ASSERTM(
                !options.gazebo_dir.empty(),
                "Please provide the Gazebo dataset directory via --gazebo_dir");
        } else if (options.dataset_name == "house_expo_lidar_2d") {
            dataset_type = DataSetType::HouseExpoLidar2D;
            ERL_ASSERTM(
                !options.house_expo_map_file.empty(),
                "Please provide the HouseExpo map file via --house_expo_map_file");
            ERL_ASSERTM(
                !options.house_expo_traj_file.empty(),
                "Please provide the HouseExpo trajectory file via --house_expo_traj_file");
        } else if (options.dataset_name == "ucsd_fah_2d") {
            dataset_type = DataSetType::UcsdFah2D;
            ERL_ASSERTM(
                !options.ucsd_fah_2d_file.empty(),
                "Please provide the ROS bag dat file via --ucsd_fah_2d_file");
        } else {
            ERL_FATAL("Unknown dataset name {} for 2D", options.dataset_name);
        }
    }

    void
    LoadSetting() {
        surf_map_setting = std::make_shared<SurfaceMappingSetting>();
        ERL_ASSERTM(
            surf_map_setting->FromYamlFile(options.surface_mapping_config_file),
            "Failed to load surface_mapping_config_file: {}",
            options.surface_mapping_config_file);

        sdf_map_setting = std::make_shared<SdfMappingSetting>();
        ERL_ASSERTM(
            sdf_map_setting->FromYamlFile(options.sdf_mapping_config_file),
            "Failed to load sdf_mapping_config_file: {}",
            options.sdf_mapping_config_file);

        ERL_INFO("Surface mapping config: {}", options.surface_mapping_config_file);
        std::cout << surf_map_setting->AsYamlString() << std::endl;

        ERL_INFO("SDF mapping config: {}", options.sdf_mapping_config_file);
        std::cout << sdf_map_setting->AsYamlString() << std::endl;

        surf_map = std::make_shared<SurfaceMapping>(surf_map_setting);
        sdf_map = std::make_shared<SdfMapping>(sdf_map_setting, surf_map);
    }

    void
    PrepareGazeboRoom2D() {
        // dataset
        gazebo_room_2d = std::make_shared<GazeboRoom2D::TrainDataLoader>(options.gazebo_dir);
        max_wp_idx = gazebo_room_2d->size();
        ERL_ASSERT_LT(options.start_wp_idx, max_wp_idx);
        ERL_ASSERT_POS_LT(options.end_wp_idx, max_wp_idx);
        train_angles = (*gazebo_room_2d)[0].angles.cast<Dtype>();
        // test data
        map_min = GazeboRoom2D::kMapMin.cast<Dtype>();
        map_max = GazeboRoom2D::kMapMax.cast<Dtype>();
    }

    void
    PrepareHouseExpoLidar2D() {
        house_expo_map = std::make_shared<HouseExpoMap>(options.house_expo_map_file, 0.2);
        trajectory = erl::common::LoadAndCastCsvFile<Dtype>(
            options.house_expo_traj_file,
            [](const std::string &str) -> double { return std::stod(str); });
        max_wp_idx = static_cast<long>(trajectory.size());
        ERL_ASSERT_LT(options.start_wp_idx, max_wp_idx);
        ERL_ASSERT_POS_LT(options.end_wp_idx, max_wp_idx);
        // sensor
        auto lidar_setting = std::make_shared<Lidar2D::Setting>();
        lidar_setting->num_lines = 720;
        lidar = std::make_shared<Lidar2D>(lidar_setting, house_expo_map->GetMeterSpace());
        train_angles = lidar->GetAngles().cast<Dtype>();
        // test data
        map_min = house_expo_map->GetMeterSpace()
                      ->GetSurface()
                      ->vertices.rowwise()
                      .minCoeff()
                      .cast<Dtype>();
        map_max = house_expo_map->GetMeterSpace()
                      ->GetSurface()
                      ->vertices.rowwise()
                      .maxCoeff()
                      .cast<Dtype>();
    }

    void
    PrepareUcsdFah2D() {
        ucsd_fah_2d = std::make_shared<UcsdFah2D>(options.ucsd_fah_2d_file);
        max_wp_idx = ucsd_fah_2d->Size();
        ERL_ASSERT_LT(options.start_wp_idx, max_wp_idx);
        ERL_ASSERT_POS_LT(options.end_wp_idx, max_wp_idx);
        train_angles = (*ucsd_fah_2d)[0].angles.cast<Dtype>();
        t_span = 500.0f * ucsd_fah_2d->GetTimeStep();
        // test data
        map_min = UcsdFah2D::kMapMin.cast<Dtype>();
        map_max = UcsdFah2D::kMapMax.cast<Dtype>();
    }

    void
    PrepareDataset() {
        switch (dataset_type) {
            case DataSetType::GazeboRoom2D:
                PrepareGazeboRoom2D();
                break;
            case DataSetType::HouseExpoLidar2D:
                PrepareHouseExpoLidar2D();
                break;
            case DataSetType::UcsdFah2D:
                PrepareUcsdFah2D();
                break;
            default:
                ERL_FATAL("Unsupported dataset type.");
        }

        max_wp_idx = (options.end_wp_idx == -1) ? max_wp_idx : options.end_wp_idx;
    }

    void
    PrepareOutputFolders() {
        GTEST_PREPARE_OUTPUT_DIR();
        test_output_folder = test_output_dir;
        img_dir = test_output_folder / "images";
        video_path = test_output_folder / "sdf_mapping.avi";
        std::filesystem::create_directory(img_dir);
        window_name = test_info->name();
    }

    void
    PrepareVisualizer() {
        legend_opt_sdf
            .SetTextColors(
                {PlplotFig::Color0::Red, PlplotFig::Color0::Blue, PlplotFig::Color0::Green})
            .SetStyles({PL_LEGEND_LINE, PL_LEGEND_LINE, PL_LEGEND_LINE})
            .SetLineColors(legend_opt_sdf.text_colors)
            .SetLineStyles({1, 1, 1})
            .SetLineWidths({1.0, 1.0, 1.0})
            .SetPosition(PL_POSITION_LEFT | PL_POSITION_TOP)
            .SetBoxStyle(PL_LEGEND_BOUNDING_BOX | PL_LEGEND_BACKGROUND)
            .SetLegendBoxLineColor0(PlplotFig::Color0::Black)
            .SetBgColor0(PlplotFig::Color0::Gray)
            .SetTextScale(1.1);

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

        timestamps_sec.reserve(max_wp_idx);
        sdf_values.reserve(max_wp_idx);
        edf_values.reserve(max_wp_idx);
        var_sdf_values.reserve(max_wp_idx);
        grad_x_values.reserve(max_wp_idx);
        grad_y_values.reserve(max_wp_idx);
        var_grad_x_values.reserve(max_wp_idx);
        var_grad_y_values.reserve(max_wp_idx);

        auto drawer_setting = std::make_shared<QuadtreeDrawerSetting>();
        drawer_setting->area_min = map_min.template cast<float>();
        drawer_setting->area_max = map_max.template cast<float>();
        drawer_setting->resolution = options.map_resolution;
        drawer_setting->scaling = surf_map_setting->scaling;
        drawer_setting->padding = 1;
        drawer_setting->border_color = cv::Scalar(255, 0, 0, 255);
        quadtree_drawer = std::make_shared<QuadtreeDrawer>(drawer_setting, surf_map->GetTree());
        grid_map_info = quadtree_drawer->GetGridMapInfo()->template CastSharedPtr<Dtype>();
        options.map_resolution = grid_map_info->Resolution(0);
        grid_points = grid_map_info->GenerateMeterCoordinates(true);

        if (options.save_video) {
            cv::Size frame_size(
                grid_map_info->Width() + 1280,
                std::max(grid_map_info->Height(), 960));
            video_writer = std::make_shared<cv::VideoWriter>(
                video_path,
                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                30.0,
                frame_size);
        }
    }

    std::string
    GetBinFileName() {
        std::string bin_file = fmt::format("sdf_mapping_2d_{}.bin", type_name<Dtype>());
        bin_file = test_output_folder / bin_file;
        return bin_file;
    }

    void
    WriteSdfMappingBin() {
        ERL_BLOCK_TIMER_MSG("WriteSdfMappingBin");
        std::string bin_file = GetBinFileName();
        using namespace erl::common::serialization;
        ERL_ASSERTM(
            Serialization<SdfMapping>::Write(bin_file, sdf_map),
            "Failed to write to file: {}",
            bin_file);
    }

    void
    ReadSdfMappingBin(SdfMapping &sdf_mapping_read) {
        ERL_BLOCK_TIMER_MSG("ReadSdfMappingBin");
        std::string bin_file = GetBinFileName();
        using namespace erl::common::serialization;
        ERL_ASSERTM(
            Serialization<SdfMapping>::Read(bin_file, &sdf_mapping_read),
            "Failed to read from file: {}",
            bin_file);
    }

    void
    TestIo() {
        ERL_BLOCK_TIMER_MSG("IO");
        WriteSdfMappingBin();

        auto surface_mapping_read =
            std::make_shared<SurfaceMapping>(std::make_shared<typename SurfaceMapping::Setting>());
        SdfMapping sdf_mapping_read(std::make_shared<SdfMappingSetting>(), surface_mapping_read);
        ReadSdfMappingBin(sdf_mapping_read);

        ERL_ASSERTM(*sdf_map == sdf_mapping_read, "sdf_map != sdf_mapping_read");
    }

    void
    LoadDataFromGazeboRoom2D() {
        const auto &frame = (*gazebo_room_2d)[wp_idx];
        rotation = frame.rotation.cast<Dtype>();
        translation = frame.translation.cast<Dtype>();
        train_ranges = frame.ranges.cast<Dtype>();
        traj_t += 0.2;  // assume 5 Hz
    }

    void
    LoadDataFromHouseExpoLidar2D() {
        const std::vector<Dtype> &wp = trajectory[wp_idx];
        rotation = Eigen::Rotation2D<Dtype>(wp[2]).toRotationMatrix();
        translation[0] = wp[0];
        translation[1] = wp[1];
        train_ranges =
            lidar->Scan(wp[2], translation.template cast<double>(), true).template cast<Dtype>();
        traj_t += 0.2;  // assume 5 Hz
    }

    void
    LoadDataFromUcsdFah2D() {
        auto
            [sequence_number,
             timestamp,
             header_timestamp,
             rotation_mat,
             translation_vec,
             angles,
             ranges] = (*ucsd_fah_2d)[wp_idx];
        rotation = rotation_mat.cast<Dtype>();
        translation = translation_vec.cast<Dtype>();
        train_ranges = ranges.cast<Dtype>();
        traj_t += ucsd_fah_2d->GetTimeStep();
    }

    void
    LoadData() {
        ERL_BLOCK_TIMER_MSG("data loading");
        switch (dataset_type) {
            case DataSetType::GazeboRoom2D:
                LoadDataFromGazeboRoom2D();
                break;
            case DataSetType::HouseExpoLidar2D:
                LoadDataFromHouseExpoLidar2D();
                break;
            case DataSetType::UcsdFah2D:
                LoadDataFromUcsdFah2D();
                break;
            default:
                ERL_FATAL("Unsupported dataset type.");
        }
    }

    bool
    UpdateSurfaceMap() {
        LoadData();

        if (!mapping_uses_points) {
            ERL_BLOCK_TIMER_MSG_TIME("surf_map.Update", surf_map_update_dt);
            return surf_map->Update(rotation, translation, train_ranges, false, true);
        }

        // transform points from sensor frame to world frame
        Matrix2X points(2, train_ranges.size());
#pragma omp parallel for default(none) schedule(static) shared(points)
        for (long i = 0; i < train_ranges.size(); ++i) {
            // clang-format off
            points.col(i) << train_ranges[i] * std::cos(train_angles[i]),
                             train_ranges[i] * std::sin(train_angles[i]);
            // clang-format on
            points.col(i) = rotation * points.col(i) + translation;
        }

        {
            ERL_BLOCK_TIMER_MSG_TIME("surf_map.Update", surf_map_update_dt);
            return surf_map->Update(rotation, translation, points, true, false);
        }
    }

    bool
    UpdateSdfMap() {
        ERL_BLOCK_TIMER_MSG_TIME("sdf_map.Update", sdf_map_update_dt);

        surf_map_updated = UpdateSurfaceMap();
        ERL_WARN_COND(!surf_map_updated, "Sdf mapping update failed");
        if (!surf_map_updated) { return false; }

        const double time_budget_us = 1e6 / sdf_map_setting->update_hz;  // us
        sdf_map_updated = sdf_map->UpdateGpSdf(time_budget_us - surf_map_update_dt * 1000);
        ERL_WARN_COND(!sdf_map_updated, "Sdf mapping update failed");
        return sdf_map_updated;
    }

    void
    PredSdfFollow() {
        ERL_BLOCK_TIMER_MSG_TIME("sdf_map.Test", test_dt);
        test_success = sdf_map->Test(
            translation,
            sdf_pred,
            sdf_gradient_pred,
            sdf_var_pred,
            sdf_covariances_pred);
    }

    static void
    DrawCurve(
        PlplotFig &fig,
        const PlplotFig::Color0 color_idx,
        const int n,
        const double *ts,
        const double *vs) {
        fig.SetCurrentColor(color_idx).SetPenWidth(2).DrawLine(n, ts, vs).SetPenWidth(1);
    }

    void
    DrawSurfaceData(cv::Mat &img) {
        for (auto it = surf_map->BeginSurfaceData(), end = surf_map->EndSurfaceData(); it != end;
             ++it) {
            Eigen::Vector2i position_px =
                quadtree_drawer->template GetPixelCoordsForPositions<Dtype>(it->position, true);
            const cv::Point position_px_cv(position_px[0], position_px[1]);
            cv::circle(img, position_px_cv, 2, color_surf_point, -1);
            Eigen::Vector2i normal_px = quadtree_drawer->template GetPixelCoordsForVectors<Dtype>(
                it->normal * options.surf_normal_scale);
            const cv::Point arrow_end_px(
                position_px[0] + normal_px[0],
                position_px[1] + normal_px[1]);
            cv::arrowedLine(
                img,
                position_px_cv,
                arrow_end_px,
                color_normal_vec,
                1,
                cv::LINE_8,
                0,
                0.1);
        }
    }

    virtual void
    InitSceneImg() {}

    void
    UpdateSceneImg() {
        if (surf_map_updated) { surf_map_update_fps = 1000.0 / surf_map_update_dt; }
        if (sdf_map_updated) { sdf_map_update_fps = 1000.0 / sdf_map_update_dt; }
        if (test_success) { test_fps = 1000.0 / test_dt; }
        if (gui_dt > 0) { gui_fps = 1000.0 / gui_dt; }

        InitSceneImg();

        // Visualize the results
        cv::Point position_px = EigenToOpenCV(grid_map_info->MeterToPixelForPoints(translation));
        DrawSdf<Dtype>(
            img_scene,
            position_px.x,
            position_px.y,
            sdf_pred[0],
            options.map_resolution);
        DrawSdfVariance<Dtype>(
            img_scene,
            position_px.x,
            position_px.y,
            sdf_pred[0],
            sdf_var_pred(0, 0),
            options.map_resolution);
        DrawSdfGradient(
            img_scene,
            *quadtree_drawer,
            position_px.x,
            position_px.y,
            Eigen::Vector2<Dtype>(sdf_gradient_pred.col(0)));

        // draw used surface points
        if (test_success) {
            auto &gps = VEC_ACCESS(sdf_map->GetUsedGps(), 0);
            auto gp1 = gps[0];
            auto gp2 = gps[1];
            if (gp1 != nullptr) {
                DrawGp<Dtype>(
                    img_scene,
                    gp1,
                    *quadtree_drawer,
                    {0, 125, 255, 255},
                    {255, 125, 0, 255});
                const auto &train_buf = gp1->edf_gp->GetTrainBuffer();
                ERL_INFO(
                    "GP1 at [{:f}, {:f}] has {} data points.",
                    gp1->position.x(),
                    gp1->position.y(),
                    train_buf.num_samples);
            }
            if (gp2 != nullptr) {
                DrawGp<Dtype>(
                    img_scene,
                    gp2,
                    *quadtree_drawer,
                    {125, 125, 255, 255},
                    {125, 255, 125, 255});
                const auto &train_buf = gp2->edf_gp->GetTrainBuffer();
                ERL_INFO(
                    "GP2 at [{:f}, {:f}] has {} data points.",
                    gp2->position.x(),
                    gp2->position.y(),
                    train_buf.num_samples);
            }
        }

        // draw trajectory
        erl::common::DrawTrajectoryInplace<Dtype>(
            img_scene,
            translation,
            grid_map_info,
            {0, 0, 0, 255},
            2,
            /*pixel_based*/ true);

        // draw fps
        constexpr int kFontFace = cv::FONT_HERSHEY_PLAIN;
        constexpr double kFontScale = 1.0;
        constexpr int kThickness = 1;
        cv::putText(
            img_scene,
            fmt::format("surf_map.Update: {:.2f} fps", surf_map_update_fps),
            cv::Point(10, 15),
            kFontFace,
            kFontScale,
            color_text,
            kThickness);
        cv::putText(
            img_scene,
            fmt::format("sdf_map.Update: {:.2f} fps", sdf_map_update_fps),
            cv::Point(10, 30),
            kFontFace,
            kFontScale,
            color_text,
            kThickness);
        cv::putText(
            img_scene,
            fmt::format("sdf_map.Test: {:.2f} fps", test_fps),
            cv::Point(10, 45),
            kFontFace,
            kFontScale,
            color_text,
            kThickness);
        cv::putText(
            img_scene,
            fmt::format("GUI: {:.2f} fps", gui_fps),
            cv::Point(10, 60),
            kFontFace,
            kFontScale,
            color_text,
            kThickness);
    }

    void
    UpdateCurves() {
        timestamps_sec.push_back(traj_t);
        sdf_values.push_back(sdf_pred[0]);
        edf_values.push_back(std::abs(sdf_pred[0]));
        var_sdf_values.push_back(sdf_var_pred(0, 0));
        grad_x_values.push_back(sdf_gradient_pred(0, 0));
        grad_y_values.push_back(sdf_gradient_pred(1, 0));
        var_grad_x_values.push_back(sdf_var_pred(1, 0));
        var_grad_y_values.push_back(sdf_var_pred(2, 0));
        const double t_min = traj_t - t_span;
        int n = 0;
        for (; n < static_cast<int>(timestamps_sec.size()) && timestamps_sec[n] < t_min; ++n) {}
        // skip the first n points
        if (n > 0) {
            timestamps_sec.erase(timestamps_sec.begin(), timestamps_sec.begin() + n);
            sdf_values.erase(sdf_values.begin(), sdf_values.begin() + n);
            edf_values.erase(edf_values.begin(), edf_values.begin() + n);
            var_sdf_values.erase(var_sdf_values.begin(), var_sdf_values.begin() + n);
            grad_x_values.erase(grad_x_values.begin(), grad_x_values.begin() + n);
            grad_y_values.erase(grad_y_values.begin(), grad_y_values.begin() + n);
            var_grad_x_values.erase(var_grad_x_values.begin(), var_grad_x_values.begin() + n);
            var_grad_y_values.erase(var_grad_y_values.begin(), var_grad_y_values.begin() + n);
        }
        n = static_cast<int>(timestamps_sec.size());
        if (timestamps_sec.empty()) { return; }

        auto minmax = std::minmax_element(sdf_values.begin(), sdf_values.end());
        double fig_sdf_y_min = *minmax.first;
        double fig_sdf_y_max = *minmax.second;

        // render fig_sdf
        minmax = std::minmax_element(edf_values.begin(), edf_values.end());
        fig_sdf_y_min = std::min(fig_sdf_y_min, *minmax.first) - 0.1;
        fig_sdf_y_max = std::max(fig_sdf_y_max, *minmax.second) + 0.1;

        fig_sdf.Clear()
            .SetMargin(0.15, 0.85, 0.15, 0.85)
            .SetAxisLimits(traj_t - t_span, traj_t, fig_sdf_y_min, fig_sdf_y_max)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .DrawAxesBox(
                PlplotFig::AxisOpt().DrawTopRightEdge(),
                PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
            .SetAxisLabelX("time (sec)")
            .SetCurrentColor(PlplotFig::Color0::Black)
            .SetAxisLabelY("SDF/EDF (meter)");

        DrawCurve(fig_sdf, PlplotFig::Color0::Red, n, timestamps_sec.data(), sdf_values.data());
        DrawCurve(fig_sdf, PlplotFig::Color0::Blue, n, timestamps_sec.data(), edf_values.data());

        minmax = std::minmax_element(var_sdf_values.begin(), var_sdf_values.end());
        fig_sdf.SetCurrentColor(PlplotFig::Color0::Black)
            .SetAxisLimits(traj_t - t_span, traj_t, *minmax.first - 0.001, *minmax.second + 0.001)
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
                fmt::format("sdf: {:.2f}, var_sdf: {:.2e}", sdf_pred[0], sdf_var_pred(0, 0))
                    .c_str());
        DrawCurve(
            fig_sdf,
            PlplotFig::Color0::Green,
            n,
            timestamps_sec.data(),
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
            .SetAxisLimits(traj_t - t_span, traj_t, fig_grad_y_min, fig_grad_y_max)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .DrawAxesBox(
                PlplotFig::AxisOpt().DrawTopRightEdge(),
                PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
            .SetAxisLabelX("time (sec)")
            .SetCurrentColor(PlplotFig::Color0::Black)
            .SetAxisLabelY("Gradient (meter)");

        DrawCurve(fig_grad, PlplotFig::Color0::Red, n, timestamps_sec.data(), grad_x_values.data());
        DrawCurve(
            fig_grad,
            PlplotFig::Color0::Blue,
            n,
            timestamps_sec.data(),
            grad_y_values.data());

        minmax = std::minmax_element(var_grad_x_values.begin(), var_grad_x_values.end());
        fig_grad_y_min = *minmax.first;
        fig_grad_y_max = *minmax.second;

        minmax = std::minmax_element(var_grad_y_values.begin(), var_grad_y_values.end());
        fig_grad_y_min = std::min(fig_grad_y_min, *minmax.first) - 0.001;
        fig_grad_y_max = std::max(fig_grad_y_max, *minmax.second) + 0.001;

        fig_grad.SetCurrentColor(PlplotFig::Color0::Black)
            .SetAxisLimits(traj_t - t_span, traj_t, fig_grad_y_min, fig_grad_y_max)
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
                fmt::format(
                    "grad: [{:.2f}, {:.2f}], var_grad: [{:.2e}, {:.2e}]",
                    sdf_gradient_pred(0, 0),
                    sdf_gradient_pred(1, 0),
                    sdf_var_pred(1, 0),
                    sdf_var_pred(2, 0))
                    .c_str());
        DrawCurve(
            fig_grad,
            PlplotFig::Color0::Green,
            n,
            timestamps_sec.data(),
            var_grad_x_values.data());
        DrawCurve(
            fig_grad,
            PlplotFig::Color0::Brown,
            n,
            timestamps_sec.data(),
            var_grad_y_values.data());
        fig_grad.SetFontSize(0.0, 0.8).Legend(legend_opt_grad).SetFontSize();
    }

    void
    UpdateCanvas() {
        cv::Mat tmp(
            std::max(img_scene.rows, 960),
            img_scene.cols + 1280,
            CV_8UC4,
            cv::Scalar(255, 255, 255, 255));
        if (img_scene.rows == tmp.rows) {
            const int offset = (tmp.rows - fig_sdf.Height() * 2) / 2;
            img_scene.copyTo(tmp(cv::Rect(0, 0, img_scene.cols, img_scene.rows)));
            fig_sdf.ToCvMat().copyTo(
                tmp(cv::Rect(img_scene.cols, offset, fig_sdf.Width(), fig_sdf.Height())));
            fig_grad.ToCvMat().copyTo(
                tmp(cv::Rect(
                    img_scene.cols,
                    offset + fig_sdf.Height(),
                    fig_grad.Width(),
                    fig_grad.Height())));
        } else {
            const int offset = (tmp.rows - img_scene.rows) / 2;
            img_scene.copyTo(tmp(cv::Rect(0, offset, img_scene.cols, img_scene.rows)));
            fig_sdf.ToCvMat().copyTo(
                tmp(cv::Rect(img_scene.cols, 0, fig_sdf.Width(), fig_sdf.Height())));
            fig_grad.ToCvMat().copyTo(tmp(
                cv::Rect(img_scene.cols, fig_sdf.Height(), fig_grad.Width(), fig_grad.Height())));
        }
        cv::cvtColor(tmp, img_canvas, cv::COLOR_BGRA2BGR);
        if (options.save_video) { video_writer->write(img_canvas); }
        cv::imshow(window_name, img_canvas);
        cv::waitKey(1);
    }

    void
    UpdateVisualization() {
        if (!options.visualize) { return; }
        ERL_BLOCK_TIMER_MSG("UpdateVisualization");

        UpdateSceneImg();
        UpdateCurves();
        UpdateCanvas();
    }

    void
    ShowFinalResults() {
        if (!options.visualize) { return; }

        VectorX sdf_out(grid_points.cols());
        Matrix2X grads_out(2, grid_points.cols());
        Matrix3X var_out(3, grid_points.cols());
        Matrix3X cov_out;
        const auto t0 = std::chrono::high_resolution_clock::now();
        bool success = sdf_map->Test(grid_points, sdf_out, grads_out, var_out, cov_out);
        const auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double t_per_point = dt / static_cast<double>(grid_points.cols()) * 1000;  // us
        ERL_ASSERTM(success, "Final SDF test failed.");
        ERL_INFO(
            "Test time: {:f} ms for {} points, {:f} us per point.",
            dt,
            grid_points.cols(),
            t_per_point);
        Dtype min_sdf = sdf_out.minCoeff();
        Dtype max_sdf = sdf_out.maxCoeff();
        ERL_INFO("min SDF: {:f}, max SDF: {:f}.", min_sdf, max_sdf);

        InitSceneImg();

        // draw all GPs
        cv::Mat img_gp = img_scene.clone();
        int gp_cnt = 0;
        for (const auto &[key, sdf_gp]: sdf_map->GetGpMap()) {
            if (sdf_gp == nullptr) { continue; }
            ++gp_cnt;
            DrawGp<Dtype>(
                img_gp,
                sdf_gp,
                *quadtree_drawer,
                {0, 125, 255, 255},
                {255, 125, 0, 255},
                {0, 0, 0, 255},
                false,
                true,
                gp_cnt < 100);  // draw only the first 100 GPs' bounding boxes
        }
        cv::imshow(window_name + ": GPs", img_gp);
        cv::imwrite(img_dir / "gps.png", img_gp);

        // draw SDF map
        cv::Mat img_sdf(
            grid_map_info->Width(),
            grid_map_info->Height(),
            sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1,
            sdf_out.data());
        img_sdf = img_sdf.t();
        cv::normalize(img_sdf, img_sdf, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::flip(img_sdf, img_sdf, 0);
        cv::applyColorMap(img_sdf, img_sdf, cv::COLORMAP_JET);
        cv::cvtColor(img_sdf, img_sdf, cv::COLOR_BGR2BGRA);
        cv::addWeighted(img_sdf, 0.5, img_scene, 0.5, 0.0, img_sdf);
        cv::imshow(window_name + ": sdf", img_sdf);
        cv::imwrite(img_dir / "sdf.png", img_sdf);

        // convert to binary image: 0 for negative, 255 for positive
        Eigen::VectorXi sdf_sign = (sdf_out.array() >= 0).template cast<int>();
        cv::Mat img_sign(
            grid_map_info->Width(),
            grid_map_info->Height(),
            CV_32SC1,
            sdf_sign.data());
        img_sign = img_sign.t();
        cv::normalize(img_sign, img_sign, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::flip(img_sign, img_sign, 0);  // flip along y axis
        cv::imshow(window_name + ": sdf sign", img_sign);
        cv::imwrite(img_dir / "sdf_sign.png", img_sign);

        // draw SDF variance map
        VectorX sdf_variances = var_out.row(0).transpose();
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
        cv::addWeighted(img_sdf_variance, 0.5, img_scene, 0.5, 0.0, img_sdf_variance);
        cv::imshow(window_name + ": sdf variance", img_sdf_variance);
        cv::imwrite(img_dir / "sdf_variance.png", img_sdf_variance);

        // show
        cv::waitKey(1);
    }

    void
    Interactive() {
        if (!options.interactive) { return; }

        InitSceneImg();

        OpenCvUserData<SurfaceMapping, SdfMapping, QuadtreeDrawer> data;
        data.window_name = window_name + ": interactive";
        data.img = img_scene;
        data.drawer = quadtree_drawer.get();
        data.surf_map = surf_map.get();
        data.sdf_map = sdf_map.get();

        cv::imshow(data.window_name, img_scene);
        cv::setMouseCallback(
            data.window_name,
            OpenCvMouseCallback<Dtype, SurfaceMapping, SdfMapping, QuadtreeDrawer>,
            &data);
        while (cv::waitKey(0) != 27) {}  // wait for the ESC key
    }
};
