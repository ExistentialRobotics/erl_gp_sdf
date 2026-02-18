#pragma once

#include "erl_common/block_timer.hpp"
#include "erl_common/csv.hpp"
#include "erl_common/plplot_fig.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_common/yaml.hpp"
#include "erl_geometry/gazebo_room_2d.hpp"
#include "erl_geometry/house_expo_map.hpp"
#include "erl_geometry/lidar_2d.hpp"
#include "erl_geometry/occupancy_quadtree.hpp"
#include "erl_geometry/occupancy_quadtree_drawer.hpp"
#include "erl_geometry/ucsd_fah_2d.hpp"

#include <open3d/geometry/LineSet.h>
#include <open3d/io/LineSetIO.h>

enum class DataSetType {
    GazeboRoom2D = 1,
    HouseExpoLidar2D = 2,
    UcsdFah2D = 3,
};
ERL_REFLECT_ENUM_SCHEMA(
    DataSetType,
    3,
    ERL_REFLECT_ENUM_MEMBER("gazebo_room_2d", DataSetType::GazeboRoom2D),
    ERL_REFLECT_ENUM_MEMBER("house_expo_lidar_2d", DataSetType::HouseExpoLidar2D),
    ERL_REFLECT_ENUM_MEMBER("ucsd_fah_2d", DataSetType::UcsdFah2D));
ERL_PARSE_ENUM(DataSetType, 3);

template<typename Dtype>
struct GridDef : public erl::common::Yamlable<GridDef<Dtype>> {
    using Vector2 = Eigen::Vector2<Dtype>;

    Vector2 size = Vector2::Zero();    // grid size, 0 means auto
    Vector2 center = Vector2::Zero();  // grid center, 0 means auto
    Dtype rotation = 0;                // grid rotation (radian)

    ERL_REFLECT_SCHEMA(
        GridDef,
        ERL_REFLECT_MEMBER(GridDef, size),
        ERL_REFLECT_MEMBER(GridDef, center),
        ERL_REFLECT_MEMBER(GridDef, rotation));
};

template<typename Dtype>
struct OptionsForTestMapping2D : public erl::common::Yamlable<OptionsForTestMapping2D<Dtype>> {
    inline static const std::filesystem::path kProjectRootDir = ERL_GP_SDF_ROOT_DIR;
    inline static const std::filesystem::path kDataDir = kProjectRootDir / "data";
    inline static const std::filesystem::path kConfigDir = kProjectRootDir / "config";

    using Vector2 = Eigen::Vector2<Dtype>;

    uint64_t random_seed = 0;
    std::filesystem::path output_dir;
    bool add_datetime_to_output_dir = true;
    DataSetType dataset_type = DataSetType::GazeboRoom2D;
    std::string gazebo_dir = kDataDir / "gazebo";
    std::string house_expo_map_file = kDataDir / "house_expo_room_1451.json";
    std::string house_expo_traj_file = kDataDir / "house_expo_room_1451.csv";
    std::string ucsd_fah_2d_file = kDataDir / "ucsd_fah_2d.dat";

    bool visualize = false;
    bool test_io = false;
    bool hold = false;
    bool interactive = false;
    bool save_images = false;
    bool save_video = false;
    long start_wp_idx = 0;
    long end_wp_idx = -1;
    long seq_stride = 1;
    long vis_stride = 1;
    long num_final_iterations = -1;
    Dtype map_resolution = 0.025;
    Dtype surf_normal_scale = 0.35;

    Dtype test_res_grid = 0.025;
    GridDef<Dtype> test_grid_def;
    bool test_grid_from_dataset = false;
    std::string test_grid_def_yaml_file;
    bool test_grid_at_end = false;

    bool extract_mesh = false;
    Dtype extract_mesh_res = 0.03;
    bool save_built_mesh = true;

    std::string mapping_bin_file;
    bool load_mapping_bin = false;

    ERL_REFLECT_SCHEMA(
        OptionsForTestMapping2D,
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, random_seed),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, output_dir),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, add_datetime_to_output_dir),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, dataset_type),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, gazebo_dir),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, house_expo_map_file),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, house_expo_traj_file),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, ucsd_fah_2d_file),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, visualize),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, test_io),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, hold),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, interactive),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, save_images),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, save_video),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, start_wp_idx),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, end_wp_idx),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, seq_stride),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, vis_stride),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, num_final_iterations),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, map_resolution),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, surf_normal_scale),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, test_res_grid),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, test_grid_def),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, test_grid_from_dataset),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, test_grid_def_yaml_file),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, test_grid_at_end),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, extract_mesh),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, extract_mesh_res),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, save_built_mesh),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, mapping_bin_file),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, load_mapping_bin));

    bool
    PostDeserialization() override {
        switch (dataset_type) {
            case DataSetType::GazeboRoom2D:
                ERL_ASSERTM(
                    !gazebo_dir.empty(),
                    "Please provide the Gazebo Room 2D dataset directory via --gazebo_dir");
                break;
            case DataSetType::HouseExpoLidar2D:
                ERL_ASSERTM(
                    !house_expo_map_file.empty(),
                    "Please provide the mesh file via --house_expo_map_file");
                ERL_ASSERTM(
                    !house_expo_traj_file.empty(),
                    "Please provide the house_expo_traj file via --house_expo_traj_file");
                break;
            case DataSetType::UcsdFah2D:
                ERL_ASSERTM(
                    !ucsd_fah_2d_file.empty(),
                    "Please provide the UcsdFah2D dataset directory via --ucsd_fah_2d_file");
                break;
            default:
                ERL_FATAL("Unknown dataset type.");
        }

        if (load_mapping_bin) {
            ERL_ASSERTM(
                !mapping_bin_file.empty(),
                "Please provide the SDF mapping bin file via --mapping_bin_file");
        }
        return true;
    }
};

template<typename Dtype, typename MappingType>
struct TestMapping2D {

    using GazeboRoom2D = erl::geometry::GazeboRoom2D;
    using HouseExpoMap = erl::geometry::HouseExpoMap;
    using UcsdFah2D = erl::geometry::UcsdFah2D;

    using Lidar2D = erl::geometry::Lidar2D;
    using GridMapInfo2D = erl::common::GridMapInfo2D<Dtype>;
    using Quadtree = erl::geometry::OccupancyQuadtree<Dtype>;
    using QuadtreeDrawer = erl::geometry::OccupancyQuadtreeDrawer<Quadtree>;
    using TreeDrawerSetting = typename QuadtreeDrawer::Setting;

    using VectorX = Eigen::VectorX<Dtype>;
    using Vector2 = Eigen::Vector2<Dtype>;
    using Matrix2 = Eigen::Matrix2<Dtype>;
    using Matrix2X = Eigen::Matrix2X<Dtype>;

    std::shared_ptr<MappingType> mapping = nullptr;
    std::shared_ptr<const Quadtree> quadtree = nullptr;

    // dataset

    DataSetType dataset_type = DataSetType::GazeboRoom2D;
    std::shared_ptr<GazeboRoom2D::TrainDataLoader> gazebo_room_2d = nullptr;
    std::shared_ptr<HouseExpoMap> house_expo_map = nullptr;
    std::vector<std::vector<Dtype>> house_expo_traj;
    std::shared_ptr<UcsdFah2D> ucsd_fah_2d = nullptr;
    long max_wp_idx = 0;
    long wp_idx = 0;
    bool mapping_uses_points = false;  // should be set externally
    VectorX train_angles;
    VectorX train_ranges;
    Matrix2X train_frame_points;
    Matrix2X train_world_points;
    Matrix2 rotation;
    Vector2 translation;
    double t_span = 100;
    double traj_t = 0;
    Vector2 map_min;
    Vector2 map_max;
    Vector2 map_translation;
    Matrix2 map_rotation;
    std::vector<cv::Point2i> surface_points_cv;
    std::vector<Vector2> cur_traj;

    // sensor

    std::shared_ptr<Lidar2D> lidar = nullptr;

    // visualization
    std::string window_name;
    cv::Mat img_canvas;
    std::shared_ptr<cv::VideoWriter> video_writer = nullptr;
    std::shared_ptr<TreeDrawerSetting> tree_drawer_setting = std::make_shared<TreeDrawerSetting>();
    std::shared_ptr<QuadtreeDrawer> quadtree_drawer = nullptr;
    std::shared_ptr<GridMapInfo2D> grid_map_info = nullptr;
    int key = -1;

    // output folders

    std::filesystem::path img_dir;
    std::filesystem::path video_path;

    // logging

    double update_map_dt = 0.0;
    double update_pred_dt = 0.0;
    double update_vis_dt = 0.0;
    double update_map_fps = 0.0;
    double update_pred_fps = 0.0;
    double update_vis_fps = 0.0;
    Eigen::MatrixXd fps_data;

private:
    std::shared_ptr<OptionsForTestMapping2D<Dtype>> options = nullptr;

protected:
    Dtype scaling = 1.0;

public:
    TestMapping2D(
        const int argc,
        char *argv[],
        std::shared_ptr<OptionsForTestMapping2D<Dtype>> options_in)
        : options(std::move(options_in)) {
        ERL_ASSERT(options->FromCommandLine(argc, argv));
        erl::common::SetGlobalRandomSeed(options->random_seed);
    }

    TestMapping2D(const TestMapping2D &) = delete;
    TestMapping2D &
    operator=(const TestMapping2D &) = delete;
    TestMapping2D(TestMapping2D &&) = delete;
    TestMapping2D &
    operator=(TestMapping2D &&) = delete;

    virtual ~TestMapping2D() = default;

    virtual void
    Run() {
        Init();

        if (options->load_mapping_bin) {
            ReadMappingBin(*mapping, options->mapping_bin_file);
        } else {
            if (options->test_io) { TestIo(); }

            auto update_map = [this]() {
                ERL_BLOCK_TIMER_MSG_TIME("[App] UpdateMap", this->update_map_dt);
                return this->UpdateMap();
            };

            auto update_pred = [this]() {
                ERL_BLOCK_TIMER_MSG_TIME("[App] UpdatePrediction", this->update_pred_dt);
                this->UpdatePrediction();
            };

            auto update_vis = [this]() {
                if (!options->visualize) { return; }
                ERL_BLOCK_TIMER_MSG_TIME("[App] UpdateVisualization", this->update_vis_dt);
                this->UpdateVisualization();
            };

            for (; wp_idx < max_wp_idx; wp_idx += options->seq_stride) {
                ERL_INFO("wp_idx: {}", wp_idx);
                LoadData();
                if (update_map()) { update_pred(); }
                if (wp_idx % options->vis_stride == 0) {
                    update_vis();
                    if (options->save_video && img_canvas.rows > 0 && img_canvas.cols > 0) {
                        if (video_writer == nullptr) {
                            cv::Size frame_size(img_canvas.rows, img_canvas.cols);
                            video_writer = std::make_shared<cv::VideoWriter>(
                                video_path,
                                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                30.0,
                                frame_size);
                        }
                        video_writer->write(img_canvas);
                    }
                }

                if (update_map_dt > 0) { update_map_fps = 1000.0 / update_map_dt; }
                if (update_pred_dt > 0) { update_pred_fps = 1000.0 / update_pred_dt; }
                if (update_vis_dt > 0) { update_vis_fps = 1000.0 / update_vis_dt; }
                ERL_INFO(
                    "update_map (fps): {:.2f}, update_pred (fps): {:.2f}, update_vis (fps): {:.2f}",
                    update_map_fps,
                    update_pred_fps,
                    update_vis_fps);

                if (options->visualize) {
                    key = cv::waitKey(1);
                    if (key == 27) { break; }  // ESC
                    if (key == 'q') { break; }

                    // check if the window is closed
                    if (cv::getWindowProperty(window_name, cv::WND_PROP_AUTOSIZE) == -1) { break; }
                }
            }

            erl::common::SaveEigenMatrixToTextFile<double>(
                options->output_dir / "fps.csv",
                fps_data,
                erl::common::EigenTextFormat::kCsvFmt);

            if (options->save_video) {
                video_writer->release();
                ERL_INFO("Saved video to {}.", video_path.c_str());
            }

            if (options->num_final_iterations > 0) {
                // clear frame data
                train_frame_points = Matrix2X();
                train_world_points = Matrix2X();
                train_ranges = VectorX();

                long num_iterations = 0;
                while (num_iterations < options->num_final_iterations && update_map()) {
                    ++num_iterations;
                    update_pred();
                    update_vis();
                }
            }

            if (options->test_io) { TestIo(); }
        }

        ShowFinalResults();

        if (options->test_grid_at_end) {
            const auto [grid_shape, grid_points] = GenerateTestGrid();
            const auto file = options->output_dir / "test_grid_shape.txt";
            ERL_INFO("Saving test grid shape to {}", file);
            erl::common::SaveEigenMatrixToTextFile<int>(file, grid_shape);
            if (grid_points.cols() > 0) {
                TestGrid(grid_points);
            } else {
                ERL_WARN("No positions to test the grid at the end.");
            }
        }

        if (options->save_built_mesh) {
            const auto [vertices, faces] = GetBuiltMesh();
            const std::string filepath = options->output_dir / "built_mesh.ply";
            WriteMesh(vertices, faces, filepath);
        }

        if (options->extract_mesh) {
            const auto [vertices, faces] = ExtractMesh();
            const std::string filepath = options->output_dir / "extracted_mesh.ply";
            WriteMesh(vertices, faces, filepath);
        }

        if (options->interactive) {
            Interactive();
        } else {
            if (options->visualize && options->hold) {
                std::cout << "Press any key to exit." << std::endl;
                cv::waitKey(0);
            } else {
                constexpr double wait_time = 10.0;
                cv::waitKey(wait_time * 1000);  // wait for 10 seconds
            }
        }
    }

protected:
    virtual void
    Init() {
        PrepareDataset();
        PrepareOutputFolders();
        PrepareVisualization();
        if (!options->load_mapping_bin) {
            options->AsYamlFile(options->output_dir / "config.yaml");
        }
    }

    virtual bool
    UpdateMap() = 0;

    virtual void
    UpdatePrediction() = 0;

    virtual void
    UpdateVisualization() = 0;

    virtual void
    ShowFinalResults() = 0;

    virtual void
    Interactive() = 0;

#pragma region dataset_prep

    void
    PrepareGazeboRoom2D() {
        // dataset
        gazebo_room_2d = std::make_shared<GazeboRoom2D::TrainDataLoader>(options->gazebo_dir);
        max_wp_idx = gazebo_room_2d->size();
        ERL_ASSERT_LT(options->start_wp_idx, max_wp_idx);
        ERL_ASSERT_POS_LT(options->end_wp_idx, max_wp_idx);
        train_angles = (*gazebo_room_2d)[0].angles.template cast<Dtype>();
        // test data
        map_min = GazeboRoom2D::kMapMin.cast<Dtype>();
        map_max = GazeboRoom2D::kMapMax.cast<Dtype>();
        if (options->test_grid_from_dataset) {
            options->test_grid_def.size = GazeboRoom2D::kOrientedBoundingBoxSize.cast<Dtype>();
            options->test_grid_def.center = GazeboRoom2D::kOrientedBoundingBoxCenter.cast<Dtype>();
            options->test_grid_def.rotation = GazeboRoom2D::kOrientedBoundingBoxRotationAngle;
        }
    }

    void
    PrepareHouseExpoLidar2D() {
        house_expo_map = std::make_shared<HouseExpoMap>(options->house_expo_map_file, 0.2);
        house_expo_traj = erl::common::LoadAndCastCsvFile<Dtype>(
            options->house_expo_traj_file,
            [](const std::string &str) -> double { return std::stod(str); });
        max_wp_idx = static_cast<long>(house_expo_traj.size());
        ERL_ASSERT_LT(options->start_wp_idx, max_wp_idx);
        ERL_ASSERT_POS_LT(options->end_wp_idx, max_wp_idx);
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
        if (options->test_grid_from_dataset) {
            options->test_grid_def.size = map_max - map_min;
            options->test_grid_def.center = (map_min + map_max) * 0.5f;
        }
    }

    void
    PrepareUcsdFah2D() {
        ucsd_fah_2d = std::make_shared<UcsdFah2D>(options->ucsd_fah_2d_file);
        max_wp_idx = ucsd_fah_2d->Size();
        ERL_ASSERT_LT(options->start_wp_idx, max_wp_idx);
        ERL_ASSERT_POS_LT(options->end_wp_idx, max_wp_idx);
        train_angles = (*ucsd_fah_2d)[0].angles.template cast<Dtype>();
        t_span = 500.0f * ucsd_fah_2d->GetTimeStep();
        // test data
        map_min = UcsdFah2D::kMapMin.cast<Dtype>();
        map_max = UcsdFah2D::kMapMax.cast<Dtype>();
        if (options->test_grid_from_dataset) {
            options->test_grid_def.size = map_max - map_min;
            options->test_grid_def.center = (map_min + map_max) * 0.5f;
        }
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

        max_wp_idx = (options->end_wp_idx == -1) ? max_wp_idx : options->end_wp_idx;
    }

#pragma endregion

    virtual void
    PrepareOutputFolders() {
        GTEST_PREPARE_OUTPUT_DIR();
        if (options->output_dir.empty()) { options->output_dir = test_output_dir; }
        std::filesystem::create_directories(options->output_dir);
        if (options->add_datetime_to_output_dir) {
            const auto latest_dir = options->output_dir / "latest";
            options->output_dir /= erl::common::Logging::GetTimeStamp();
            options->add_datetime_to_output_dir = false;  // to avoid adding multiple times
            if (std::filesystem::exists(latest_dir)) { std::filesystem::remove(latest_dir); }
            std::filesystem::create_symlink(options->output_dir.filename(), latest_dir);
        }
        img_dir = options->output_dir / "images";
        video_path = options->output_dir / "mapping.avi";
        std::filesystem::create_directories(img_dir);
        window_name = test_info->name();
    }

    virtual void
    PrepareVisualization() {
        tree_drawer_setting->area_min = map_min.template cast<float>();
        tree_drawer_setting->area_max = map_max.template cast<float>();
        tree_drawer_setting->resolution = options->map_resolution;
        tree_drawer_setting->scaling = scaling;
        tree_drawer_setting->padding = 1;
        tree_drawer_setting->border_color = cv::Scalar(255, 0, 0, 255);

        quadtree_drawer = std::make_shared<QuadtreeDrawer>(tree_drawer_setting, quadtree);
        grid_map_info = quadtree_drawer->GetGridMapInfo()->template CastSharedPtr<Dtype>();

        options->map_resolution = grid_map_info->Resolution(0);
    }

#pragma region data_loading

    void
    LoadDataFromGazeboRoom2D() {
        const auto &frame = (*gazebo_room_2d)[wp_idx];
        rotation = frame.rotation.template cast<Dtype>();
        translation = frame.translation.template cast<Dtype>();
        train_ranges = frame.ranges.template cast<Dtype>();
        traj_t += 0.2;  // assume 5 Hz
    }

    void
    LoadDataFromHouseExpoLidar2D() {
        const std::vector<Dtype> &wp = house_expo_traj[wp_idx];
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
        rotation = rotation_mat.template cast<Dtype>();
        translation = translation_vec.template cast<Dtype>();
        train_ranges = ranges.template cast<Dtype>();
        traj_t += ucsd_fah_2d->GetTimeStep();
    }

    void
    LoadData() {
        const ERL_BLOCK_TIMER_MSG("[App] Data Loading");
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

        if (!mapping_uses_points) { return; }

        if (train_frame_points.cols() != train_ranges.size()) {
            train_frame_points.resize(2, train_ranges.size());
            train_world_points.resize(2, train_ranges.size());
        }
        surface_points_cv.clear();
        surface_points_cv.reserve(train_ranges.size());
        for (long i = 0; i < train_ranges.size(); ++i) {
            // clang-format off
            train_frame_points.col(i) << train_ranges[i] * std::cos(train_angles[i]),
                                         train_ranges[i] * std::sin(train_angles[i]);
            // clang-format on
            train_world_points.col(i) = rotation * train_frame_points.col(i) + translation;

            Eigen::Vector2i px = grid_map_info->MeterToPixelForPoints(train_world_points.col(i));
            surface_points_cv.emplace_back(px[0], px[1]);
        }

        cur_traj.emplace_back(translation);
    }

    [[nodiscard]] long
    GetNumOfFrames() const {
        return (max_wp_idx - options->start_wp_idx + options->seq_stride - 1) / options->seq_stride;
    }

#pragma endregion

#pragma region test_io

    virtual std::string
    GetBinFileName() = 0;

    virtual void
    TestIo() = 0;

    void
    WriteSdfMappingBin() {
        const ERL_BLOCK_TIMER_MSG("[App] WriteMappingBin");
        std::string bin_file = GetBinFileName();
        using namespace erl::common::serialization;
        ERL_ASSERTM(
            Serialization<MappingType>::Write(bin_file, mapping.get()),
            "Failed to write to file: {}",
            bin_file);
    }

    void
    ReadMappingBin(MappingType &mapping_read, std::string bin_file = "") {
        const ERL_BLOCK_TIMER_MSG("[App] ReadMappingBin");
        if (bin_file.empty()) { bin_file = GetBinFileName(); }
        using namespace erl::common::serialization;
        ERL_ASSERTM(
            Serialization<MappingType>::Read(bin_file, &mapping_read),
            "Failed to read from file: {}",
            bin_file);
    }

    void
    TestIo(MappingType &mapping_read) {
        WriteSdfMappingBin();
        ReadMappingBin(mapping_read);
        ERL_ASSERT(*mapping == mapping_read);
    }

#pragma endregion

    std::pair<Eigen::Vector2i, Matrix2X>
    GenerateTestGrid() {
        Vector2 max = options->test_grid_def.size.array() * 0.5f;
        Vector2 min = -max;
        Eigen::Vector2i grid_shape;
        const Dtype res = options->test_res_grid;
        grid_shape[0] = static_cast<int>(std::ceil(options->test_grid_def.size[0] / res));
        grid_shape[1] = static_cast<int>(std::ceil(options->test_grid_def.size[1] / res));
        const Vector2 resolution = Vector2::Constant(res);
        max = min.array() + grid_shape.cast<Dtype>().array() * resolution.array();
        ERL_INFO(
            "Grid size: [{}], resolution: [{}], shape: [{}], min: [{}], max: [{}]",
            options->test_grid_def.size.transpose(),
            resolution.transpose(),
            grid_shape.transpose(),
            min.transpose(),
            max.transpose());
        constexpr bool row_major = false;
        constexpr bool grid_coords = true;
        Matrix2X positions =
            erl::common::CalculateMeterCoordinates<Dtype, int, 2, row_major, grid_coords>(
                grid_shape,
                min,
                max,
                resolution);
        ERL_INFO(
            "Grid rotation: {:.2f} rad, center: [{}]",
            options->test_grid_def.rotation,
            options->test_grid_def.center.transpose());
        ERL_INFO(
            "Before transform, positions min: [{}], max: [{}]",
            positions.rowwise().minCoeff().transpose(),
            positions.rowwise().maxCoeff().transpose());
        Matrix2 rot = Eigen::Rotation2D<Dtype>(options->test_grid_def.rotation).toRotationMatrix();
        positions = (rot * positions).colwise() + options->test_grid_def.center;
        ERL_INFO(
            "After transform, positions min: [{}], max: [{}]",
            positions.rowwise().minCoeff().transpose(),
            positions.rowwise().maxCoeff().transpose());
        return {grid_shape, positions};
    }

    virtual void
    TestGrid(const Matrix2X & /*grid_positions*/) = 0;

    virtual std::pair<std::vector<Vector2>, std::vector<Eigen::Vector2i>>
    GetBuiltMesh() = 0;

    virtual std::pair<std::vector<Vector2>, std::vector<Eigen::Vector2i>>
    ExtractMesh() = 0;

    static void
    WriteMesh(
        const std::vector<Vector2> &vertices,
        const std::vector<Eigen::Vector2i> &faces,
        const std::string &filepath) {

        open3d::geometry::LineSet line_set;
        line_set.points_.reserve(vertices.size());
        for (const auto &v: vertices) { line_set.points_.emplace_back(v[0], v[1], 0.0); }
        line_set.lines_ = faces;

        open3d::io::WriteLineSetToPLY(filepath, line_set);
    }

    cv::Mat
    VisualizeMesh(
        const std::vector<Vector2> &vertices,
        const std::vector<Eigen::Vector2i> &faces,
        const cv::Mat &img_init) {

        Eigen::Matrix2Xi lines_to_vertices = Eigen::Map<const Eigen::Matrix2Xi>(
            faces.data()->data(),
            2,
            static_cast<long>(faces.size()));
        Eigen::Matrix2Xi objects_to_lines;
        erl::geometry::MarchingSquares::SortLinesToObjects(lines_to_vertices, objects_to_lines);

        std::vector<std::vector<cv::Point2i>> contours(objects_to_lines.cols());
        for (long i = 0; i < objects_to_lines.cols(); ++i) {
            const Eigen::Vector2i &lines = objects_to_lines.col(i);
            std::vector<cv::Point2i> &contour = contours[i];
            contour.reserve(lines[1] - lines[0]);
            for (long j = lines[0]; j < lines[1]; ++j) {
                const int k = lines_to_vertices(0, j);
                const Vector2 &v = vertices[k];
                Eigen::Vector2i pt = grid_map_info->MeterToPixelForPoint(v);
                contour.emplace_back(pt[0], pt[1]);
            }
        }
        cv::Mat mat = img_init.clone();
        cv::polylines(mat, contours, false, cv::Scalar(255, 255, 255), 2);
        return mat;
    }
};
