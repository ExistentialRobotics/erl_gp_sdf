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
struct OptionsForTestMapping2D : public erl::common::Yamlable<OptionsForTestMapping2D<Dtype>> {
    inline static const std::filesystem::path kProjectRootDir = ERL_GP_SDF_ROOT_DIR;
    inline static const std::filesystem::path kDataDir = kProjectRootDir / "data";
    inline static const std::filesystem::path kConfigDir = kProjectRootDir / "config";

    DataSetType dataset_type = DataSetType::GazeboRoom2D;
    std::string gazebo_dir = kDataDir / "gazebo";
    std::string house_expo_map_file = kDataDir / "house_expo_room_1451.json";
    std::string house_expo_traj_file = kDataDir / "house_expo_room_1451.csv";
    std::string ucsd_fah_2d_file = kDataDir / "ucsd_fah_2d.dat";
    std::string mapping_bin_file;
    bool load_mapping_bin = false;
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
    Dtype map_resolution = 0.025;
    Dtype surf_normal_scale = 0.35;

    ERL_REFLECT_SCHEMA(
        OptionsForTestMapping2D,
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, dataset_type),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, gazebo_dir),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, house_expo_map_file),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, house_expo_traj_file),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, ucsd_fah_2d_file),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, mapping_bin_file),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, load_mapping_bin),
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
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, map_resolution),
        ERL_REFLECT_MEMBER(OptionsForTestMapping2D, surf_normal_scale));

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
    Vector2 map_min, map_max;
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

    std::filesystem::path test_output_folder;
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
        options->FromCommandLine(argc, argv);
    }

    virtual ~TestMapping2D() = default;

    virtual void
    Run() {
        Init();

        if (options->load_mapping_bin) {
            ReadSdfMappingBin(*mapping);
        } else {
            if (options->test_io) { TestIo(); }

            auto update_map = [this]() {
                ERL_BLOCK_TIMER_MSG_TIME("UpdateMap", this->update_map_dt);
                return this->UpdateMap();
            };

            auto update_pred = [this]() {
                ERL_BLOCK_TIMER_MSG_TIME("UpdatePrediction", this->update_pred_dt);
                this->UpdatePrediction();
            };

            auto update_vis = [this]() {
                if (!options->visualize) { return; }
                ERL_BLOCK_TIMER_MSG_TIME("UpdateVisualization", this->update_vis_dt);
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
                test_output_folder / "fps.csv",
                fps_data,
                erl::common::EigenTextFormat::kCsvFmt);

            if (options->save_video) {
                video_writer->release();
                ERL_INFO("Saved video to {}.", video_path.c_str());
            }

            if (options->test_io) { TestIo(); }
        }

        ShowFinalResults();
        if (options->interactive) { Interactive(); }

        if (options->visualize && options->hold) {
            std::cout << "Press any key to exit." << std::endl;
            cv::waitKey(0);
        } else {
            constexpr double wait_time = 10.0;
            cv::waitKey(wait_time * 1000);  // wait for 10 seconds
        }
    }

protected:
    virtual void
    Init() {
        PrepareDataset();
        PrepareOutputFolders();
        PrepareVisualization();
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
        test_output_folder = test_output_dir;
        img_dir = test_output_folder / "images";
        video_path = test_output_folder / "mapping.avi";
        std::filesystem::create_directory(img_dir);
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

#pragma endregion

#pragma region test_io

    virtual std::string
    GetBinFileName() = 0;

    virtual void
    TestIo() = 0;

    void
    WriteSdfMappingBin() {
        ERL_BLOCK_TIMER_MSG("WriteMappingBin");
        std::string bin_file = GetBinFileName();
        using namespace erl::common::serialization;
        ERL_ASSERTM(
            Serialization<MappingType>::Write(bin_file, mapping.get()),
            "Failed to write to file: {}",
            bin_file);
    }

    void
    ReadSdfMappingBin(MappingType &mapping_read) {
        ERL_BLOCK_TIMER_MSG("ReadMappingBin");
        std::string bin_file = GetBinFileName();
        using namespace erl::common::serialization;
        ERL_ASSERTM(
            Serialization<MappingType>::Read(bin_file, &mapping_read),
            "Failed to read from file: {}",
            bin_file);
    }

    void
    TestIo(MappingType &mapping_read) {
        ERL_BLOCK_TIMER_MSG("IO");
        WriteSdfMappingBin();
        ReadSdfMappingBin(mapping_read);
        ERL_ASSERT(*mapping == mapping_read);
    }

#pragma endregion
};
