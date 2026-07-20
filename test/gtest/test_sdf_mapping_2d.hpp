#pragma once

#include "test_mapping_2d.hpp"

#include "erl_gp_sdf/gp_sdf_mapping.hpp"

inline cv::Point
EigenToOpenCV(const Eigen::Vector2i &p) {
    return {p.x(), p.y()};
}

template<typename Dtype>
void
DrawSdf(cv::Mat &img, const int x, const int y, Dtype sdf, Dtype resolution) {
    const auto radius = static_cast<int>(std::abs(sdf) / resolution);
    if (radius < 0) { return; }
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
    if (radius < 0) { return; }
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

template<typename SurfMap, typename SdfMapping, typename Drawer>
struct OpenCvUserData {
    std::string window_name;
    SurfMap *surf_map = nullptr;
    SdfMapping *sdf_map = nullptr;
    Drawer *drawer = nullptr;
    cv::Mat img;
};

template<typename Dtype, typename SurfMap, typename SdfMapping, typename Drawer>
void
OpenCvMouseCallback(const int event, const int x, const int y, int /*flags*/, void *userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        auto *data = static_cast<OpenCvUserData<SurfMap, SdfMapping, Drawer> *>(userdata);
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
            (void) data->surf_map->CollectSurfaceDataInAabb(aabb, distances_indices);
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

template<typename Dtype>
struct Options : public erl::common::Yamlable<Options<Dtype>, OptionsForTestMapping2D<Dtype>> {
    using Super = OptionsForTestMapping2D<Dtype>;

    std::string surf_map_config_file;
    std::string sdf_map_config_file;
    long plplot_width = 1280;
    long plplot_height = 480;

    ERL_REFLECT_SCHEMA(
        Options,
        ERL_REFLECT_MEMBER(Options, surf_map_config_file),
        ERL_REFLECT_MEMBER(Options, sdf_map_config_file),
        ERL_REFLECT_MEMBER(Options, plplot_width),
        ERL_REFLECT_MEMBER(Options, plplot_height));

    bool
    PostDeserialization() override {
        if (!Super::PostDeserialization()) { return false; }
        ERL_ASSERTM(
            !surf_map_config_file.empty(),
            "Please provide the surface mapping config file via --surf_map_config_file");
        ERL_ASSERTM(
            !sdf_map_config_file.empty(),
            "Please provide the SDF mapping config file via --sdf_map_config_file");
        return true;
    }
};

template<typename Dtype, typename SurfMapType>
struct TestSdfMapping2D : public TestMapping2D<Dtype, erl::gp_sdf::GpSdfMapping<Dtype, 2>> {
    using OptionType = Options<Dtype>;
private:
    std::shared_ptr<OptionType> options = nullptr;

public:
    using Super = TestMapping2D<Dtype, erl::gp_sdf::GpSdfMapping<Dtype, 2>>;
    using SurfMap = SurfMapType;
    using SdfMap = erl::gp_sdf::GpSdfMapping<Dtype, 2>;
    using SurfMapSetting = typename SurfMap::Setting;
    using SdfMapSetting = typename SdfMap::Setting;
    using PlplotFig = erl::common::PlplotFig;

    using Matrix3 = Eigen::Matrix3<Dtype>;
    using MatrixX = Eigen::MatrixX<Dtype>;
    using Matrix2X = Eigen::Matrix2X<Dtype>;
    using Matrix3X = Eigen::Matrix3X<Dtype>;

    using typename Super::QuadtreeDrawer;
    using typename Super::Vector2;
    using typename Super::VectorX;

    using Super::grid_map_info;
    using Super::img_canvas;
    using Super::img_dir;
    using Super::map_max;
    using Super::map_min;
    using Super::map_rotation;
    using Super::map_translation;
    using Super::mapping_uses_points;
    using Super::max_wp_idx;
    using Super::quadtree;
    using Super::quadtree_drawer;
    using Super::rotation;
    using Super::scaling;
    using Super::t_span;
    using Super::train_ranges;
    using Super::train_world_points;
    using Super::traj_t;
    using Super::translation;
    using Super::update_pred_fps;
    using Super::update_vis_fps;
    using Super::window_name;
    using Super::wp_idx;

    std::shared_ptr<SurfMapSetting> surf_map_setting = nullptr;
    std::shared_ptr<SdfMapSetting> sdf_map_setting = nullptr;
    std::shared_ptr<SurfMap> surf_map = nullptr;
    std::shared_ptr<SdfMap> sdf_map = nullptr;

    // visualization

    PlplotFig fig_sdf{1280, 480, true};
    PlplotFig fig_grad{1280, 480, true};
    PlplotFig::LegendOpt legend_opt_sdf{3, {"SDF", "EDF", "Variance"}};
    PlplotFig::LegendOpt legend_opt_grad{4, {"grad_x", "grad_y", "var_grad_x", "var_grad_y"}};
    double t_ms = 0;
    std::vector<double> timestamps_sec;
    std::vector<double> sdf_values;
    std::vector<double> edf_values;
    std::vector<double> var_sdf_values;
    std::vector<double> grad_x_values;
    std::vector<double> grad_y_values;
    std::vector<double> var_grad_x_values;
    std::vector<double> var_grad_y_values;
    cv::Mat img_scene;
    cv::Mat img_final;
    cv::Scalar color_trajectory{0, 0, 0, 255};
    cv::Scalar color_surf_point{0, 255, 125, 255};
    cv::Scalar color_normal_vec{0, 0, 255, 255};
    cv::Scalar color_text{0, 255, 0, 255};
    Matrix2X grid_points;
    bool surf_map_supports_mesh = false;

    // test data
    VectorX sdf_pred_follow{1};
    Matrix2X sdf_gradient_pred_follow{2, 1};
    Matrix3X sdf_var_pred_follow{3, 1};
    Matrix3X sdf_covariances_pred_follow{3, 1};

    VectorX sdf_pred_whole_map;
    Matrix2X sdf_gradient_pred_whole_map;
    Matrix3X sdf_var_pred_whole_map;
    Matrix3X sdf_covariances_pred_whole_map;

    // logging

    bool surf_map_updated = false;
    bool sdf_map_updated = false;
    bool test_success = false;
    Eigen::Matrix4Xd fps_data;
    double surf_map_update_dt = 0;
    double sdf_map_update_dt = 0;
    double surf_map_update_fps = 0;
    double sdf_map_update_fps = 0;

public:
    TestSdfMapping2D(
        const int argc,
        char *argv[],
        std::shared_ptr<OptionType> options = std::make_shared<OptionType>())
        : Super(argc, argv, options),
          options(options),
          fig_sdf(options->plplot_width, options->plplot_height, true),
          fig_grad(options->plplot_width, options->plplot_height, true) {}

protected:
    // initialization
    void
    Init() override {
        surf_map_setting = std::make_shared<SurfMapSetting>();
        ERL_ASSERTM(
            surf_map_setting->FromYamlFile(options->surf_map_config_file),
            "Failed to load surf_map_config_file: {}",
            options->surf_map_config_file);

        sdf_map_setting = std::make_shared<SdfMapSetting>();
        ERL_ASSERTM(
            sdf_map_setting->FromYamlFile(options->sdf_map_config_file),
            "Failed to load sdf_map_config_file: {}",
            options->sdf_map_config_file);

        // create mappings
        surf_map = std::make_shared<SurfMap>(surf_map_setting);
        sdf_map = std::make_shared<SdfMap>(sdf_map_setting, surf_map);
        quadtree = surf_map->GetTree();
        Super::mapping = sdf_map;

        // cluster size
        scaling = surf_map_setting->scaling;

        // base init
        Super::Init();

        // save configs
        if (!options->load_mapping_bin) {
            std::filesystem::create_directories(options->output_dir);
            surf_map_setting->AsYamlFile(options->output_dir / "surf_map.yaml");
            sdf_map_setting->AsYamlFile(options->output_dir / "sdf_map.yaml");

            ERL_INFO("Surface mapping config: {}", options->surf_map_config_file);
            std::cout << surf_map_setting->AsYamlString() << std::endl;

            ERL_INFO("SDF mapping config: {}", options->sdf_map_config_file);
            std::cout << sdf_map_setting->AsYamlString() << std::endl;
        }

        // other
        sdf_pred_whole_map.resize(grid_points.cols());
        sdf_gradient_pred_whole_map.resize(2, grid_points.cols());
        sdf_var_pred_whole_map.resize(3, grid_points.cols());
        sdf_covariances_pred_whole_map.resize(3, grid_points.cols());

        try {
            std::vector<Vector2> vertices;
            std::vector<Eigen::Vector2i> faces;
            surf_map_supports_mesh = surf_map->GetMesh(true, vertices, faces);
        } catch (std::exception &e) {
            ERL_WARN("Surface mapping does not support mesh extraction: {}", e.what());
            surf_map_supports_mesh = false;
        }
    }

    void
    PrepareVisualization() override {
        Super::PrepareVisualization();

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

        grid_points = grid_map_info->GenerateMeterCoordinates(false);
    }

    bool
    UpdateSurfaceMap() {
        if (mapping_uses_points) {
            ERL_BLOCK_TIMER_MSG_TIME("[App] SurfMap.Update", surf_map_update_dt);
            // are_points: true, are_local: false
            return surf_map->Update(rotation, translation, train_world_points, true, false);
        }

        {
            ERL_BLOCK_TIMER_MSG_TIME("[App] SurfMap.Update", surf_map_update_dt);
            // are_points: false, are_local: true
            return surf_map->Update(rotation, translation, train_ranges, false, true);
        }
    }

    bool
    UpdateSdfMap() {
        ERL_BLOCK_TIMER_MSG_TIME("sdf_map.Update", sdf_map_update_dt);

        surf_map_updated = UpdateSurfaceMap();

        const double time_budget_us = 1e6 / sdf_map_setting->update_hz;  // us
        sdf_map_updated = sdf_map->UpdateGpSdf(time_budget_us - surf_map_update_dt * 1000);

        return surf_map_updated || sdf_map_updated;
    }

    bool
    UpdateMap() override {
        return UpdateSdfMap();
    }

    void
    UpdatePrediction() override {
        test_success = sdf_map->Test(
            translation,
            sdf_pred_follow,
            sdf_gradient_pred_follow,
            sdf_var_pred_follow,
            sdf_covariances_pred_follow);
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
                it->normal * options->surf_normal_scale);
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

        InitSceneImg();

        // Visualize the results
        cv::Point position_px = EigenToOpenCV(grid_map_info->MeterToPixelForPoints(translation));
        DrawSdf<Dtype>(
            img_scene,
            position_px.x,
            position_px.y,
            sdf_pred_follow[0],
            options->map_resolution);
        DrawSdfVariance<Dtype>(
            img_scene,
            position_px.x,
            position_px.y,
            sdf_pred_follow[0],
            sdf_var_pred_follow(0, 0),
            options->map_resolution);
        DrawSdfGradient(
            img_scene,
            *quadtree_drawer,
            position_px.x,
            position_px.y,
            Eigen::Vector2<Dtype>(sdf_gradient_pred_follow.col(0)));

        // draw used surface points
        if (test_success) {
            auto &gps = CHECKED_AT(sdf_map->GetUsedGps(), 0);
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

        // draw house_expo_traj
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
            fmt::format("SurfMap.Update: {:.2f} fps", surf_map_update_fps),
            cv::Point(10, 15),
            kFontFace,
            kFontScale,
            color_text,
            kThickness);
        cv::putText(
            img_scene,
            fmt::format("SdfMap.Update: {:.2f} fps", sdf_map_update_fps),
            cv::Point(10, 30),
            kFontFace,
            kFontScale,
            color_text,
            kThickness);
        cv::putText(
            img_scene,
            fmt::format("SdfMap.Test: {:.2f} fps", update_pred_fps),
            cv::Point(10, 45),
            kFontFace,
            kFontScale,
            color_text,
            kThickness);
        cv::putText(
            img_scene,
            fmt::format("GUI: {:.2f} fps", update_vis_fps),
            cv::Point(10, 60),
            kFontFace,
            kFontScale,
            color_text,
            kThickness);
    }

    void
    UpdateCurves() {
        timestamps_sec.push_back(traj_t);
        sdf_values.push_back(sdf_pred_follow[0]);
        edf_values.push_back(std::abs(sdf_pred_follow[0]));
        var_sdf_values.push_back(sdf_var_pred_follow(0, 0));
        grad_x_values.push_back(sdf_gradient_pred_follow(0, 0));
        grad_y_values.push_back(sdf_gradient_pred_follow(1, 0));
        var_grad_x_values.push_back(sdf_var_pred_follow(1, 0));
        var_grad_y_values.push_back(sdf_var_pred_follow(2, 0));
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
                fmt::format(
                    "sdf: {:.2f}, var_sdf: {:.2e}",
                    sdf_pred_follow[0],
                    sdf_var_pred_follow(0, 0))
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
                    sdf_gradient_pred_follow(0, 0),
                    sdf_gradient_pred_follow(1, 0),
                    sdf_var_pred_follow(1, 0),
                    sdf_var_pred_follow(2, 0))
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
        const int fig_col_width = fig_sdf.Width();
        const int fig_col_height = fig_sdf.Height() + fig_grad.Height();
        cv::Mat tmp(
            std::max(img_scene.rows, fig_col_height),
            img_scene.cols + fig_col_width,
            CV_8UC4,
            cv::Scalar(255, 255, 255, 255));
        if (img_scene.rows == tmp.rows) {
            const int offset = (tmp.rows - fig_col_height) / 2;
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
        cv::imshow(window_name, img_canvas);
        cv::waitKey(1);
    }

    void
    UpdateVisualization() override {
        UpdateSceneImg();
        UpdateCurves();
        UpdateCanvas();
    }

    void
    ShowFinalResults() override {
        if (!options->visualize) { return; }

        const auto t0 = std::chrono::high_resolution_clock::now();
        const bool success = sdf_map->Test(
            grid_points,
            sdf_pred_whole_map,
            sdf_gradient_pred_whole_map,
            sdf_var_pred_whole_map,
            sdf_covariances_pred_whole_map);
        const auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double t_per_point = dt / static_cast<double>(grid_points.cols()) * 1000;  // us
        ERL_ASSERTM(success, "Final SDF test failed.");
        ERL_INFO(
            "Test time: {:f} ms for {} points, {:f} us per point.",
            dt,
            grid_points.cols(),
            t_per_point);
        Dtype min_sdf = sdf_pred_whole_map.minCoeff();
        Dtype max_sdf = sdf_pred_whole_map.maxCoeff();
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
        cv::imshow("GPs", img_gp);
        cv::imwrite(img_dir / "gps.png", img_gp);

        // draw SDF map
        cv::Mat img_sdf(
            grid_map_info->Height(),
            grid_map_info->Width(),
            sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1,
            sdf_pred_whole_map.data());
        cv::normalize(img_sdf, img_sdf, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::flip(img_sdf, img_sdf, 0);
        cv::applyColorMap(img_sdf, img_sdf, cv::COLORMAP_JET);
        img_sdf.copyTo(img_final);
        cv::cvtColor(img_sdf, img_sdf, cv::COLOR_BGR2BGRA);
        cv::addWeighted(img_sdf, 0.5, img_scene, 0.5, 0.0, img_sdf);
        cv::imshow("sdf", img_sdf);
        cv::imwrite(img_dir / "sdf.png", img_sdf);

        // convert to binary image: 0 for negative, 255 for positive
        Eigen::VectorXi sdf_sign = (sdf_pred_whole_map.array() >= 0).template cast<int>();
        cv::Mat img_sign(
            grid_map_info->Height(),
            grid_map_info->Width(),
            CV_32SC1,
            sdf_sign.data());
        cv::normalize(img_sign, img_sign, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::flip(img_sign, img_sign, 0);  // flip along y axis
        cv::imshow("sdf sign", img_sign);
        cv::imwrite(img_dir / "sdf_sign.png", img_sign);

        // draw SDF variance map
        VectorX sdf_variances = sdf_var_pred_whole_map.row(0).transpose();
        cv::Mat img_sdf_variance(
            grid_map_info->Height(),
            grid_map_info->Width(),
            sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1,
            sdf_variances.data());
        cv::normalize(img_sdf_variance, img_sdf_variance, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::flip(img_sdf_variance, img_sdf_variance, 0);
        cv::applyColorMap(img_sdf_variance, img_sdf_variance, cv::COLORMAP_JET);
        cv::cvtColor(img_sdf_variance, img_sdf_variance, cv::COLOR_BGR2BGRA);
        cv::addWeighted(img_sdf_variance, 0.5, img_scene, 0.5, 0.0, img_sdf_variance);
        cv::imshow("sdf variance", img_sdf_variance);
        cv::imwrite(img_dir / "sdf_variance.png", img_sdf_variance);

        // show
        cv::waitKey(1);
    }

    void
    Interactive() override {
        if (!options->interactive) { return; }

        InitSceneImg();

        OpenCvUserData<SurfMap, SdfMap, QuadtreeDrawer> data;
        data.window_name = "interactive";
        data.img = img_scene;
        data.drawer = quadtree_drawer.get();
        data.surf_map = surf_map.get();
        data.sdf_map = sdf_map.get();

        cv::imshow(data.window_name, img_scene);
        cv::setMouseCallback(
            data.window_name,
            OpenCvMouseCallback<Dtype, SurfMap, SdfMap, QuadtreeDrawer>,
            &data);
        while (cv::waitKey(0) != 27 && cv::waitKey(0) != 'q') {}  // wait for ESC/q key
    }

    std::string
    GetBinFileName() override {
        std::string bin_file = fmt::format("sdf_mapping_2d_{}.bin", type_name<Dtype>());
        bin_file = options->output_dir / bin_file;
        return bin_file;
    }

    void
    TestIo() override {
        auto surf_map_read = std::make_shared<SurfMap>(std::make_shared<SurfMapSetting>());
        SdfMap sdf_map_read(std::make_shared<SdfMapSetting>(), surf_map_read);
        Super::TestIo(sdf_map_read);
    }

    void
    TestGrid(const Matrix2X &points) override {
        VectorX pred_sdf;
        Matrix2X pred_grads;
        Matrix3X pred_vars;
        Matrix3X pred_covars;
        {
            const ERL_BLOCK_TIMER_MSG("sdf_map.Test grid");
            ERL_ASSERT(sdf_map->Test(points, pred_sdf, pred_grads, pred_vars, pred_covars));
        }

        std::filesystem::path file = options->output_dir / "test_grid_points.bin";
        ERL_INFO("Saving test grid points to {}", file.string());
        ERL_ASSERT(erl::common::SaveEigenMatrixToBinaryFile<Dtype>(file, points));

        file = options->output_dir / "test_grid_sdf.bin";
        ERL_INFO("Saving test grid sdf to {}", file.string());
        ERL_ASSERT(erl::common::SaveEigenMatrixToBinaryFile<Dtype>(file, pred_sdf));

        Eigen::Vector2i grid_shape;
        const Dtype res = options->test_res_grid;
        grid_shape[0] = static_cast<int>(std::ceil(options->test_grid_def.size[0] / res));
        grid_shape[1] = static_cast<int>(std::ceil(options->test_grid_def.size[1] / res));

        cv::Mat img_sdf(
            grid_shape[1],  // height
            grid_shape[0],  // width
            sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1,
            pred_sdf.data());
        cv::flip(img_sdf, img_sdf, 0);
        cv::normalize(img_sdf, img_sdf, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(img_sdf, img_sdf, cv::COLORMAP_JET);
        cv::imwrite(img_dir / "test_grid_sdf.png", img_sdf);
        if (options->visualize) {
            cv::imshow("test grid sdf", img_sdf);
            cv::waitKey(1);
        }

        if (pred_grads.cols() > 0) {
            file = options->output_dir / "test_grid_gradients.bin";
            ERL_INFO("Saving test grid gradients to {}", file.string());
            ERL_ASSERT(erl::common::SaveEigenMatrixToBinaryFile<Dtype>(file, pred_grads));

            cv::Mat img_grad(
                grid_shape[1],
                grid_shape[0],
                sizeof(Dtype) == 4 ? CV_32FC2 : CV_64FC2,
                pred_grads.data());
            cv::flip(img_grad, img_grad, 0);
            img_grad *= 255;  // scale to [0, 255]
            std::vector<cv::Mat> grad_channels(2);
            cv::split(img_grad, grad_channels);
            grad_channels[0].convertTo(grad_channels[0], CV_8UC1);
            grad_channels[1].convertTo(grad_channels[1], CV_8UC1);
            cv::imwrite(img_dir / "test_grid_grad_x.png", grad_channels[0]);
            cv::imwrite(img_dir / "test_grid_grad_y.png", grad_channels[1]);
            grad_channels.push_back(cv::Mat::zeros(img_grad.size(), CV_8UC1));
            cv::Mat img_grad_color;
            cv::merge(grad_channels, img_grad_color);
            cv::imwrite(img_dir / "test_grid_grad_xy.png", img_grad_color);
            if (options->visualize) {
                cv::imshow("test grid grad x", grad_channels[0]);
                cv::imshow("test grid grad y", grad_channels[1]);
                cv::imshow("test grid grad xy", img_grad_color);
                cv::waitKey(1);
            }
        }

        if (pred_vars.cols() > 0) {
            file = options->output_dir / "test_grid_variances.bin";
            ERL_INFO("Saving test grid variances to {}", file.string());
            ERL_ASSERT(erl::common::SaveEigenMatrixToBinaryFile<Dtype>(file, pred_vars));

            ERL_INFO(
                "max variance sdf: {}, grad_x: {}, grad_y: {}",
                pred_vars.row(0).maxCoeff(),
                pred_vars.row(1).maxCoeff(),
                pred_vars.row(2).maxCoeff());

            cv::Mat img_var(
                grid_shape[1],  // height
                grid_shape[0],  // width
                sizeof(Dtype) == 4 ? CV_32FC3 : CV_64FC3,
                pred_vars.data());
            cv::flip(img_var, img_var, 0);
            std::vector<cv::Mat> var_channels(3);
            cv::split(img_var, var_channels);
            // visualize var(sdf)
            cv::normalize(var_channels[0], var_channels[0], 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::applyColorMap(var_channels[0], var_channels[0], cv::COLORMAP_JET);
            cv::imwrite(img_dir / "test_grid_var_sdf.png", var_channels[0]);
            if (options->visualize) {
                cv::imshow("test grid var sdf", var_channels[0]);
                cv::waitKey(1);
            }
        }

        if (pred_covars.cols() > 0) {
            file = options->output_dir / "test_grid_covariances.bin";
            ERL_INFO("Saving test grid covariances to {}", file.string());
            ERL_ASSERT(erl::common::SaveEigenMatrixToBinaryFile<Dtype>(file, pred_covars));
        }
    }

    std::pair<std::vector<Vector2>, std::vector<Eigen::Vector2i>>
    GetBuiltMesh() override {
        std::pair<std::vector<Vector2>, std::vector<Eigen::Vector2i>> mesh_data;
        if (!surf_map_supports_mesh) { return mesh_data; }
        surf_map->GetMesh(false, mesh_data.first, mesh_data.second);

        const cv::Mat img_mesh = this->VisualizeMesh(mesh_data.first, mesh_data.second, img_final);
        const std::string filepath = img_dir / "built_mesh.png";
        cv::imwrite(filepath, img_mesh);

        if (options->visualize) {
            cv::imshow("built_mesh", img_mesh);
            cv::waitKey(1);
        }

        return mesh_data;
    }

    std::pair<std::vector<Vector2>, std::vector<Eigen::Vector2i>>
    ExtractMesh() override {
        std::pair<std::vector<Vector2>, std::vector<Eigen::Vector2i>> mesh_data;
        if (surf_map_supports_mesh) {
            surf_map->GetMesh(options->extract_mesh_res, mesh_data.first, mesh_data.second);
        } else {
            std::vector<Vector2> face_normals;
            sdf_map->GetMesh(
                map_max - map_min,
                map_rotation,
                map_translation,
                options->extract_mesh_res,
                0.0f,
                mesh_data.first,
                mesh_data.second,
                face_normals);
        }

        const cv::Mat img_mesh = this->VisualizeMesh(mesh_data.first, mesh_data.second, img_final);
        const std::string filepath = img_dir / "extracted_mesh.png";
        cv::imwrite(filepath, img_mesh);

        if (options->visualize) {
            cv::imshow("extracted_mesh", img_mesh);
            cv::waitKey(1);
        }

        return mesh_data;
    }
};
