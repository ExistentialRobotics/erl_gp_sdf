#include "erl_common/plplot_fig.hpp"
#include "erl_gaussian_process/lidar_gp_2d.hpp"
#include "erl_geometry/gazebo_room_2d.hpp"

using namespace erl::common;
using namespace erl::gaussian_process;
using namespace erl::geometry;
using Gp = LidarGaussianProcess2Dd;

struct Options : Yamlable<Options> {
    int frame_idx = 0;
    double margin = 0.2;
    double perturb_delta = 0.01;
    int test_num_x = 201;
    int test_num_y = 201;
    std::shared_ptr<Gp::Setting> gp = std::make_shared<Gp::Setting>();
    std::string output_dir = "gp_occ_regression";
};

template<>
struct YAML::convert<Options> {
    static Node
    encode(const Options &options) {
        Node node;
        ERL_YAML_SAVE_ATTR(node, options, frame_idx);
        ERL_YAML_SAVE_ATTR(node, options, margin);
        ERL_YAML_SAVE_ATTR(node, options, perturb_delta);
        ERL_YAML_SAVE_ATTR(node, options, test_num_x);
        ERL_YAML_SAVE_ATTR(node, options, test_num_y);
        ERL_YAML_SAVE_ATTR(node, options, gp);
        ERL_YAML_SAVE_ATTR(node, options, output_dir);
        return node;
    }

    static bool
    decode(const Node &node, Options &options) {
        if (!node.IsMap()) { return false; }
        ERL_YAML_LOAD_ATTR(node, options, frame_idx);
        ERL_YAML_LOAD_ATTR(node, options, margin);
        ERL_YAML_LOAD_ATTR(node, options, perturb_delta);
        ERL_YAML_LOAD_ATTR(node, options, test_num_x);
        ERL_YAML_LOAD_ATTR(node, options, test_num_y);
        ERL_YAML_LOAD_ATTR(node, options, gp);
        ERL_YAML_LOAD_ATTR(node, options, output_dir);
        return true;
    }
};

struct App {
    Options options;

    explicit App(const std::string &option_file) {
        if (!std::filesystem::exists(option_file)) { options.AsYamlFile(option_file); }
        if (!option_file.empty()) {
            ERL_ASSERTM(
                options.FromYamlFile(option_file),
                "Failed to load options from file: {}",
                option_file);
        }
    }

    void
    Run() {
        GazeboRoom2D::TrainDataLoader dataloader(ERL_SDF_MAPPING_ROOT_DIR "/data/gazebo");
        auto &frame = dataloader[options.frame_idx];
        Gp gp(options.gp);
        gp.Reset();
        ERL_ASSERTM(
            gp.Train(frame.rotation, frame.translation, frame.ranges),
            "Failed to train GP.");

        auto sensor_frame = gp.GetSensorFrame();
        Eigen::Map<const Eigen::Matrix2Xd> pts_world(
            sensor_frame->GetEndPointsInWorld().data()->data(),
            2,
            sensor_frame->GetNumHitRays());

        Eigen::Vector2d bbx_min = pts_world.rowwise().minCoeff();
        Eigen::Vector2d bbx_max = pts_world.rowwise().maxCoeff();

        GridMapInfo2Dd grid_info(
            Eigen::Vector2i(options.test_num_x, options.test_num_y),
            bbx_min.array() - options.margin,
            bbx_max.array() + options.margin);
        Eigen::Matrix2Xd test_pts = grid_info.GenerateMeterCoordinates(true /*c_stride*/);
        Eigen::VectorXd occ_values(test_pts.cols());

        auto compute_occ = [&gp](const double x, const double y, double &occ) -> bool {
            Eigen::Vector2d p_local = gp.GetSensorFrame()->PosWorldToFrame(Eigen::Vector2d(x, y));
            Eigen::Scalard angle_local;
            angle_local[0] = std::atan2(p_local.y(), p_local.x());
            const double r = p_local.norm();
            double range_pred;
            const bool valid = gp.ComputeOcc(angle_local, r, range_pred, occ);
            if (!valid) { occ = 0.0; }
            return valid;
        };

#pragma omp parallel for default(none) shared(occ_values, sensor_frame, test_pts, compute_occ)
        for (long i = 0; i < occ_values.size(); ++i) {
            auto pt = test_pts.col(i);
            compute_occ(pt.x(), pt.y(), occ_values[i]);
        }

        constexpr long stride = 1;
        Eigen::VectorXd x(pts_world.cols() / stride);
        Eigen::VectorXd y(x.size());
        Eigen::VectorXd nx(x.size());
        Eigen::VectorXd ny(x.size());
#pragma omp parallel for default(none) shared(pts_world, x, y, nx, ny, compute_occ, options)
        for (long i = 0; i < pts_world.cols(); i += stride) {
            long j = i / stride;
            x[j] = pts_world(0, i);
            y[j] = pts_world(1, i);
            nx[j] = 0.0;
            ny[j] = 0.0;

            double occ1, occ2, occ3, occ4;
            if (!compute_occ(x[j] + options.perturb_delta, y[j], occ1)) { continue; }
            if (!compute_occ(x[j] - options.perturb_delta, y[j], occ2)) { continue; }
            if (!compute_occ(x[j], y[j] + options.perturb_delta, occ3)) { continue; }
            if (!compute_occ(x[j], y[j] - options.perturb_delta, occ4)) { continue; }

            nx[j] = (occ1 - occ2) / (2 * options.perturb_delta);
            ny[j] = (occ3 - occ4) / (2 * options.perturb_delta);
            const double norm = std::sqrt(nx[j] * nx[j] + ny[j] * ny[j]);
            nx[j] /= norm;
            ny[j] /= norm;

            // offset the point a bit for visualization
            x[j] += 0.5 * nx[j];
            y[j] += 0.5 * ny[j];
        }

        PlplotFig fig(640, 640, true);
        auto clear_fig = [&fig, &grid_info, this]() {
            fig.Clear(1.0, 1.0, 1.0)
                .SetFontSize(0.0, 1.1)
                .SetMargin(0.12, 0.82, 0.15, 0.85)
                .SetAxisLimits(
                    grid_info.Min(0),
                    grid_info.Max(0),
                    grid_info.Min(1),
                    grid_info.Max(1))
                .SetCurrentColor(PlplotFig::Color0::Black);
        };
        auto draw_axes = [&fig]() {
            fig.SetCurrentColor(PlplotFig::Color0::Black)
                .DrawAxesBox(
                    PlplotFig::AxisOpt().DrawTopRightEdge(),
                    PlplotFig::AxisOpt().DrawTopRightEdge().DrawPerpendicularTickLabels())
                .SetAxisLabelX("x")
                .SetAxisLabelY("y");
        };

        PlplotFig::ShadesOpt shades_opt;
        shades_opt.SetColorLevels(occ_values.data(), options.test_num_x, options.test_num_y, 127)
            .SetXMin(grid_info.Min(0))
            .SetXMax(grid_info.Max(0))
            .SetYMin(grid_info.Min(1))
            .SetYMax(grid_info.Max(1));
        PlplotFig::ColorBarOpt color_bar_opt;
        color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM})
            .SetLabelTexts({"Occupancy"})
            .AddColorMap(0, shades_opt.color_levels, 10);
        clear_fig();
        fig.SetTitle("Occupancy Prediction")
            .SetAreaFillPattern(PlplotFig::AreaFillPattern::Solid)
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(occ_values.data(), options.test_num_x, options.test_num_y, true, shades_opt)
            .ColorBar(color_bar_opt)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .Scatter(
                static_cast<int>(pts_world.cols()),
                Eigen::VectorXd(pts_world.row(0)).data(),
                Eigen::VectorXd(pts_world.row(1)).data())
            .SetCurrentColor(PlplotFig::Color0::White)
            .SetPenWidth(2)
            .VectorField(x.data(), y.data(), nx.data(), ny.data(), static_cast<int>(x.size()), 2.0);
        draw_axes();
        cv::imshow("gp_occ_regression: occupancy prediction", fig.ToCvMat());

        cv::waitKey(0);
    }
};

int
main(int argc, char **argv) {
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " <options_file>" << std::endl;
        return EXIT_FAILURE;
    }
    try {
        const std::string options_file =
            (argc == 2) ? argv[1]
                        : (ERL_SDF_MAPPING_ROOT_DIR "/config/demo_gp_occ_regression.yaml");
        App app(options_file);
        app.Run();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
