#include "erl_common/plplot_fig.hpp"
#include "erl_common/random.hpp"
#include "erl_common/yaml.hpp"
#include "erl_gaussian_process/noisy_input_gp.hpp"

using namespace erl::common;
using namespace erl::gaussian_process;
using Gp = NoisyInputGaussianProcessD;

struct Options : Yamlable<Options> {
    double max_x = 2.0;
    double max_y = 2.0;
    double min_x = -2.0;
    double min_y = -2.0;
    double radius = 1.0;
    long num_samples = 10000;
    int test_num_x = 201;
    int test_num_y = 201;
    uint64_t seed = 0;
    bool near_surface = false;
    double near_surface_delta = 0.05;
    double var_x = 0.01;
    double var_y = 0.01;
    std::shared_ptr<Gp::Setting> gp = std::make_shared<Gp::Setting>();
    std::string output_dir = "sdf_regression";
};

template<>
struct YAML::convert<Options> {
    static Node
    encode(const Options &options) {
        Node node;
        ERL_YAML_SAVE_ATTR(node, options, max_x);
        ERL_YAML_SAVE_ATTR(node, options, max_y);
        ERL_YAML_SAVE_ATTR(node, options, min_x);
        ERL_YAML_SAVE_ATTR(node, options, min_y);
        ERL_YAML_SAVE_ATTR(node, options, radius);
        ERL_YAML_SAVE_ATTR(node, options, num_samples);
        ERL_YAML_SAVE_ATTR(node, options, test_num_x);
        ERL_YAML_SAVE_ATTR(node, options, test_num_y);
        ERL_YAML_SAVE_ATTR(node, options, seed);
        ERL_YAML_SAVE_ATTR(node, options, near_surface);
        ERL_YAML_SAVE_ATTR(node, options, near_surface_delta);
        ERL_YAML_SAVE_ATTR(node, options, var_x);
        ERL_YAML_SAVE_ATTR(node, options, var_y);
        ERL_YAML_SAVE_ATTR(node, options, gp);
        ERL_YAML_SAVE_ATTR(node, options, output_dir);
        return node;
    }

    static bool
    decode(const Node &node, Options &options) {
        if (!node.IsMap()) { return false; }
        ERL_YAML_LOAD_ATTR(node, options, max_x);
        ERL_YAML_LOAD_ATTR(node, options, max_y);
        ERL_YAML_LOAD_ATTR(node, options, min_x);
        ERL_YAML_LOAD_ATTR(node, options, min_y);
        ERL_YAML_LOAD_ATTR(node, options, radius);
        ERL_YAML_LOAD_ATTR(node, options, num_samples);
        ERL_YAML_LOAD_ATTR(node, options, test_num_x);
        ERL_YAML_LOAD_ATTR(node, options, test_num_y);
        ERL_YAML_LOAD_ATTR(node, options, seed);
        ERL_YAML_LOAD_ATTR(node, options, near_surface);
        ERL_YAML_LOAD_ATTR(node, options, near_surface_delta);
        ERL_YAML_LOAD_ATTR(node, options, var_x);
        ERL_YAML_LOAD_ATTR(node, options, var_y);
        ERL_YAML_LOAD_ATTR(node, options, gp);
        ERL_YAML_LOAD_ATTR(node, options, output_dir);
        return true;
    }
};

struct App {
    Options options;
    std::mt19937_64 rng;
    std::uniform_real_distribution<double> dist;

    explicit App(const std::string &option_file) {  // NOLINT(*-msc51-cpp)
        if (!std::filesystem::exists(option_file)) { options.AsYamlFile(option_file); }
        if (!option_file.empty()) {
            ERL_ASSERTM(
                options.FromYamlFile(option_file),
                "Failed to load options from file: {}",
                option_file);
        }

        rng = std::mt19937_64(options.seed);
        dist = std::uniform_real_distribution<double>(0.0, 1.0);
    }

    std::pair<Eigen::Matrix2Xd, Eigen::VectorXd>
    GenerateDataset() {
        if (options.near_surface) {
            Eigen::Matrix2Xd points(2, options.num_samples);
            Eigen::VectorXd sdf_values(options.num_samples);
            for (long i = 0; i < options.num_samples; ++i) {
                double theta = dist(rng) * 2.0 * M_PI;
                double r = options.radius + dist(rng) * options.near_surface_delta;
                points.col(i) << r * std::cos(theta), r * std::sin(theta);
                sdf_values[i] = r - options.radius;  // SDF value
            }
            return {points, sdf_values};
        }

        Eigen::Matrix2Xd points(2, options.num_samples);
        Eigen::VectorXd sdf_values(options.num_samples);
        const double range_x = options.max_x - options.min_x;
        const double range_y = options.max_y - options.min_y;
        for (long i = 0; i < options.num_samples; ++i) {
            auto point = points.col(i);
            point.x() = dist(rng) * range_x + options.min_x;
            point.y() = dist(rng) * range_y + options.min_y;
            sdf_values[i] = point.norm() - options.radius;
        }
        return {points, sdf_values};
    }

    std::shared_ptr<Gp>
    Train(const Eigen::Matrix2Xd &points, const Eigen::VectorXd &sdf_values) {
        auto gp = std::make_shared<Gp>(options.gp);
        gp->Reset(options.num_samples, 2, 1);
        auto &train_set = gp->GetTrainSet();
        train_set.x.topRows<2>() = points;
        train_set.y.leftCols<1>() = sdf_values;
        train_set.var_x.setConstant(options.var_x);
        train_set.var_y.setConstant(options.var_y);
        train_set.num_samples = options.num_samples;
        train_set.num_samples_with_grad = 0;
        train_set.grad_flag.setConstant(false);
        train_set.x_dim = 2;
        train_set.y_dim = 1;
        ERL_ASSERTM(gp->Train(), "Failed to train Gaussian Process.");
        return gp;
    }

    [[nodiscard]] std::tuple<  //
        Eigen::Vector2i,
        Eigen::Matrix2Xd,
        Eigen::VectorXd,
        Eigen::VectorXd,
        Eigen::Matrix2Xd,
        Eigen::VectorXb>
    Test(const std::shared_ptr<Gp> &gp, const int stride = 1, const double margin = 0.0) const {
        GridMapInfo2Dd grid_info(
            Eigen::Vector2i(options.test_num_x / stride, options.test_num_y / stride),
            Eigen::Vector2d(options.min_x + margin, options.min_y + margin),
            Eigen::Vector2d(options.max_x - margin, options.max_y - margin));
        Eigen::Matrix2Xd points = grid_info.GenerateMeterCoordinates(true /*c_stride*/);
        Eigen::VectorXd sdf_gt = points.colwise().norm().array() - options.radius;
        Eigen::VectorXd sdf_pred(points.cols());
        Eigen::Matrix2Xd gradients(2, points.cols());
        auto test_result = gp->Test(points, true /*predict_gradient*/);
        ERL_ASSERTM(test_result != nullptr, "Failed to test Gaussian Process.");
        test_result->GetMean(0, sdf_pred, true /*parallel*/);
        Eigen::VectorXb valid_gradients = test_result->GetGradient(0, gradients, true /*parallel*/);
        return {grid_info.Shape(), points, sdf_gt, sdf_pred, gradients, valid_gradients};
    }

    void
    Run() {
        if (!std::filesystem::exists(options.output_dir)) {
            std::filesystem::create_directories(options.output_dir);
        }

        auto dataset = GenerateDataset();
        auto gp = Train(dataset.first, dataset.second);
        auto [grid_shape, test_points, sdf_gt, sdf_pred, gradients, valid_gradients] = Test(gp);

        PlplotFig fig(640, 640, true);
        auto clear_fig = [&fig, this]() {
            fig.Clear()
                .SetFontSize(0.0, 1.1)
                .SetMargin(0.12, 0.82, 0.15, 0.85)
                .SetAxisLimits(options.min_x, options.max_x, options.min_y, options.max_y)
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
        shades_opt.SetColorLevels(sdf_pred.data(), options.test_num_x, options.test_num_y, 127)
            .SetXMin(options.min_x)
            .SetXMax(options.max_x)
            .SetYMin(options.min_y)
            .SetYMax(options.max_y);
        PlplotFig::ColorBarOpt color_bar_opt;
        color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM})
            .SetLabelTexts({"Pred"})
            .AddColorMap(0, shades_opt.color_levels, 10);
        clear_fig();
        fig.SetTitle("Prediction")
            .SetAreaFillPattern(PlplotFig::AreaFillPattern::Solid)
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(sdf_pred.data(), options.test_num_x, options.test_num_y, true, shades_opt)
            .ColorBar(color_bar_opt)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .SetPenWidth(2)
            .DrawContour(
                sdf_pred.data(),
                options.test_num_x,
                options.test_num_y,
                options.min_x,
                options.max_x,
                options.min_y,
                options.max_y,
                true,
                {0.0})
            .SetPenWidth(1);
        draw_axes();
        cv::imshow("sdf_regression: prediction", fig.ToCvMat());

        shades_opt.SetColorLevels(sdf_gt.data(), options.test_num_x, options.test_num_y, 127);
        color_bar_opt.SetLabelTexts({"G.T."}).AddColorMap(0, shades_opt.color_levels, 10);
        clear_fig();
        fig.SetTitle("Ground Truth")
            .SetAreaFillPattern(PlplotFig::AreaFillPattern::Solid)
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(sdf_gt.data(), options.test_num_x, options.test_num_y, true, shades_opt)
            .ColorBar(color_bar_opt)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .SetPenWidth(2)
            .DrawContour(
                sdf_gt.data(),
                options.test_num_x,
                options.test_num_y,
                options.min_x,
                options.max_x,
                options.min_y,
                options.max_y,
                true,
                {0.0})
            .SetPenWidth(1);
        draw_axes();
        cv::imshow("sdf_regression: ground truth", fig.ToCvMat());

        Eigen::VectorXd sdf_error = (sdf_pred - sdf_gt).cwiseAbs();
        shades_opt.SetColorLevels(sdf_error.data(), options.test_num_x, options.test_num_y, 127);
        color_bar_opt.SetLabelTexts({"Abs. Err."}).AddColorMap(0, shades_opt.color_levels, 10);
        clear_fig();
        fig.SetTitle("Absolute Error")
            .SetAreaFillPattern(PlplotFig::AreaFillPattern::Solid)
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(sdf_error.data(), options.test_num_x, options.test_num_y, true, shades_opt)
            .ColorBar(color_bar_opt)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .SetPenWidth(2)
            .DrawContour(
                sdf_pred.data(),
                options.test_num_x,
                options.test_num_y,
                options.min_x,
                options.max_x,
                options.min_y,
                options.max_y,
                true,
                {0.0})
            .SetPenWidth(1);
        draw_axes();
        cv::imshow("sdf_regression: error", fig.ToCvMat());

        clear_fig();
        auto results = Test(gp, 20, 0.1);
        grid_shape = std::get<0>(results);
        gradients = std::get<4>(results);
        gradients.colwise().normalize();
        Eigen::VectorXd u = gradients.row(0);
        Eigen::VectorXd v = gradients.row(1);
        fig.SetTitle("Gradient Prediction")
            .SetCurrentColor(PlplotFig::Color0::Green)
            .DrawContour(
                sdf_gt.data(),
                options.test_num_x,
                options.test_num_y,
                options.min_x,
                options.max_x,
                options.min_y,
                options.max_y,
                true,
                {0.0})
            .SetCurrentColor(PlplotFig::Color0::Red)
            .SetPenWidth(2)
            .VectorField(
                u.data(),
                v.data(),
                grid_shape.x(),
                grid_shape.y(),
                options.min_x + 0.1,
                options.max_x - 0.1,
                options.min_y + 0.1,
                options.max_y - 0.1,
                true,
                0.5);
        draw_axes();
        cv::imshow("sdf_regression: gradient pred", fig.ToCvMat());

        cv::waitKey();
    }
};

int
main(const int argc, char **argv) {
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " <options_file>" << std::endl;
        return EXIT_FAILURE;
    }
    try {
        const std::string options_file =
            (argc == 2) ? argv[1] : (ERL_SDF_MAPPING_ROOT_DIR "/config/demo_sdf_regression.yaml");
        App app(options_file);
        app.Run();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
