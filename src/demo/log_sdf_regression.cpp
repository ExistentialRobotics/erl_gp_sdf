#include "erl_common/plplot_fig.hpp"
#include "erl_sdf_mapping/log_edf_gp.hpp"

using namespace erl::common;
using namespace erl::sdf_mapping;
using Gp = LogEdfGaussianProcessD;

struct Options : Yamlable<Options> {
    double max_x = 2.0;
    double max_y = 2.0;
    double min_x = -2.0;
    double min_y = -2.0;
    double radius = 1.0;
    long num_samples = 10000;
    int test_num_x = 201;
    int test_num_y = 201;
    double var_x = 0.01;
    double var_y = 0.01;
    std::shared_ptr<Gp::Setting> gp = std::make_shared<Gp::Setting>();
    std::string output_dir = "log_sdf_regression";
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
        ERL_YAML_LOAD_ATTR(node, options, var_x);
        ERL_YAML_LOAD_ATTR(node, options, var_y);
        ERL_YAML_LOAD_ATTR(node, options, gp);
        ERL_YAML_LOAD_ATTR(node, options, output_dir);
        return true;
    }
};

struct App {
    Options options;

    explicit App(const std::string &option_file) {  // NOLINT(*-msc51-cpp)
        if (!option_file.empty()) {
            ERL_ASSERTM(
                options.FromYamlFile(option_file),
                "Failed to load options from file: {}",
                option_file);
        }
    }

    [[nodiscard]] Eigen::Matrix2Xd
    GenerateDataset() const {
        Eigen::Matrix2Xd points(2, options.num_samples);
        Eigen::VectorXd angles = Eigen::VectorXd::LinSpaced(options.num_samples, 0.0, 2.0 * M_PI);
        for (long i = 0; i < options.num_samples; ++i) {
            auto point = points.col(i);
            point.x() = options.radius * std::cos(angles[i]);
            point.y() = options.radius * std::sin(angles[i]);
        }
        return points;
    }

    std::shared_ptr<Gp>
    Train(const Eigen::Matrix2Xd &points) {
        options.gp->no_gradient_observation = true;
        auto gp = std::make_shared<Gp>(options.gp);
        gp->Reset(options.num_samples, 2, 3);

        Eigen::Matrix2Xd normals = points;
        normals.colwise().normalize();

        auto &train_set = gp->GetTrainSet();
        train_set.x.topRows<2>() = points;
        train_set.y.leftCols<1>().setConstant(1);
        train_set.y.col(1) = 10 * normals.row(0).transpose();
        train_set.y.col(2) = 10 * normals.row(1).transpose();
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

    // [[nodiscard]] std::tuple<  //
    //     Eigen::Vector2i,
    //     Eigen::Matrix2Xd,
    //     Eigen::VectorXd,
    //     Eigen::VectorXd,
    //     Eigen::VectorXd,
    //     Eigen::Matrix2Xd,
    //     Eigen::VectorXb,
    //     Eigen::Matrix2Xd>
    // Test(const std::shared_ptr<Gp> &gp, const int stride = 1, const double margin = 0.0) const {
    //     GridMapInfo2Dd grid_info(
    //         Eigen::Vector2i(options.test_num_x / stride, options.test_num_y / stride),
    //         Eigen::Vector2d(options.min_x + margin, options.min_y + margin),
    //         Eigen::Vector2d(options.max_x - margin, options.max_y - margin));
    //     Eigen::Matrix2Xd points = grid_info.GenerateMeterCoordinates(true /*c_stride*/);
    //     Eigen::VectorXd sdf_gt = points.colwise().norm().array() - options.radius;
    //     Eigen::VectorXd edf_pred(points.cols());
    //     Eigen::Matrix2Xd gradients(2, points.cols());
    //     auto test_result = gp->Test(points, true /*predict_gradient*/);
    //     ERL_ASSERTM(test_result != nullptr, "Failed to test Gaussian Process.");
    //     test_result->GetMean(0, edf_pred, true /*parallel*/);
    //     Eigen::VectorXb valid_gradients = test_result->GetGradient(0, gradients, true
    //     /*parallel*/);
    //
    //     Eigen::Matrix2Xd normals(2, points.cols());
    //     Eigen::VectorXd u(points.cols());
    //     test_result->GetMean(1, u, true);
    //     normals.row(0) = u.transpose();
    //     test_result->GetMean(2, u, true);
    //     normals.row(1) = u.transpose();
    //
    //     Eigen::VectorXd var(points.cols());
    //     test_result->GetMeanVariance(var, true);
    //     return {
    //         grid_info.Shape(),
    //         points,
    //         sdf_gt,
    //         edf_pred,
    //         var,
    //         gradients,
    //         valid_gradients,
    //         normals};
    // }

    void
    Run() {
        if (!std::filesystem::exists(options.output_dir)) {
            std::filesystem::create_directories(options.output_dir);
        }

        auto dataset = GenerateDataset();
        auto gp = Train(dataset);

        GridMapInfo2Dd grid_info(
            Eigen::Vector2i(options.test_num_x, options.test_num_y),
            Eigen::Vector2d(options.min_x, options.min_y),
            Eigen::Vector2d(options.max_x, options.max_y));
        Eigen::Matrix2Xd test_pts = grid_info.GenerateMeterCoordinates(true /*c_stride*/);
        Eigen::VectorXd sdf_gt = test_pts.colwise().norm().array() - options.radius;
        Eigen::VectorXd edf_pred(test_pts.cols());
        Eigen::Matrix2Xd edf_grads(2, test_pts.cols());
        auto test_result = gp->Test(test_pts, true /*predict_gradient*/);
        ERL_ASSERTM(test_result != nullptr, "Failed to test Gaussian Process.");
        test_result->GetMean(0, edf_pred, true /*parallel*/);
        Eigen::VectorXb valid_gradients = test_result->GetGradient(0, edf_grads, true /*parallel*/);

        Eigen::Matrix2Xd normals(2, test_pts.cols());
        Eigen::VectorXd buf(test_pts.cols());
        test_result->GetMean(1, buf, true);
        normals.row(0) = buf.transpose();
        test_result->GetMean(2, buf, true);
        normals.row(1) = buf.transpose();
        normals.colwise().normalize();

        ERL_INFO("Computing var");
        Eigen::VectorXd var(test_pts.cols());
        test_result->GetMeanVariance(var, true);

        ERL_INFO("Computing SDF from EDF");
        Eigen::VectorXd edf_pred_dot_normals =
            (edf_grads.array() * normals.array()).colwise().sum();
        Eigen::VectorXd signs = edf_pred_dot_normals.unaryExpr(
            [](const double a) -> double { return (a < 0.0) ? -1.0 : 1.0; });
        Eigen::VectorXd sdf_pred = edf_pred.array() * signs.array();

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
        shades_opt.SetColorLevels(edf_pred.data(), options.test_num_x, options.test_num_y, 127)
            .SetXMin(options.min_x)
            .SetXMax(options.max_x)
            .SetYMin(options.min_y)
            .SetYMax(options.max_y);
        PlplotFig::ColorBarOpt color_bar_opt;
        color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM})
            .SetLabelTexts({"Pred"})
            .AddColorMap(0, shades_opt.color_levels, 10);
        clear_fig();
        fig.SetTitle("EDF Prediction")
            .SetAreaFillPattern(PlplotFig::AreaFillPattern::Solid)
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(edf_pred.data(), options.test_num_x, options.test_num_y, true, shades_opt)
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
        cv::imshow("log_sdf_regression: edf prediction", fig.ToCvMat());

        shades_opt.SetColorLevels(sdf_pred.data(), options.test_num_x, options.test_num_y, 127);
        color_bar_opt.SetLabelTexts({"Pred"}).AddColorMap(0, shades_opt.color_levels, 10);
        clear_fig();
        fig.SetTitle("SDF Prediction")
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
        cv::imshow("log_sdf_regression: sdf prediction", fig.ToCvMat());

        shades_opt.SetColorLevels(
            edf_pred_dot_normals.data(),
            options.test_num_x,
            options.test_num_y,
            127);
        color_bar_opt.SetLabelTexts({"Dot"}).AddColorMap(0, shades_opt.color_levels, 10);
        clear_fig();
        fig.SetAreaFillPattern(PlplotFig::AreaFillPattern::Solid)
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(
                edf_pred_dot_normals.data(),
                options.test_num_x,
                options.test_num_y,
                true,
                shades_opt)
            .ColorBar(color_bar_opt);
        draw_axes();
        cv::imshow("log_sdf_regression: edf_pred_dot_normals", fig.ToCvMat());

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
        cv::imshow("log_sdf_regression: ground truth", fig.ToCvMat());

        Eigen::VectorXd error = (edf_pred - sdf_gt).cwiseAbs();
        shades_opt.SetColorLevels(error.data(), options.test_num_x, options.test_num_y, 127);
        color_bar_opt.SetLabelTexts({"Abs. Err."}).AddColorMap(0, shades_opt.color_levels, 10);
        clear_fig();
        fig.SetTitle("Absolute Error")
            .SetAreaFillPattern(PlplotFig::AreaFillPattern::Solid)
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(error.data(), options.test_num_x, options.test_num_y, true, shades_opt)
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
        cv::imshow("log_sdf_regression: edf error", fig.ToCvMat());

        error = (sdf_pred - sdf_gt).cwiseAbs();
        shades_opt.SetColorLevels(error.data(), options.test_num_x, options.test_num_y, 127);
        color_bar_opt.SetLabelTexts({"Abs. Err."}).AddColorMap(0, shades_opt.color_levels, 10);
        clear_fig();
        fig.SetTitle("Absolute Error")
            .SetAreaFillPattern(PlplotFig::AreaFillPattern::Solid)
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(error.data(), options.test_num_x, options.test_num_y, true, shades_opt)
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
        cv::imshow("log_sdf_regression: sdf error", fig.ToCvMat());

        shades_opt.SetColorLevels(var.data(), options.test_num_x, options.test_num_y, 127);
        color_bar_opt.SetLabelTexts({"Var."}).AddColorMap(0, shades_opt.color_levels, 10);
        clear_fig();
        fig.SetTitle("Variance")
            .SetAreaFillPattern(PlplotFig::AreaFillPattern::Solid)
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(var.data(), options.test_num_x, options.test_num_y, true, shades_opt)
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
        cv::imshow("log_sdf_regression: variance", fig.ToCvMat());

        GridMapInfo2Dd grid_info_stride(
            Eigen::Vector2i(options.test_num_x / 20, options.test_num_y / 20),
            Eigen::Vector2d(options.min_x + 0.1, options.min_y + 0.1),
            Eigen::Vector2d(options.max_x - 0.1, options.max_y - 0.1));
        test_pts = grid_info_stride.GenerateMeterCoordinates(true /*c_stride*/);
        test_result = gp->Test(test_pts, true /*predict_gradient*/);
        edf_grads.resize(2, test_pts.cols());
        (void) test_result->GetGradient(0, edf_grads, true /*parallel*/);
        Eigen::VectorXd x = test_pts.row(0);
        Eigen::VectorXd y = test_pts.row(1);
        Eigen::VectorXd u = edf_grads.row(0);
        Eigen::VectorXd v = edf_grads.row(1);
        clear_fig();
        fig.SetTitle("EDF Gradient Prediction")
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
            .VectorField(x.data(), y.data(), u.data(), v.data(), static_cast<int>(x.size()), 0.5);
        draw_axes();
        cv::imshow("log_sdf_regression: edf gradient pred", fig.ToCvMat());

        buf.resize(test_pts.cols());
        test_result->GetMean(1, buf, true);
        normals.row(0) = buf.transpose();
        test_result->GetMean(2, buf, true);
        normals.row(1) = buf.transpose();

        normals.colwise().normalize();
        u = normals.row(0).transpose();
        v = normals.row(1).transpose();
        clear_fig();
        fig.SetTitle("Normal Diffusion")
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
            .VectorField(x.data(), y.data(), u.data(), v.data(), static_cast<int>(x.size()), 0.5);
        draw_axes();
        cv::imshow("log_sdf_regression: normal diffusion", fig.ToCvMat());

        constexpr long stride = 200;
        u.resize(dataset.cols() / stride);
        v.resize(u.size());
        x.resize(u.size());
        y.resize(u.size());
        for (long i = 0, j = 0; i < dataset.cols(); i += stride, ++j) {
            x[j] = dataset(0, i);
            y[j] = dataset(1, i);
            const double norm = std::sqrt(x[j] * x[j] + y[j] * y[j]);
            u[j] = x[j] / norm;
            v[j] = y[j] / norm;
            // shift the points slightly to make the vector starting from the point
            x[j] += 0.125 * u[j];
            y[j] += 0.125 * v[j];
        }
        clear_fig();
        fig.SetTitle("Training Data")
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
            .VectorField(x.data(), y.data(), u.data(), v.data(), static_cast<int>(x.size()), 0.5);
        draw_axes();
        cv::imshow("log_sdf_regression: training data", fig.ToCvMat());

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
            (argc == 2) ? argv[1]
                        : (ERL_SDF_MAPPING_ROOT_DIR "/config/demo_log_sdf_regression.yaml");
        App app(options_file);
        app.Run();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
