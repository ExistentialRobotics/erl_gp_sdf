#include "erl_common/plplot_fig.hpp"
#include "erl_geometry/kdtree_eigen_adaptor.hpp"
#include "erl_gp_sdf/log_edf_gp.hpp"

#include <random>

using namespace erl::common;
using namespace erl::geometry;
using namespace erl::gp_sdf;
using Gp = LogEdfGaussianProcessD;

struct Options : Yamlable<Options> {
    double max_x = 2.0;
    double max_y = 2.0;
    double min_x = -2.0;
    double min_y = -2.0;
    double radius = 1.0;
    long num_surf_samples = 1000;
    bool add_off_surf_points = false;
    int off_surf_grid_size = 21;
    int test_num_x = 201;
    int test_num_y = 201;
    bool add_noise_to_surf_points = false;
    bool add_noise_along_radius = false;
    double var_x = 0.01;
    double var_y = 0.01;
    double softmin_alpha = 10.0;
    int softmin_knn = 100;
    std::shared_ptr<Gp::Setting> gp = std::make_shared<Gp::Setting>();
    std::string output_dir = "log_sdf_regression";
    bool visualize = true;
    bool hold = true;
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
        ERL_YAML_SAVE_ATTR(node, options, num_surf_samples);
        ERL_YAML_SAVE_ATTR(node, options, add_off_surf_points);
        ERL_YAML_SAVE_ATTR(node, options, off_surf_grid_size);
        ERL_YAML_SAVE_ATTR(node, options, test_num_x);
        ERL_YAML_SAVE_ATTR(node, options, test_num_y);
        ERL_YAML_SAVE_ATTR(node, options, add_noise_to_surf_points);
        ERL_YAML_SAVE_ATTR(node, options, add_noise_along_radius);
        ERL_YAML_SAVE_ATTR(node, options, var_x);
        ERL_YAML_SAVE_ATTR(node, options, var_y);
        ERL_YAML_SAVE_ATTR(node, options, softmin_alpha);
        ERL_YAML_SAVE_ATTR(node, options, softmin_knn);
        ERL_YAML_SAVE_ATTR(node, options, gp);
        ERL_YAML_SAVE_ATTR(node, options, output_dir);
        ERL_YAML_SAVE_ATTR(node, options, visualize);
        ERL_YAML_SAVE_ATTR(node, options, hold);
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
        ERL_YAML_LOAD_ATTR(node, options, num_surf_samples);
        ERL_YAML_LOAD_ATTR(node, options, add_off_surf_points);
        ERL_YAML_LOAD_ATTR(node, options, off_surf_grid_size);
        ERL_YAML_LOAD_ATTR(node, options, test_num_x);
        ERL_YAML_LOAD_ATTR(node, options, test_num_y);
        ERL_YAML_LOAD_ATTR(node, options, add_noise_to_surf_points);
        ERL_YAML_LOAD_ATTR(node, options, add_noise_along_radius);
        ERL_YAML_LOAD_ATTR(node, options, var_x);
        ERL_YAML_LOAD_ATTR(node, options, var_y);
        ERL_YAML_LOAD_ATTR(node, options, softmin_alpha);
        ERL_YAML_LOAD_ATTR(node, options, softmin_knn);
        if (!ERL_YAML_LOAD_ATTR(node, options, gp)) { return false; }
        ERL_YAML_LOAD_ATTR(node, options, output_dir);
        ERL_YAML_LOAD_ATTR(node, options, visualize);
        ERL_YAML_LOAD_ATTR(node, options, hold);
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

    [[nodiscard]] Eigen::Matrix2Xd
    GenerateDataset() const {
        const long num_off_surf_samples =
            options.add_off_surf_points ? options.off_surf_grid_size * options.off_surf_grid_size
                                        : 0;
        Eigen::Matrix2Xd points(2, options.num_surf_samples + num_off_surf_samples);
        Eigen::VectorXd angles =
            Eigen::VectorXd::LinSpaced(options.num_surf_samples, 0.0, 2.0 * M_PI);
        std::normal_distribution<double> dist(0.0, std::sqrt(options.var_x));
        std::mt19937_64 gen(0);  // NOLINT(*-msc51-cpp)
        for (long i = 0; i < options.num_surf_samples; ++i) {
            auto point = points.col(i);
            if (options.add_noise_to_surf_points) {
                if (options.add_noise_along_radius) {
                    const double r = options.radius + dist(gen);
                    point.x() = r * std::cos(angles[i]);
                    point.y() = r * std::sin(angles[i]);
                } else {
                    point.x() = options.radius * std::cos(angles[i]);
                    point.y() = options.radius * std::sin(angles[i]);
                    point.x() += dist(gen);
                    point.y() += dist(gen);
                }
            } else {
                point.x() = options.radius * std::cos(angles[i]);
                point.y() = options.radius * std::sin(angles[i]);
            }
        }
        if (options.add_off_surf_points) {
            GridMapInfo2Dd grid_info(
                Eigen::Vector2i(options.off_surf_grid_size, options.off_surf_grid_size),
                Eigen::Vector2d(options.min_x, options.min_y),
                Eigen::Vector2d(options.max_x, options.max_y));
            points.rightCols(num_off_surf_samples) = grid_info.GenerateMeterCoordinates(true);
        }
        return points;
    }

    std::shared_ptr<Gp>
    Train(const Eigen::Matrix2Xd &points) {
        options.gp->no_gradient_observation = true;
        options.gp->kernel->scale = std::sqrt(3.) / options.gp->log_lambda;
        auto gp = std::make_shared<Gp>(options.gp);
        const long num_samples = points.cols();
        gp->Reset(num_samples, 2, 3);

        Eigen::Matrix2Xd normals = points;
        normals.colwise().normalize();
        Eigen::VectorXd sdf = points.colwise().norm().array() - options.radius;

        auto &train_set = gp->GetTrainSet();
        train_set.x.topRows<2>() = points;
        train_set.y.col(0) = (sdf.array() * -options.gp->log_lambda).exp();
        train_set.y.col(1) = 10 * normals.row(0).transpose();
        train_set.y.col(2) = 10 * normals.row(1).transpose();
        train_set.var_x.setConstant(options.var_x);
        train_set.var_y.setConstant(options.var_y);
        train_set.num_samples = num_samples;
        train_set.num_samples_with_grad = 0;
        train_set.grad_flag.setConstant(false);
        train_set.x_dim = 2;
        train_set.y_dim = 3;
        ERL_ASSERTM(gp->Train(), "Failed to train Gaussian Process.");
        return gp;
    }

    [[nodiscard]] Eigen::VectorXd
    PredictUdfBySoftmin(
        const Eigen::Ref<const Eigen::Matrix2Xd> &test_points,
        const Eigen::Ref<const Eigen::Matrix2Xd> &surf_points) const {
        KdTree2d tree(surf_points);
        const long num_test = test_points.cols();
        Eigen::VectorXd udf(num_test);
        Eigen::VectorXl indices(options.softmin_knn);
        Eigen::VectorXd dists(options.softmin_knn);
        for (long i = 0; i < num_test; ++i) {
            indices.setConstant(-1l);
            tree.Knn(options.softmin_knn, test_points.col(i), indices, dists);
            double dist = 0.0;
            double weight_sum = 0.0;
            for (long j = 0; j < options.softmin_knn; ++j) {
                if (indices[j] < 0) { break; }
                const double d = std::sqrt(dists[j]);
                const double w = std::exp(-options.softmin_alpha * d);
                dist += w * d;
                weight_sum += w;
            }
            udf[i] = dist / weight_sum;
        }
        return udf;
    }

    void
    DrawValueField(
        const Eigen::VectorXd &value_field,
        const double *contour_field,
        const char *title,
        const char *color_bar_title,
        const std::string &img_file,
        const std::string &win_name,
        PlplotFig &fig,
        PlplotFig::ShadesOpt &shades_opt,
        PlplotFig::ColorBarOpt &color_bar_opt) const {

        shades_opt.SetColorLevels(value_field.data(), options.test_num_x, options.test_num_y, 127);
        color_bar_opt.SetLabelTexts({color_bar_title}).AddColorMap(0, shades_opt.color_levels, 10);

        fig.Clear()
            .SetFontSize(0.0, 1.1)
            .SetMargin(0.12, 0.82, 0.15, 0.85)
            .SetAxisLimits(options.min_x, options.max_x, options.min_y, options.max_y)
            .SetCurrentColor(PlplotFig::Color0::Black);

        fig.SetTitle(title)
            .SetAreaFillPattern(PlplotFig::AreaFillPattern::Solid)
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(value_field.data(), options.test_num_x, options.test_num_y, true, shades_opt)
            .ColorBar(color_bar_opt)
            .SetCurrentColor(PlplotFig::Color0::Black);
        if (contour_field != nullptr) {
            fig.SetPenWidth(2)
                .DrawContour(
                    contour_field,
                    options.test_num_x,
                    options.test_num_y,
                    options.min_x,
                    options.max_x,
                    options.min_y,
                    options.max_y,
                    true,
                    {0.0})
                .SetPenWidth(1);
        }

        fig.SetCurrentColor(PlplotFig::Color0::Black)
            .DrawAxesBox(
                PlplotFig::AxisOpt().DrawTopRightEdge(),
                PlplotFig::AxisOpt().DrawTopRightEdge().DrawPerpendicularTickLabels())
            .SetAxisLabelX("x")
            .SetAxisLabelY("y");

        cv::Mat img = fig.ToCvMat();
        cv::imwrite(img_file, img);
        cv::imshow(win_name, img);
    }

    void
    DrawVectorField(
        const Eigen::VectorXd &x,
        const Eigen::VectorXd &y,
        const Eigen::VectorXd &u,
        const Eigen::VectorXd &v,
        const double *contour_field,
        const char *title,
        const std::string &img_file,
        const std::string &win_name,
        PlplotFig &fig) const {

        fig.Clear()
            .SetFontSize(0.0, 1.1)
            .SetMargin(0.12, 0.82, 0.15, 0.85)
            .SetAxisLimits(options.min_x, options.max_x, options.min_y, options.max_y)
            .SetCurrentColor(PlplotFig::Color0::Black);

        fig.SetTitle(title)
            .SetCurrentColor(PlplotFig::Color0::Green)
            .DrawContour(
                contour_field,
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
            .VectorField(x.data(), y.data(), u.data(), v.data(), static_cast<int>(x.size()), 0.5)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .Scatter(x.size(), x.data(), y.data());

        fig.SetCurrentColor(PlplotFig::Color0::Black)
            .DrawAxesBox(
                PlplotFig::AxisOpt().DrawTopRightEdge(),
                PlplotFig::AxisOpt().DrawTopRightEdge().DrawPerpendicularTickLabels())
            .SetAxisLabelX("x")
            .SetAxisLabelY("y");

        cv::Mat img = fig.ToCvMat();
        cv::imwrite(img_file, img);
        cv::imshow(win_name, img);
    }

    void
    Run() {
        if (!std::filesystem::exists(options.output_dir)) {
            std::filesystem::create_directories(options.output_dir);
        }

        ERL_INFO("Generating dataset");
        auto dataset = GenerateDataset();
        ERL_INFO("Training gp");
        auto gp = Train(dataset);

        GridMapInfo2Dd grid_info(
            Eigen::Vector2i(options.test_num_x, options.test_num_y),
            Eigen::Vector2d(options.min_x, options.min_y),
            Eigen::Vector2d(options.max_x, options.max_y));
        Eigen::Matrix2Xd test_pts = grid_info.GenerateMeterCoordinates(true /*c_stride*/);
        Eigen::VectorXd sdf_gt = test_pts.colwise().norm().array() - options.radius;
        Eigen::VectorXd udf_gt = sdf_gt.cwiseAbs();
        Eigen::VectorXd signs_gt =
            sdf_gt.unaryExpr([](const double a) -> double { return (a < 0.0) ? -1.0 : 1.0; });
        Eigen::VectorXd udf_pred_gp(test_pts.cols());
        Eigen::Matrix2Xd udf_grads(2, test_pts.cols());

        ERL_INFO("Testing gp");
        auto test_result = gp->Test(test_pts, true /*predict_gradient*/);
        ERL_ASSERTM(test_result != nullptr, "Failed to test Gaussian Process.");

        ERL_INFO("Getting mean and gradients");
        test_result->GetMean(0, udf_pred_gp, true /*parallel*/);
        Eigen::VectorXb valid_gradients = test_result->GetGradient(0, udf_grads, true /*parallel*/);

        ERL_INFO("Normal diffusion");
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

        ERL_INFO("Softmin prediction");
        Eigen::VectorXd udf_pred_softmin =
            PredictUdfBySoftmin(test_pts, dataset.leftCols(options.num_surf_samples));

        ERL_INFO("Computing SDF from UDF");
        Eigen::VectorXd udf_grads_dot_normals =
            (udf_grads.array() * normals.array()).colwise().sum();
        Eigen::VectorXd signs_gp = udf_grads_dot_normals.unaryExpr(
            [](const double a) -> double { return (a < 0.0) ? -1.0 : 1.0; });
        Eigen::VectorXd sdf_pred_gp = udf_pred_gp.array() * signs_gp.array();
        Eigen::VectorXd sdf_pred_softmin = udf_pred_softmin.array() * signs_gt.array();

        ERL_INFO("Errors:");
        double sign_error = (signs_gp.array() != signs_gt.array()).cast<double>().sum();
        sign_error /= static_cast<double>(signs_gp.size());
        Eigen::VectorXd error_udf_gp = (udf_pred_gp - udf_gt).cwiseAbs();
        Eigen::VectorXd error_udf_softmin = (udf_pred_softmin - udf_gt).cwiseAbs();
        Eigen::VectorXd error_sdf_gp = (sdf_pred_gp - sdf_gt).cwiseAbs();
        Eigen::VectorXd error_sdf_softmin = (sdf_pred_softmin - sdf_gt).cwiseAbs();
        Eigen::VectorXd outside_mask = (signs_gt.array() > 0.0).cast<double>();
        const double outside_mask_sum = outside_mask.sum();
        ERL_INFO("Sign error: {}.", sign_error);
        ERL_INFO(
            "UDF error (GP): {}, {} (outside).",
            error_udf_gp.mean(),
            error_udf_gp.cwiseProduct(outside_mask).sum() / outside_mask_sum);
        ERL_INFO(
            "UDF error (Softmin): {}, {} (outside).",
            error_udf_softmin.mean(),
            error_udf_softmin.cwiseProduct(outside_mask).sum() / outside_mask_sum);
        ERL_INFO("SDF error (log-GP): {}.", error_sdf_gp.mean());
        ERL_INFO("SDF error (Softmin): {}.", error_sdf_softmin.mean());

        if (!options.visualize) { return; }

        PlplotFig fig(640, 640, true);

        PlplotFig::ShadesOpt shades_opt;
        shades_opt.SetXMin(options.min_x)
            .SetXMax(options.max_x)
            .SetYMin(options.min_y)
            .SetYMax(options.max_y);
        PlplotFig::ColorBarOpt color_bar_opt;
        color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM});

        DrawValueField(
            udf_pred_gp,
            sdf_gt.data(),
            "UDF Prediction (Log-GP)",
            "Pred",
            options.output_dir + "/log_sdf_regression_udf_pred_gp.png",
            "UDF Pred (Log-GP)",
            fig,
            shades_opt,
            color_bar_opt);

        DrawValueField(
            udf_pred_softmin,
            sdf_gt.data(),
            "UDF Prediction (Softmin)",
            "Pred",
            options.output_dir + "/log_sdf_regression_udf_pred_softmin.png",
            "UDF Pred (Softmin)",
            fig,
            shades_opt,
            color_bar_opt);

        DrawValueField(
            udf_gt,
            sdf_gt.data(),
            "UDF Ground Truth",
            "G.T.",
            options.output_dir + "/log_sdf_regression_udf_gt.png",
            "UDF GT",
            fig,
            shades_opt,
            color_bar_opt);

        DrawValueField(
            sdf_pred_gp,
            sdf_pred_gp.data(),
            "SDF Prediction (Log-GP)",
            "Pred",
            options.output_dir + "/log_sdf_regression_sdf_pred_gp.png",
            "SDF Pred (Log-GP)",
            fig,
            shades_opt,
            color_bar_opt);

        DrawValueField(
            sdf_pred_softmin,
            sdf_pred_softmin.data(),
            "SDF Prediction (Softmin)",
            "Pred",
            options.output_dir + "/log_sdf_regression_sdf_pred_softmin.png",
            "SDF Pred (Softmin)",
            fig,
            shades_opt,
            color_bar_opt);

        DrawValueField(
            sdf_gt,
            sdf_gt.data(),
            "SDF Ground Truth",
            "G.T.",
            options.output_dir + "/log_sdf_regression_sdf_gt.png",
            "SDF GT",
            fig,
            shades_opt,
            color_bar_opt);

        DrawValueField(
            udf_grads_dot_normals,
            nullptr,
            "UDF Gradients Dot Normals",
            "Dot",
            options.output_dir + "/log_sdf_regression_udf_pred_dot_normals.png",
            "UDF Gradients Dot Normals",
            fig,
            shades_opt,
            color_bar_opt);

        DrawValueField(
            error_udf_gp,
            sdf_gt.data(),
            "UDF Error (Log-GP)",
            "Abs. Err.",
            options.output_dir + "/log_sdf_regression_udf_error_gp.png",
            "UDF Error (Log-GP)",
            fig,
            shades_opt,
            color_bar_opt);

        DrawValueField(
            error_udf_softmin,
            sdf_gt.data(),
            "UDF Error (Softmin)",
            "Abs. Err.",
            options.output_dir + "/log_sdf_regression_udf_error_softmin.png",
            "UDF Error (Softmin)",
            fig,
            shades_opt,
            color_bar_opt);

        DrawValueField(
            error_sdf_gp,
            sdf_gt.data(),
            "SDF Error (Log-GP)",
            "Abs. Err.",
            options.output_dir + "/log_sdf_regression_sdf_error_gp.png",
            "SDF Error (Log-GP)",
            fig,
            shades_opt,
            color_bar_opt);

        DrawValueField(
            error_sdf_softmin,
            sdf_gt.data(),
            "SDF Error (Softmin)",
            "Abs. Err.",
            options.output_dir + "/log_sdf_regression_sdf_error_softmin.png",
            "SDF Error (Softmin)",
            fig,
            shades_opt,
            color_bar_opt);

        DrawValueField(
            var,
            nullptr,
            "Variance",
            "Var.",
            options.output_dir + "/log_sdf_regression_gp_variance.png",
            "GP Variance",
            fig,
            shades_opt,
            color_bar_opt);

        GridMapInfo2Dd grid_info_stride(
            Eigen::Vector2i(options.test_num_x / 20, options.test_num_y / 20),
            Eigen::Vector2d(options.min_x + 0.1, options.min_y + 0.1),
            Eigen::Vector2d(options.max_x - 0.1, options.max_y - 0.1));
        test_pts = grid_info_stride.GenerateMeterCoordinates(true /*c_stride*/);
        test_result = gp->Test(test_pts, true /*predict_gradient*/);
        udf_grads.resize(2, test_pts.cols());
        (void) test_result->GetGradient(0, udf_grads, true /*parallel*/);
        Eigen::VectorXd x = test_pts.row(0);
        Eigen::VectorXd y = test_pts.row(1);
        Eigen::VectorXd u = udf_grads.row(0);
        Eigen::VectorXd v = udf_grads.row(1);
        DrawVectorField(
            x,
            y,
            u,
            v,
            sdf_gt.data(),
            "UDF Gradient Prediction",
            options.output_dir + "/log_sdf_regression_udf_gradient_pred.png",
            "UDF Gradient Prediction",
            fig);

        buf.resize(test_pts.cols());
        normals.resize(2, test_pts.cols());
        test_result->GetMean(1, buf, true);
        normals.row(0) = buf.transpose();
        test_result->GetMean(2, buf, true);
        normals.row(1) = buf.transpose();
        normals.colwise().normalize();
        u = normals.row(0).transpose();
        v = normals.row(1).transpose();
        DrawVectorField(
            x,
            y,
            u,
            v,
            sdf_gt.data(),
            "Normal Diffusion",
            options.output_dir + "/log_sdf_regression_normal_diffusion.png",
            "Normal Diffusion",
            fig);

        const long stride = std::max(1l, options.num_surf_samples / 100l);
        u.resize((options.num_surf_samples + stride - 1l) / stride);
        v.resize(u.size());
        x.resize(u.size());
        y.resize(u.size());
        for (long i = 0, j = 0; i < options.num_surf_samples; i += stride, ++j) {
            x[j] = dataset(0, i);
            y[j] = dataset(1, i);
            const double norm = std::sqrt(x[j] * x[j] + y[j] * y[j]);
            u[j] = x[j] / norm;
            v[j] = y[j] / norm;
        }
        DrawVectorField(
            x,
            y,
            u,
            v,
            sdf_gt.data(),
            "Training Data",
            options.output_dir + "/log_sdf_regression_training_data.png",
            "Training Data",
            fig);

        cv::waitKey(options.hold ? 0 : 1000);
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
                        : (ERL_GP_SDF_ROOT_DIR "/config/demo/demo_log_sdf_regression.yaml");
        App app(options_file);
        app.Run();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
