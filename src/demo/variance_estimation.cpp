#include "erl_common/plplot_fig.hpp"
#include "erl_gp_sdf/log_edf_gp.hpp"

using namespace erl::common;
using namespace erl::sdf_mapping;
constexpr int Dim = 2;
using Dtype = double;
using VectorD = Eigen::Vector<Dtype, Dim>;
using VectorX = Eigen::VectorXd;
using Gp = LogEdfGaussianProcess<Dtype>;

struct Options : Yamlable<Options> {
    double max_x = 2.0;
    double max_y = 2.0;
    double min_x = -2.0;
    double min_y = -2.0;
    int num_x = 201;
    int num_y = 201;
    double radius = 1.0;
    long num_samples = 1000;
    Eigen::Vector2d test_position = {1.5, 1.5};
    double softmin_temperature = 10.0;
    double var_x_min = 0.001;
    double var_x_max = 1.0;
    int num_var_x = 10;
    double var_y = 0.01;
    std::shared_ptr<Gp::Setting> gp = std::make_shared<Gp::Setting>();
    std::filesystem::path output_dir = "variance_estimation";
    bool show_images = true;
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
        ERL_YAML_SAVE_ATTR(node, options, num_x);
        ERL_YAML_SAVE_ATTR(node, options, num_y);
        ERL_YAML_SAVE_ATTR(node, options, radius);
        ERL_YAML_SAVE_ATTR(node, options, num_samples);
        ERL_YAML_SAVE_ATTR(node, options, test_position);
        ERL_YAML_SAVE_ATTR(node, options, softmin_temperature);
        ERL_YAML_SAVE_ATTR(node, options, var_x_min);
        ERL_YAML_SAVE_ATTR(node, options, var_x_max);
        ERL_YAML_SAVE_ATTR(node, options, num_var_x);
        ERL_YAML_SAVE_ATTR(node, options, var_y);
        ERL_YAML_SAVE_ATTR(node, options, gp);
        ERL_YAML_SAVE_ATTR(node, options, output_dir);
        ERL_YAML_SAVE_ATTR(node, options, show_images);
        return node;
    }

    static bool
    decode(const Node &node, Options &options) {
        if (!node.IsMap()) { return false; }
        ERL_YAML_LOAD_ATTR(node, options, max_x);
        ERL_YAML_LOAD_ATTR(node, options, max_y);
        ERL_YAML_LOAD_ATTR(node, options, min_x);
        ERL_YAML_LOAD_ATTR(node, options, min_y);
        ERL_YAML_LOAD_ATTR(node, options, num_x);
        ERL_YAML_LOAD_ATTR(node, options, num_y);
        ERL_YAML_LOAD_ATTR(node, options, radius);
        ERL_YAML_LOAD_ATTR(node, options, num_samples);
        ERL_YAML_LOAD_ATTR(node, options, test_position);
        ERL_YAML_LOAD_ATTR(node, options, softmin_temperature);
        ERL_YAML_LOAD_ATTR(node, options, var_x_min);
        ERL_YAML_LOAD_ATTR(node, options, var_x_max);
        ERL_YAML_LOAD_ATTR(node, options, num_var_x);
        ERL_YAML_LOAD_ATTR(node, options, var_y);
        if (!ERL_YAML_LOAD_ATTR(node, options, gp)) { return false; }
        ERL_YAML_LOAD_ATTR(node, options, output_dir);
        ERL_YAML_LOAD_ATTR(node, options, show_images);
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
        Eigen::Matrix2Xd points(2, options.num_samples);
        Eigen::VectorXd angles = Eigen::VectorXd::LinSpaced(options.num_samples, 0.0, 2 * M_PI);
        for (long i = 0; i < options.num_samples; ++i) {
            auto point = points.col(i);
            point.x() = options.radius * std::cos(angles[i]);
            point.y() = options.radius * std::sin(angles[i]);
        }
        return points;
    }

    [[nodiscard]] std::shared_ptr<Gp>
    Train(const Eigen::Matrix2Xd &points, const double var_x) const {
        options.gp->no_gradient_observation = true;
        auto gp = std::make_shared<Gp>(options.gp);
        gp->Reset(options.num_samples, 2, 1);

        auto &train_set = gp->GetTrainSet();
        train_set.x.topRows<2>() = points;
        train_set.y.leftCols<1>().setConstant(1);
        train_set.var_x.setConstant(var_x);
        train_set.var_y.setConstant(options.var_y);
        train_set.num_samples = options.num_samples;
        train_set.num_samples_with_grad = 0;
        train_set.grad_flag.setConstant(false);
        train_set.x_dim = 2;
        train_set.y_dim = 1;
        ERL_ASSERTM(gp->Train(), "Failed to train Gaussian Process.");
        return gp;
    }

    void
    EstimateVariance(
        const std::shared_ptr<Gp> &gp,
        const VectorD &test_position,
        const Dtype edf_pred,
        const bool compute_var_grad,
        Dtype *var) const {
        const Gp::TrainSet &train_set = gp->GetTrainSet();
        const long num_samples = train_set.num_samples;

        VectorX s(num_samples);
        Dtype s_sum = 0;
        VectorX z(num_samples);
        Eigen::Matrix<Dtype, Dim, Eigen::Dynamic> mat_v(Dim, num_samples);
        for (long k = 0; k < num_samples; ++k) {
            const VectorD v = test_position - train_set.x.col(k);
            Dtype &d = z[k];
            d = v.norm();  // distance to the training sample

            s[k] = std::max(1.0e-15, std::exp(-(d - edf_pred) * options.softmin_temperature));
            s_sum += s[k];

            mat_v.col(k) = v / d;
        }
        const Dtype inv_s_sum = 1.0f / s_sum;
        const Dtype sz = s.dot(z) * inv_s_sum;
        var[0] = 0.0f;  // var_sdf
        VectorX l(num_samples);
        VectorD g = VectorD::Zero();  // sum_i (l_i * v_i)
        VectorD f = VectorD::Zero();  // sum_i (s_i * v_i)
        for (long k = 0; k < num_samples; ++k) {
            Dtype &w = l[k];
            w = inv_s_sum * s[k] * (1.0f + options.softmin_temperature * (sz - z[k]));
            var[0] += w * w * train_set.var_x[k];
            // prepare for gradient variance
            g += w * mat_v.col(k);
            f += s[k] * mat_v.col(k);
        }

        if (!compute_var_grad) { return; }

        using SqMat = Eigen::Matrix<Dtype, Dim, Dim>;
        SqMat cov_grad = SqMat::Zero();
        const SqMat identity = SqMat::Identity();
        const double g_norm = g.norm();
        const VectorD g_normalized = g / g_norm;
        const SqMat grad_norm =
            (1.0f / g_norm) * (identity - g_normalized * g_normalized.transpose());
        for (long j = 0; j < num_samples; ++j) {
            const Dtype a = options.softmin_temperature * l[j];
            const Dtype b = options.softmin_temperature * s[j];
            const Dtype c = l[j] / z[j];
            const auto vj = mat_v.col(j);
            const VectorD v = (a + b + c) * vj - a * f - b * g;
            SqMat grad_j = vj * v.transpose();
            grad_j.diagonal().array() -= c;
            grad_j = grad_j * grad_norm;
            cov_grad += train_set.var_x[j] * (grad_j.transpose() * grad_j);
        }
        for (long i = 1; i <= Dim; ++i) { var[i] = cov_grad(i - 1, i - 1); }  // var_grad
    }

    [[nodiscard]] Eigen::VectorXd
    ComputeGmm(const Eigen::Matrix2Xd &points, const double var) const {
        GridMapInfo2Dd grid_info(
            Eigen::Vector2i(options.num_x, options.num_y),
            Eigen::Vector2d(options.min_x, options.min_y),
            Eigen::Vector2d(options.max_x, options.max_y));
        Eigen::Matrix2Xd test_pts = grid_info.GenerateMeterCoordinates(true /*c_stride*/);

        const double a = 1.0 / (2.0 * M_PI * std::sqrt(var));  // normalization constant
        const double b = -0.5 / var;                           // exponent coefficient
        Eigen::VectorXd gmm_values(test_pts.cols());
#pragma omp parallel for default(none) shared(test_pts, points, a, b, gmm_values)
        for (long i = 0; i < test_pts.cols(); ++i) {
            double sum = 0.0;
            auto p = test_pts.col(i);
            for (long j = 0; j < points.cols(); ++j) {
                const double sq_dist = (points.col(j) - p).squaredNorm();
                sum += std::exp(b * sq_dist);
            }
            gmm_values[i] = a * sum;  // GMM value at test point
        }
        gmm_values *= static_cast<double>(points.cols()) / gmm_values.sum();
        return gmm_values;
    }

    [[nodiscard]] Eigen::VectorXd
    ComputeSdfVar(const std::shared_ptr<Gp> &gp) const {
        ERL_INFO("Computing SDF variances");

        GridMapInfo2Dd grid_info(
            Eigen::Vector2i(options.num_x, options.num_y),
            Eigen::Vector2d(options.min_x, options.min_y),
            Eigen::Vector2d(options.max_x, options.max_y));
        Eigen::Matrix2Xd test_pts = grid_info.GenerateMeterCoordinates(true /*c_stride*/);
        auto test_result = gp->Test(test_pts, false);
        Eigen::VectorXd edf_pred(test_pts.cols());
        test_result->GetMean(0, edf_pred, true /*parallel*/);
        Eigen::VectorXd var(test_pts.cols());
#pragma omp parallel for default(none) shared(gp, test_pts, edf_pred, var)
        for (long i = 0; i < test_pts.cols(); ++i) {
            EstimateVariance(gp, test_pts.col(i), edf_pred[i], false, &var[i]);
        }
        return var;
    }

    void
    Run() const {
        std::filesystem::create_directories(options.output_dir);
        auto img_dir0 = options.output_dir / "variances_estimation";
        auto img_dir1 = options.output_dir / "variance_estimation";
        std::filesystem::create_directories(img_dir0);
        std::filesystem::create_directories(img_dir1);

        const auto dataset = GenerateDataset();
        Eigen::VectorXd var_xs =
            Eigen::VectorXd::LinSpaced(options.num_var_x, options.var_x_min, options.var_x_max);
        Eigen::VectorXd edf_pred(options.num_var_x);
        Eigen::Matrix3Xd gp_variances(3, options.num_var_x);
        Eigen::Matrix3Xd variances(3, options.num_var_x);
        std::vector<Eigen::VectorXd> sdf_var_values(options.num_var_x);
        std::vector<Eigen::VectorXd> gmm_values(options.num_var_x);

        for (int i = 0; i < options.num_var_x; ++i) {
            ERL_INFO("Training GP with var_x = {}", var_xs[i]);
            auto gp = Train(dataset, var_xs[i]);
            ERL_INFO("Testing GP");
            auto test_result = gp->Test(options.test_position, true);
            test_result->GetMean(0, 0, edf_pred[i]);
            test_result->GetMeanVariance(0, gp_variances(0, i));
            test_result->GetGradientVariance(0, gp_variances.col(i).tail<2>().data());
            EstimateVariance(gp, options.test_position, edf_pred[i], true, variances.col(i).data());
            sdf_var_values[i] = ComputeSdfVar(gp);
            gmm_values[i] = ComputeGmm(dataset, var_xs[i]);
        }

        Eigen::VectorXd angles = Eigen::VectorXd::LinSpaced(options.num_samples, 0.0, 2.0 * M_PI);
        Eigen::VectorXd xs(options.num_samples);
        Eigen::VectorXd ys(options.num_samples);
        auto generate_circle = [&angles, &xs, &ys](double r, double cx, double cy) {
            for (long k = 0; k < angles.size(); ++k) {
                xs[k] = r * std::cos(angles[k]) + cx;
                ys[k] = r * std::sin(angles[k]) + cy;
            }
        };
        PlplotFig fig(640, 640, true);
        auto clear_fig = [&fig, this]() {
            fig.Clear()
                .SetFontSize(0.0, 1.1)
                .SetMargin(0.12, 0.82, 0.15, 0.85)
                .SetAxisLimits(options.min_x, options.max_x, options.min_y, options.max_y);
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
        shades_opt.SetXMin(options.min_x)
            .SetXMax(options.max_x)
            .SetYMin(options.min_y)
            .SetYMax(options.max_y);
        PlplotFig::ColorBarOpt color_bar_opt;

        shades_opt.SetColorLevels(
            sdf_var_values.front().minCoeff(),
            sdf_var_values.back().maxCoeff(),
            127);
        color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM})
            .AddColorMap(0, shades_opt.color_levels, 10);
        for (int i = 0; i < options.num_var_x; ++i) {
            clear_fig();

            fig.SetAreaFillPattern(PlplotFig::AreaFillPattern::Solid)
                .SetColorMap(1, PlplotFig::ColorMap::Jet)
                .Shades(sdf_var_values[i].data(), options.num_x, options.num_y, true, shades_opt)
                .SetCurrentColor(PlplotFig::Color0::Black)
                .ColorBar(color_bar_opt);

            generate_circle(options.radius, 0, 0);
            fig.SetCurrentColor(PlplotFig::Color0::Black)
                .SetPenWidth(2)
                .DrawLine(static_cast<int>(xs.size()), xs.data(), ys.data())
                .SetPenWidth(1)
                .SetCurrentColor(PlplotFig::Color0::Black)
                .SetTitle(fmt::format("variance of dataset: {:.3e}", var_xs[i]).c_str());
            draw_axes();

            cv::Mat img = fig.ToCvMat();
            cv::imwrite(fmt::format("{}/{:03d}.png", img_dir0, i), img);
            if (options.show_images) {
                cv::imshow(fmt::format("variances_estimation_{:03d}", i), img);
            }
        }

        shades_opt.SetColorLevels(gmm_values.back().data(), options.num_x, options.num_y, 127);
        for (int i = 0; i < options.num_var_x; ++i) {
            clear_fig();
            shades_opt.SetColorLevels(gmm_values[i].data(), options.num_x, options.num_y, 127);

            fig.SetAreaFillPattern(PlplotFig::AreaFillPattern::Solid)
                .SetColorMap(1, PlplotFig::ColorMap::Jet)
                .Shades(gmm_values[i].data(), options.num_x, options.num_y, true, shades_opt);

            generate_circle(options.radius, 0, 0);
            fig.SetCurrentColor(PlplotFig::Color0::Black)
                .SetPenWidth(2)
                .DrawLine(static_cast<int>(xs.size()), xs.data(), ys.data())
                .SetPenWidth(1);
            generate_circle(edf_pred[i], options.test_position.x(), options.test_position.y());
            fig.SetCurrentColor(PlplotFig::Color0::Green)
                .SetPenWidth(2)
                .DrawLine(static_cast<int>(xs.size()), xs.data(), ys.data())
                .SetPenWidth(1);
            generate_circle(
                edf_pred[i] + std::sqrt(variances(0, i)),
                options.test_position.x(),
                options.test_position.y());

            fig.SetCurrentColor(PlplotFig::Color0::Red)
                .SetPenWidth(2)
                .DrawLine(static_cast<int>(xs.size()), xs.data(), ys.data())
                .SetPenWidth(1);

            fig.SetCurrentColor(PlplotFig::Color0::Black)
                .SetTitle(fmt::format("variance of dataset: {:.3e}", var_xs[i]).c_str());
            draw_axes();

            cv::Mat img = fig.ToCvMat();
            cv::imwrite(fmt::format("{}/{:03d}.png", img_dir1, i), img);
            if (options.show_images) {
                cv::imshow(fmt::format("variance_estimation_{:03d}", i), img);
            }
        }

        PlplotFig::LegendOpt legend_opt(2, {"SDF Variance", "GP Variance"});
        legend_opt.SetLegendBoxLineColor0(PlplotFig::Color0::Black)
            .SetBoxStyle(PL_LEGEND_BOUNDING_BOX)
            .SetPosition(
                PL_POSITION_TOP | PL_POSITION_LEFT | PL_POSITION_INSIDE | PL_POSITION_VIEWPORT)
            .SetTextColors({PlplotFig::Color0::Red, PlplotFig::Color0::Green})
            .SetStyles({PL_LEGEND_LINE, PL_LEGEND_LINE})
            .SetLineColors({PlplotFig::Color0::Red, PlplotFig::Color0::Green})
            .SetLineStyles({1, 1})
            .SetLineWidths({2.0, 2.0})
            .SetTextSpacing(2.2);
        fig.Clear()
            .SetFontSize(0.0, 1.1)
            .SetMargin(0.15, 0.85, 0.15, 0.85)
            .SetAxisLimits(
                options.var_x_min,
                options.var_x_max,
                0,
                variances.row(0).maxCoeff() * 1.1)
            .SetPenWidth(2)
            .SetCurrentColor(PlplotFig::Color0::Red)
            .DrawLine(options.num_var_x, var_xs.data(), Eigen::VectorXd(variances.row(0)).data())
            .SetPenWidth(1)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .DrawAxesBox(
                PlplotFig::AxisOpt().DrawTopRightEdge(),
                PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
            .SetAxisLabelX("Dataset Variance")
            .SetAxisLabelY("SDF Variance")
            .SetAxisLimits(options.var_x_min, options.var_x_max, 0.9, 1.1)
            .SetCurrentColor(PlplotFig::Color0::Green)
            .SetPenWidth(2)
            .DrawLine(options.num_var_x, var_xs.data(), Eigen::VectorXd(gp_variances.row(0)).data())
            .SetPenWidth(1)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .DrawAxesBox(
                PlplotFig::AxisOpt::Off(),
                PlplotFig::AxisOpt()
                    .DrawBottomLeftEdge(false)
                    .DrawBottomLeftTickLabels(false)
                    .DrawTopRightEdge()
                    .DrawTopRightTickLabels()
                    .DrawPerpendicularTickLabels())
            .SetAxisLabelY("GP Variance", true)
            .Legend(legend_opt);
        cv::Mat img = fig.ToCvMat();
        cv::imwrite(fmt::format("{}/variance_estimation_var_sdf.png", options.output_dir), img);
        if (options.show_images) { cv::imshow("variance_estimation: var_sdf", img); }

        legend_opt.SetNumLegend(4)
            .SetTexts(
                {"Gradient_x Variance",
                 "Gradient_y Variance",
                 "GP Gradient_x Variance",
                 "GP Gradient_y Variance"})
            .SetLegendBoxLineColor0(PlplotFig::Color0::Black)
            .SetBoxStyle(PL_LEGEND_BOUNDING_BOX)
            .SetPosition(
                PL_POSITION_TOP | PL_POSITION_LEFT | PL_POSITION_INSIDE | PL_POSITION_VIEWPORT)
            .SetTextColors(
                {PlplotFig::Color0::Red,
                 PlplotFig::Color0::Red,
                 PlplotFig::Color0::Green,
                 PlplotFig::Color0::Green})
            .SetStyles({PL_LEGEND_LINE, PL_LEGEND_LINE, PL_LEGEND_LINE, PL_LEGEND_LINE})
            .SetLineColors(
                {PlplotFig::Color0::Red,
                 PlplotFig::Color0::Red,
                 PlplotFig::Color0::Green,
                 PlplotFig::Color0::Green})
            .SetLineStyles({1, 2, 1, 2})
            .SetLineWidths({2.0, 2.0, 2.0, 2.0})
            .SetTextSpacing(2.2);
        fig.Clear()
            .SetFontSize(0.0, 1.1)
            .SetMargin(0.15, 0.85, 0.15, 0.85)
            .SetAxisLimits(
                options.var_x_min,
                options.var_x_max,
                0,
                variances.row(1).maxCoeff() * 1.1)
            .SetCurrentColor(PlplotFig::Color0::Red)
            .SetPenWidth(2)
            .SetLineStyle(1)
            .DrawLine(options.num_var_x, var_xs.data(), Eigen::VectorXd(variances.row(1)).data())
            .SetLineStyle(2)
            .DrawLine(options.num_var_x, var_xs.data(), Eigen::VectorXd(variances.row(2)).data())
            .SetPenWidth(1)
            .SetLineStyle(1)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .DrawAxesBox(
                PlplotFig::AxisOpt().DrawTopRightEdge(),
                PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
            .SetAxisLabelX("Dataset Variance")
            .SetAxisLabelY("Gradient Variance")
            .SetAxisLimits(options.var_x_min, options.var_x_max, 3.9e4, 4.1e4)
            .SetCurrentColor(PlplotFig::Color0::Green)
            .SetPenWidth(2)
            .SetLineStyle(1)
            .DrawLine(options.num_var_x, var_xs.data(), Eigen::VectorXd(gp_variances.row(1)).data())
            .SetLineStyle(2)
            .DrawLine(options.num_var_x, var_xs.data(), Eigen::VectorXd(gp_variances.row(1)).data())
            .SetPenWidth(1)
            .SetLineStyle(1)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .DrawAxesBox(
                PlplotFig::AxisOpt::Off(),
                PlplotFig::AxisOpt()
                    .DrawBottomLeftEdge(false)
                    .DrawBottomLeftTickLabels(false)
                    .DrawTopRightEdge()
                    .DrawTopRightTickLabels()
                    .DrawPerpendicularTickLabels())
            .SetAxisLabelY("GP Variance", true)
            .Legend(legend_opt);
        img = fig.ToCvMat();
        cv::imwrite(fmt::format("{}/variance_estimation_var_grad.png", options.output_dir), img);
        if (options.show_images) { cv::imshow("variance_estimation: var_grad", img); }
        if (options.show_images) { cv::waitKey(); }
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
                        : (ERL_GP_SDF_ROOT_DIR "/config/demo/demo_variance_estimation.yaml");
        App app(options_file);
        app.Run();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
