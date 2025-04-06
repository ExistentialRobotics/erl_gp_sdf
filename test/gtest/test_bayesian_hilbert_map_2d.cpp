#include "erl_common/grid_map_info.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_covariance/radial_bias_function.hpp"
#include "erl_geometry/lidar_2d.hpp"
#include "erl_geometry/space_2d.hpp"
#include "erl_sdf_mapping/bayesian_hilbert_map.hpp"

template<typename Dtype>
std::vector<Eigen::Vector3<Dtype>>
GenerateTrajectory(const int n = 50, const int repeats = 1) {
    const int total = n * repeats;
    std::vector<Eigen::Vector3<Dtype>> trajectory;
    trajectory.reserve(total);
    const Dtype angle_step = 2 * M_PI / n;
    constexpr Dtype a = 1.6;
    constexpr Dtype b = 1.2;
    Dtype angle = 0;
    for (int i = 0; i < n; ++i) {
        trajectory.emplace_back(a * std::cos(angle), b * std::sin(angle), angle);
        angle += angle_step;
    }
    for (int r = 1; r < repeats; ++r) { trajectory.insert(trajectory.end(), trajectory.begin(), trajectory.begin() + n); }
    return trajectory;
}

using namespace erl::common;
using namespace erl::geometry;
using namespace erl::covariance;
using namespace erl::sdf_mapping;

std::shared_ptr<Space2D>
GenerateSpace() {

    std::vector<Eigen::Vector2d> ordered_object_vertices;

    int n = 50;
    Eigen::Matrix2Xd pts_circle1(2, n);
    double angle_step = 2 * M_PI / static_cast<double>(n);
    double angle = 0;
    for (int i = 0; i < n; ++i) {
        constexpr double r1 = 0.3;
        constexpr double x1 = -1.0;
        constexpr double y1 = 0.2;
        pts_circle1.col(i) << r1 * std::cos(angle) + x1, r1 * std::sin(angle) + y1;
        angle += angle_step;
    }

    n = 100;
    Eigen::Matrix2Xd pts_circle2(2, n);
    angle_step = 2 * M_PI / static_cast<double>(n);
    angle = 0;
    for (int i = 0; i < n; ++i) {
        constexpr double r2 = 0.8;
        constexpr double x2 = 0.3;
        constexpr double y2 = 0.0;
        pts_circle2.col(i) << r2 * std::cos(angle) + x2, r2 * std::sin(angle) + y2;
        angle += angle_step;
    }

    n = 40;
    Eigen::Matrix2Xd pts_box(2, n * 4);
    constexpr double half_size = 2.0;
    const double step = half_size / static_cast<double>(n);
    double v = -half_size;
    for (int i = 0; i < n; ++i) {
        int j = i;
        pts_box.col(j) << -half_size, v;
        j += n;
        pts_box.col(j) << v, half_size;
        j += n;
        pts_box.col(j) << half_size, -v;
        j += n;
        pts_box.col(j) << -v, -half_size;
        v += step;
    }

    Eigen::VectorXb outside_flags(3);
    outside_flags << true, true, false;

    return std::make_shared<Space2D>({pts_circle1, pts_circle2, pts_box}, outside_flags);
}

template<typename Dtype>
void
TestBayesianHilbertMap2D(Lidar2D& lidar, const int grid_size, const Dtype rbf_gamma, const bool diagonal_sigma) {
    std::vector<Eigen::Vector3<Dtype>> trajectory = GenerateTrajectory<Dtype>(50, 1);

    std::shared_ptr<BayesianHilbertMapSetting> bhm_setting = std::make_shared<BayesianHilbertMapSetting>();
    bhm_setting->diagonal_sigma = diagonal_sigma;

    auto kernel_setting = std::make_shared<typename Covariance<Dtype>::Setting>();
    kernel_setting->scale = 1.0 / rbf_gamma;
    std::shared_ptr<Covariance<Dtype>> kernel = std::make_shared<RadialBiasFunction<Dtype, 2>>(kernel_setting);

    Aabb<Dtype, 2> map_boundary(Eigen::Vector2<Dtype>(-5.0, -5.0), Eigen::Vector2<Dtype>(5.0, 5.0));
    GridMapInfo2D<Dtype> grid_map(Eigen::Vector2i(grid_size, grid_size), map_boundary.min(), map_boundary.max());
    Eigen::Matrix2Xd hinged_points = grid_map.GenerateMeterCoordinates(true);

    BayesianHilbertMap<Dtype, 2> bhm(bhm_setting, kernel, hinged_points, map_boundary, 0);
}

TEST(BayesianHilbertMap, 2D) {
    std::shared_ptr<Lidar2D::Setting> lidar_setting = std::make_shared<Lidar2D::Setting>();
    lidar_setting->max_angle = 135.0 / 180.0 * M_PI;   // 135 degrees
    lidar_setting->min_angle = -135.0 / 180.0 * M_PI;  // -135 degrees
    lidar_setting->num_lines = 135;
    std::shared_ptr<Space2D> space = GenerateSpace();
    Lidar2D lidar(lidar_setting, space);
}
