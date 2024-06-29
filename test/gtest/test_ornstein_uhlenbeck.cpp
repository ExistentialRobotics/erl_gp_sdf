#include "covFnc.h"

#include "erl_common/test_helper.hpp"
#include "erl_covariance/ornstein_uhlenbeck.hpp"

#include <gtest/gtest.h>

constexpr double kOrnsteinUhlenbeckScale = 2.;
constexpr double kOrnsteinUhlenbeckAlpha = 1.;
using namespace erl;

struct Config {
    int d = 2;
    int n = 10;
    int m = 5;
    Eigen::MatrixXd mat_x_1;
    Eigen::MatrixXd mat_x_2;
    double sigma_output_scalar = 2.;
    Eigen::VectorXd sigma_x;
    std::shared_ptr<covariance::OrnsteinUhlenbeck<2>::Setting> ou_setting;

    Config()
        : ou_setting(std::make_shared<covariance::OrnsteinUhlenbeck<2>::Setting>()) {
        mat_x_1 = Eigen::MatrixXd::Random(d, n);
        mat_x_2 = Eigen::MatrixXd::Random(d, m);
        mat_x_1.array() *= 10;
        mat_x_2.array() *= 10;

        sigma_x = Eigen::VectorXd::Random(n);

        ou_setting->alpha = kOrnsteinUhlenbeckAlpha;
        ou_setting->scale = kOrnsteinUhlenbeckScale;
    }
};

struct TestEnvironment : public ::testing::Environment {
public:
    static Config
    GetConfig() {
        static Config config;
        return config;
    }

    void
    SetUp() override {
        GetConfig();
    }
};

// verify that my implementation is consistent with GPisMap's
TEST(OrnsteinUhlenbeck, ComputeKtrain) {
    const auto &config = TestEnvironment::GetConfig();
    const auto ornstein_uhlenbeck = std::make_shared<covariance::OrnsteinUhlenbeck2D>(config.ou_setting);

    std::cout << "==============" << std::endl;
    const auto [rows, cols] = covariance::Covariance::GetMinimumKtrainSize(config.n, 0, 2);
    Eigen::MatrixXd ans(rows, cols), gt;
    common::ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() {
        (void) ornstein_uhlenbeck->ComputeKtrain(ans, config.mat_x_1, Eigen::VectorXd::Constant(config.n, config.sigma_output_scalar));
    });
    common::ReportTime<std::chrono::microseconds>("gt", 10, false, [&]() {
        gt = OrnsteinUhlenbeck(config.mat_x_1, kOrnsteinUhlenbeckScale, config.sigma_output_scalar);
    });
    ASSERT_EIGEN_MATRIX_EQUAL("ComputeKtrain1", ans, gt);

    std::cout << "==============" << std::endl;
    common::ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() { (void) ornstein_uhlenbeck->ComputeKtrain(ans, config.mat_x_1, config.sigma_x); });
    common::ReportTime<std::chrono::microseconds>("gt", 10, false, [&]() { gt = OrnsteinUhlenbeck(config.mat_x_1, kOrnsteinUhlenbeckScale, config.sigma_x); });
    ASSERT_EIGEN_MATRIX_EQUAL("ComputeKtrain2", ans, gt);
}

TEST(OrnsteinUhlenbeck, ComputeKtest) {
    const auto &config = TestEnvironment::GetConfig();
    const auto ornstein_uhlenbeck = std::make_shared<covariance::OrnsteinUhlenbeck2D>(config.ou_setting);

    std::cout << "==============" << std::endl;
    Eigen::MatrixXd ans, gt;
    const auto [rows, cols] = covariance::Covariance::GetMinimumKtestSize(config.n, 0, 0, config.m);
    ans.resize(rows, cols);
    common::ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() { (void) ornstein_uhlenbeck->ComputeKtest(ans, config.mat_x_1, config.mat_x_2); });
    common::ReportTime<std::chrono::microseconds>("gt", 10, false, [&]() { gt = OrnsteinUhlenbeck(config.mat_x_1, config.mat_x_2, kOrnsteinUhlenbeckScale); });
    ASSERT_EIGEN_MATRIX_EQUAL("ComputeKtest", ans, gt);
}
