#include "GPisMap.h"

#include "erl_common/test_helper.hpp"
#include "erl_gaussian_process/noisy_input_gp.hpp"
#include "erl_geometry/gazebo_room.hpp"
#include "erl_gp_sdf/gpis/gpis_map_2d.hpp"

#include <filesystem>

void
PrintStatusOfErlGpisMap(const erl::gp_sdf::gpis::GpisMap2D &gpis_map, std::ostream &os = std::cout) {
    if (&os == &std::cout) {
        erl::common::Logging::Info("Tree Structure of ERL-GpisMap2D:");
    } else {
        erl::common::Logging::Info("Print tree structure of ERL-GpisMap2D to ostream ...");
    }
    if (gpis_map.GetQuadtree() == nullptr) {
        os << "null" << std::endl;
    } else {
        gpis_map.GetQuadtree()->Print(os);
    }
}

void
PrintStatusOfGpisMap(const GPisMap &gpis_map, std::ostream &os = std::cout) {
    if (&os == &std::cout) {
        erl::common::Logging::Info("Tree Structure of GPisMap:");
    } else {
        erl::common::Logging::Info("Print tree structure of GPisMap to ostream ...");
    }
    os << gpis_map.m_tree_;
}

TEST(ERL_SDF_MAPPING, GpisMap2D) {
    std::filesystem::path path = __FILE__;
    path = path.parent_path() / "double/train.dat";
    auto train_data_loader = erl::geometry::GazeboRoom2D::TrainDataLoader(path.string().c_str());

    GPisMap gpm;
    erl::gp_sdf::gpis::GpisMap2D gpis_map;
    auto gpis_map_setting = gpis_map.GetSetting();
    gpis_map_setting->num_threads = std::thread::hardware_concurrency();
    gpis_map_setting->update_gp_sdf->offset_distance = GPISMAP_FBIAS;
    gpis_map_setting->gp_theta->gp->kernel_type = "OrnsteinUhlenbeck1D";
    gpis_map_setting->gp_theta->gp->kernel->x_dim = 1;
    std::cout << *gpis_map.GetSetting() << std::endl;

    std::cout.precision(5);
    std::cout << std::scientific;

    double t_train_gt = 0;
    double t_train_ans = 0;
    int cnt = 0;
    std::stringstream ss;
    for (auto df: train_data_loader) {
        ss.str(std::string());
        ss << "GPisMap-Train[" << cnt << ']';
        t_train_gt += erl::common::ReportTime<std::chrono::microseconds>(ss.str().c_str(), 0, true, [&]() {
            gpm.Update(df.angles.data(), df.distances.data(), static_cast<int>(df.angles.size()), df.pose_matlab);
        });
        ss.str(std::string());
        ss << "GpisMap2D-Train[" << cnt << ']';
        t_train_ans += erl::common::ReportTime<std::chrono::microseconds>(
            ss.str().c_str(),
            0,     // no repeat
            true,  // print all repetitions
            [&]() { gpis_map.Update(df.angles, df.distances, df.pose_numpy); });
        cnt++;
        std::cout << "===============================================" << std::endl;
    }

    t_train_gt /= static_cast<double>(train_data_loader.size());
    t_train_ans /= static_cast<double>(train_data_loader.size());

    erl::common::Logging::Info("Average training time:");
    std::cout << "ERL-GpisMap2D: " << t_train_ans << " us" << std::endl << "GPisMap: " << t_train_gt << " us" << std::endl;

    std::ofstream fs;
    fs.open("GpisMap2D-QuadTree-gt.txt", std::ios::out);
    PrintStatusOfGpisMap(gpm, fs);
    fs.close();

    fs.open("GpisMap2D-QuadTree-ans.txt", std::ios::out);
    PrintStatusOfErlGpisMap(gpis_map, fs);
    fs.close();

    // Check m_train_buffer_
    auto &train_buffer = gpis_map.GetGpTheta()->GetTrainBuffer();
    ASSERT_EIGEN_VECTOR_EQUAL("m_gp_theta_->m_train_buffer_.vec_angles", train_buffer.vec_angles, gpm.m_obs_theta_);
    ASSERT_EIGEN_VECTOR_EQUAL("m_gp_theta_->m_train_buffer_.vec_ranges", train_buffer.vec_ranges, gpm.m_obs_range_);
    ASSERT_EIGEN_VECTOR_EQUAL("m_gp_theta_->m_train_buffer_.vec_mapped_distances", train_buffer.vec_mapped_distances, gpm.m_obs_f_);

    EXPECT_EQ(train_buffer.position.x(), gpm.m_pose_tr_[0]) << "m_gp_theta_->m_train_buffer_.position.x()";
    EXPECT_EQ(train_buffer.position.y(), gpm.m_pose_tr_[1]) << "m_gp_theta_->m_train_buffer_.position.y()";
    EXPECT_EQ(train_buffer.rotation(0, 0), gpm.m_pose_r_[0]) << "m_gp_theta_->m_train_buffer_.rotation(0, 0)";
    EXPECT_EQ(train_buffer.rotation(1, 0), gpm.m_pose_r_[1]) << "m_gp_theta_->m_train_buffer_.rotation(1, 0)";
    EXPECT_EQ(train_buffer.rotation(0, 1), gpm.m_pose_r_[2]) << "m_gp_theta_->m_train_buffer_.rotation(0, 1)";
    EXPECT_EQ(train_buffer.rotation(1, 1), gpm.m_pose_r_[3]) << "m_gp_theta_->m_train_buffer_.rotation(1, 1)";
    EXPECT_EQ(train_buffer.max_distance, gpm.m_range_obs_max_) << "m_gp_theta_->m_train_buffer_.max_distance";

    Eigen::MatrixXd mat_xy_local_ans = train_buffer.mat_xy_local;
    Eigen::MatrixXd mat_xy_local_gt = Eigen::Map<Eigen::MatrixXd>(gpm.m_obs_xy_local_.data(), 2, static_cast<long>(gpm.m_obs_xy_local_.size()) / 2);
    ASSERT_EIGEN_MATRIX_EQUAL("m_gp_theta_->m_train_buffer_.mat_xy_local", mat_xy_local_ans, mat_xy_local_gt);

    Eigen::MatrixXd mat_xy_global_ans = train_buffer.mat_xy_global;
    Eigen::MatrixXd mat_xy_global_gt = Eigen::Map<Eigen::MatrixXd>(gpm.m_obs_xy_global_.data(), 2, static_cast<long>(gpm.m_obs_xy_global_.size()) / 2);
    ASSERT_EIGEN_MATRIX_EQUAL("m_gp_theta_->m_train_buffer_.mat_xy_global", mat_xy_global_ans, mat_xy_global_gt);

    // Check regress observation
    const auto gp_theta_setting = gpis_map.GetGpTheta()->GetSetting();
    EXPECT_EQ(gp_theta_setting->overlap_size, gpm.m_gpo_->param.overlap) << "overlap_size";
    EXPECT_EQ(gp_theta_setting->group_size - gp_theta_setting->overlap_size, gpm.m_gpo_->param.group_size) << "group_size";
    EXPECT_EQ(gp_theta_setting->boundary_margin, gpm.m_gpo_->param.margin) << "boundary_margin";
    EXPECT_EQ(gp_theta_setting->gp->kernel->scale, gpm.m_gpo_->param.scale) << "gp->kernel->scale";
    EXPECT_EQ(gp_theta_setting->gp->kernel->x_dim, 1) << "gp->kernel->x_dim";
    EXPECT_EQ(gp_theta_setting->sensor_range_var, gpm.m_gpo_->param.noise) << "gp_theta->sensor_range_var";
    EXPECT_EQ(gp_theta_setting->train_buffer->valid_angle_min, gpm.m_setting_.angle_obs_limit[0]) << "valid_angle_min";
    EXPECT_EQ(gp_theta_setting->train_buffer->valid_angle_max, gpm.m_setting_.angle_obs_limit[1]) << "valid_angle_max";
    EXPECT_EQ(gp_theta_setting->train_buffer->valid_range_min, 0.2) << "valid_range_min";
    EXPECT_EQ(gp_theta_setting->train_buffer->valid_range_max, 30.0) << "valid_range_max";
    ASSERT_STD_VECTOR_EQUAL("gpm.m_gpo_->range", gpis_map.GetGpTheta()->GetPartitions(), gpm.m_gpo_->range);

    auto gps = gpis_map.GetGpTheta()->GetGps();
    for (size_t i = 0; i < gps.size(); ++i) {
        ss.str(std::string());
        ss << "gpm.m_gpo_->gps[" << i << "]->x";
        long n = gpm.m_gpo_->gps[i]->x.cols();
        Eigen::MatrixXd mat_x = gps[i]->GetTrainInputSamplesBuffer().leftCols(n);
        ASSERT_EIGEN_MATRIX_EQUAL(ss.str(), mat_x, gpm.m_gpo_->gps[i]->x);
    }

    for (size_t i = 0; i < gps.size(); ++i) {
        ss.str(std::string());
        ss << "gpm.m_gpo_->gps[" << i << "]->K";
        long n = gpm.m_gpo_->gps[i]->L.rows();
        Eigen::MatrixXd mat_k = gps[i]->GetKtrain().topLeftCorner(n, n);
        ASSERT_EIGEN_MATRIX_EQUAL(ss.str(), mat_k, gpm.m_gpo_->gps[i]->K);
    }

    for (size_t i = 0; i < gps.size(); ++i) {
        ss.str(std::string());
        ss << "gpm.m_gpo_->gps[" << i << "]->L";
        long n = gpm.m_gpo_->gps[i]->L.rows();
        Eigen::MatrixXd mat_l = gps[i]->GetCholeskyDecomposition().topLeftCorner(n, n);
        ASSERT_EIGEN_MATRIX_EQUAL(ss.str(), mat_l, gpm.m_gpo_->gps[i]->L);
    }

    for (size_t i = 0; i < gps.size(); ++i) {
        ss.str(std::string());
        ss << "gpm.m_gpo_->gps[" << i << "]->alpha";
        Eigen::VectorXd alpha = gps[i]->GetTrainOutputSamplesBuffer().head(gpm.m_gpo_->gps[i]->alpha.size());
        ASSERT_EIGEN_VECTOR_EQUAL(ss.str(), alpha, gpm.m_gpo_->gps[i]->alpha);
    }

    // Check UpdateSurfacePoints & AddNewSurfaceSamples
    std::vector<std::shared_ptr<erl::gp_sdf::gpis::GpisNode2D>> nodes_ans;
    gpis_map.GetQuadtree()->CollectNodesOfTypeInArea(0, gpis_map.GetQuadtree()->GetArea(), nodes_ans);
    std::vector<std::shared_ptr<Node>> nodes_gt;
    gpm.m_tree_->QueryRange(gpm.m_tree_->m_boundary_, nodes_gt);
    ASSERT_EQ(nodes_ans.size(), nodes_gt.size()) << "Number of Nodes stored in the QuadTree";

    for (size_t j = 0; j < nodes_ans.size(); ++j) {
        ss.str(std::string());
        auto &node_ans = nodes_ans[j];
        auto &node_gt = nodes_gt[j];
        auto node_data_ans = node_ans->node_data;

        ss << "nodes_ans[" << j << "]position.x";
#ifdef NDEBUG
        EXPECT_NEAR(node_ans->position.x(), node_gt->GetPos().x, 1.e-10) << ss.str();
#else
        EXPECT_EQ(node_ans->position.x(), node_gt->GetPos().x) << ss.str();
#endif

        ss.str(std::string());
        ss << "nodes_ans[" << j << "]position.y";
#ifdef NDEBUG
        ASSERT_NEAR(node_ans->position.y(), node_gt->GetPos().y, 1.e-10) << ss.str();
#else
        ASSERT_EQ(node_ans->position.y(), node_gt->GetPos().y) << ss.str();
#endif
        ss.str(std::string());
        ss << "nodes_ans[" << j << "]distance";
#ifdef NDEBUG
        ASSERT_NEAR(node_data_ans->distance, node_gt->m_val_, 1.e-10) << ss.str();
#else
        ASSERT_EQ(node_data_ans->distance, node_gt->m_val_) << ss.str();
#endif

        ss.str(std::string());
        ss << "nodes_ans[" << j << "]gradient.x";
#ifdef NDEBUG
        ASSERT_NEAR(node_data_ans->gradient.x(), node_gt->m_grad_.x, 1.e-10) << ss.str();
#else
        ASSERT_EQ(node_data_ans->gradient.x(), node_gt->m_grad_.x) << ss.str();
#endif

        ss.str(std::string());
        ss << "nodes_ans[" << j << "]gradient.y";
#ifdef NDEBUG
        ASSERT_NEAR(node_data_ans->gradient.y(), node_gt->m_grad_.y, 1.e-10) << ss.str();
#else
        ASSERT_EQ(node_data_ans->gradient.y(), node_gt->m_grad_.y) << ss.str();
#endif

        ss.str(std::string());
        ss << "nodes_ans[" << j << "]var_position";
#ifdef NDEBUG
        ASSERT_NEAR(node_data_ans->var_position, node_gt->GetPosNoise(), 1.e-10) << ss.str();
#else
        ASSERT_EQ(node_data_ans->var_position, node_gt->GetPosNoise()) << ss.str();
#endif

        ss.str(std::string());
        ss << "nodes_ans[" << j << "]var_gradient";
#ifdef NDEBUG
        ASSERT_NEAR(node_data_ans->var_gradient, node_gt->GetGradNoise(), 1.e-10) << ss.str();
#else
        ASSERT_EQ(node_data_ans->var_gradient, node_gt->GetGradNoise()) << ss.str();
#endif
    }

    // Check UpdateGpSdf
    std::vector<std::shared_ptr<const erl::gp_sdf::gpis::IncrementalQuadtree>> clusters_ans;
    std::vector<QuadTree *> clusters_gt;
    gpis_map.GetQuadtree()->CollectTrees(
        [](const std::shared_ptr<const erl::gp_sdf::gpis::IncrementalQuadtree> &tree) -> bool { return tree->GetData<void>() != nullptr; },
        clusters_ans);
    gpm.m_tree_->CollectTrees([](const QuadTree *tree) -> bool { return tree->m_gp_ != nullptr; }, clusters_gt);
    ASSERT_EQ(clusters_ans.size(), clusters_gt.size()) << "Number of QuadTrees having GP";

    int i = 0;
    for (auto &cluster: clusters_ans) {
        auto itr = std::find_if(clusters_gt.begin(), clusters_gt.end(), [&](QuadTree *tree) -> bool {
            const auto c = tree->GetCenter();
            return (cluster->GetArea().center - Eigen::Vector2d{c.x, c.y}).squaredNorm() < 1.e-10;
        });
        ASSERT_TRUE(itr != clusters_gt.end()) << "failed to find the corresponding GP for the cluster at " << cluster->GetArea().center;

        auto gp_ans = cluster->GetData<erl::gaussian_process::NoisyInputGaussianProcess>();
        auto gp_gt = (*itr)->m_gp_;

        ss.str(std::string());
        ss << "cluster_gp[" << i << "]->m_trained_";
        EXPECT_EQ(gp_ans->IsTrained(), gp_gt->m_trained_) << ss.str();

        ss.str(std::string());
        ss << "cluster_gp[" << i << "]->m_x_";
        Eigen::MatrixXd mat_x = gp_ans->GetTrainInputSamplesBuffer().leftCols(gp_gt->m_x_.cols());
#ifdef NDEBUG
        ASSERT_EIGEN_MATRIX_NEAR(ss.str(), mat_x, gp_gt->m_x_, 1.e-10);
#else
        ASSERT_EIGEN_MATRIX_EQUAL(ss.str(), mat_x, gp_gt->m_x_);
#endif

        ss.str(std::string());
        ss << "cluster_gp[" << i << "]->m_sigx_";
        Eigen::VectorXd sigx = gp_ans->GetTrainInputSamplesVarianceBuffer().head(gp_gt->m_sigx_.size());
#ifdef NDEBUG
        ASSERT_EIGEN_VECTOR_NEAR(ss.str(), sigx, gp_gt->m_sigx_, 1.e-10);
#else
        ASSERT_EIGEN_VECTOR_EQUAL(ss.str(), sigx, gp_gt->m_sigx_);
#endif

        ss.str(std::string());
        ss << "cluster_gp[" << i << "]->m_sigy_";
        Eigen::VectorXd sigy = gp_ans->GetTrainOutputValueSamplesVarianceBuffer().head(gp_gt->m_sigx_.size());
        Eigen::VectorXd sigy_gt = Eigen::VectorXd::Constant(sigx.size(), gp_gt->m_param_.noise);
        ASSERT_EIGEN_VECTOR_EQUAL(ss.str(), sigy, sigy_gt);

        ss.str(std::string());
        ss << "cluster_gp[" << i << "]->m_siggrad_";
        Eigen::VectorXd siggrad = gp_ans->GetTrainOutputGradientSamplesVarianceBuffer().head(gp_gt->m_siggrad_.size());
#ifdef NDEBUG
        ASSERT_EIGEN_VECTOR_NEAR(ss.str(), siggrad, gp_gt->m_siggrad_, 1.e-10);
#else
        ASSERT_EIGEN_VECTOR_EQUAL(ss.str(), siggrad, gp_gt->m_siggrad_);
#endif

        ss.str(std::string());
        ss << "cluster_gp[" << i << "]->m_gradflag_";
        Eigen::VectorXb grad_flag_ans = gp_ans->GetTrainGradientFlagsBuffer().head(gp_gt->m_gradflag_.size());
        Eigen::VectorXb grad_flag_gt(gp_gt->m_gradflag_.size());
        for (long j = 0; j < grad_flag_gt.size(); ++j) { grad_flag_gt[j] = gp_gt->m_gradflag_[j] > 0.5; }
        ASSERT_EIGEN_VECTOR_EQUAL(ss.str(), grad_flag_ans, grad_flag_gt);

        ASSERT_EQ(gp_ans->GetSetting()->kernel->scale, gp_gt->m_param_.scale) << "cluster_gp[" << i << "]->m_param_.scale";
        ss.str(std::string());
        ss << "cluster_gp[" << i << "]->m_k_";
        Eigen::MatrixXd ktrain = gp_ans->GetKtrain().topLeftCorner(gp_gt->m_k_.rows(), gp_gt->m_k_.cols());
#ifdef NDEBUG
        ASSERT_EIGEN_MATRIX_NEAR(ss.str(), ktrain, gp_gt->m_k_, 1.e-10);
#else
        ASSERT_EIGEN_MATRIX_EQUAL(ss.str(), ktrain, gp_gt->m_k_);
#endif

        ss.str(std::string());
        ss << "cluster_gp[" << i << "]->m_l_";
        Eigen::MatrixXd mat_l = gp_ans->GetCholeskyDecomposition().topLeftCorner(gp_gt->m_l_.rows(), gp_gt->m_l_.cols());
#ifdef NDEBUG
        ASSERT_EIGEN_MATRIX_NEAR(ss.str(), mat_l, gp_gt->m_l_, 1.e-10);
#else
        ASSERT_EIGEN_MATRIX_EQUAL(ss.str(), mat_l, gp_gt->m_l_);
#endif

        ss.str(std::string());
        ss << "cluster_gp[" << i << "]->m_y_ (sdf)";
        Eigen::VectorXd y_ans = gp_ans->GetTrainOutputSamplesBuffer().head(mat_x.cols());
        Eigen::VectorXd y_gt = gp_gt->m_y_.head(y_ans.size());
        ASSERT_EIGEN_VECTOR_EQUAL(ss.str(), y_ans, y_gt);

        ss.str(std::string());
        ss << "cluster_gp[" << i << "]->m_y_ (gradient)";
        long num_valid_grad = gp_ans->GetNumTrainSamplesWithGrad();
        Eigen::Matrix2Xd grad_ans(2, num_valid_grad);
        for (long j = 0, k = 0; j < mat_x.cols(); ++j) {
            if (!grad_flag_ans[j]) { continue; }
            grad_ans.col(k++) = gp_ans->GetTrainOutputGradientSamplesBuffer().col(j);
        }
        Eigen::Matrix2Xd grad_gt = gp_gt->m_y_.tail(2 * num_valid_grad).reshaped(num_valid_grad, 2).transpose();
#ifdef NDEBUG
        ASSERT_EIGEN_MATRIX_NEAR(ss.str(), grad_ans, grad_gt, 1.e-10);
#else
        ASSERT_EIGEN_MATRIX_EQUAL(ss.str(), grad_ans, grad_gt);
#endif

        ss.str(std::string());
        ss << "cluster_gp[" << i << "]->m_alpha_";
        Eigen::VectorXd alpha_ans = gp_ans->GetAlpha().head(gp_gt->m_alpha_.size());
#ifdef NDEBUG
        ASSERT_EIGEN_VECTOR_NEAR(ss.str(), alpha_ans, gp_gt->m_alpha_, 1.e-10);
#else
        ASSERT_EIGEN_VECTOR_EQUAL(ss.str(), alpha_ans, gp_gt->m_alpha_);
#endif
        i++;
    }

    // Check Test
    path = path.parent_path() / "test.dat";
    auto df = erl::geometry::GazeboRoom2D::TestDataFrame(path.string().c_str());
    Eigen::VectorXd distance_ans(df.positions.cols()), distance_variance_ans(df.positions.cols());
    Eigen::Matrix2Xd gradient_ans(2, df.positions.cols()), gradient_variance_ans(2, df.positions.cols());
    double t_test_ans, t_test_gt;
    t_test_gt = erl::common::ReportTime<std::chrono::milliseconds>("GPisMap-test", 5, false, [&]() {
        gpm.Test(df.positions.data(), df.dim, df.num_queries, df.out_buf.data());
    });
    std::cout << "\t---------------------------------" << std::endl;
    t_test_ans = erl::common::ReportTime<std::chrono::milliseconds>("ERL-GpisMap2D-test", 5, false, [&]() {
        gpis_map.Test(df.positions, distance_ans, gradient_ans, distance_variance_ans, gradient_variance_ans);
    });
    std::cout << "\t---------------------------------" << std::endl;

    Eigen::VectorXd distance_gt, distance_variance_gt;
    Eigen::Matrix2Xd gradient_gt, gradient_variance_gt;
    df.Extract(distance_gt, gradient_gt, distance_variance_gt, gradient_variance_gt);
    gradient_gt = gradient_gt.colwise().normalized();
    ASSERT_EIGEN_VECTOR_NEAR("distance", distance_ans, distance_gt, 1.e-10);
    ASSERT_EIGEN_MATRIX_NEAR("gradient", gradient_ans, gradient_gt, 1.e-10);
    ASSERT_EIGEN_VECTOR_NEAR("distanceVariance", distance_variance_ans, distance_variance_gt, 1.e-10);
    ASSERT_EIGEN_MATRIX_NEAR("gradientVariance", gradient_variance_ans, gradient_variance_gt, 1.e-10);

    erl::common::Logging::Info("Average training time:");
    std::cout << "ERL-GpisMap2D: " << t_train_ans << " us" << std::endl << "GPisMap: " << t_train_gt << " us" << std::endl;
    erl::common::Logging::Info("Average testing time:");
    std::cout << "ERL-GpisMap2D: " << t_test_ans << " ms" << std::endl << "GPisMap: " << t_test_gt << " ms" << std::endl;

    // erl::common::SaveBinaryFile<double>("GPisMap-xy.dat", df.positions.data(), (std::streamsize) df.positions.size());
    // erl::common::SaveBinaryFile<double>("GPisMap-distances.dat", distance_gt.data(), distance_gt.size());
    // erl::common::SaveBinaryFile<double>("GPisMap-gradients.dat", gradient_gt.data(), gradient_gt.size());
    // erl::common::SaveBinaryFile<double>("GPisMap-distance_variances.dat", distance_variance_gt.data(), distance_variance_gt.size());
    // erl::common::SaveBinaryFile<double>("GPisMap-gradientVariances.dat", gradient_variance_gt.data(), gradient_variance_gt.size());
}

/**
 * On AMD Ryzen 9 5950X 16-Core 32-Thread CPU @ 3.40GHz (Without LAPACK nor Intel MKL, Compiled with GNU 12)
 * Average training time:
 * ERL-GpisMap2D: 3738 us
 * GPisMap: 3508 us
 * Average testing time:
 * ERL-GpisMap2D: 132 ms
 * GPisMap: 148 ms
 *
 * On AMD Ryzen 9 5950X 16-Core 32-Thread CPU @ 3.40GHz (With LAPACK, Intel MKL 2023, Compiled with GNU 12)
 * Average training time:
 * ERL-GpisMap2D: 2974 us (0.80x slower) <--- This is the best performance
 * GPisMap: 3273 us (0.93x slower) <--- This is the best performance
 * Average testing time:
 * ERL-GpisMap2D: 101 ms (0.77x slower)
 * GPisMap: 96 ms (0.65x slower)
 *
 * On AMD Ryzen 9 5950X 16-Core 32-Thread CPU @ 3.40GHz (With LAPACK, Intel MKL 2023, Compiled with Intel C++ Compiler 2023)
 * Average training time:
 * ERL-GpisMap2D: 3898 us (1.04x slower)
 * GPisMap: 4671 us (1.33x slower)
 * Average testing time:
 * ERL-GpisMap2D: 92 ms (0.70x slower) <--- This is the best performance
 * GPisMap: 97 ms (0.66x slower)
 *
 * On Intel(R) Core(TM) i7-7820HQ 4-Core 8-Thread CPU @ 2.90GHz
 * Average training time:
 * ERL-GpisMap2D: 22268 us (5.96x slower)
 * GPisMap: 18931 us (5.40x slower)
 * Average testing time:
 * ERL-GpisMap2D: 1285 ms (9.73x slower)
 * GPisMap: 2323 ms (15.70x slower)
 *
 * On Intel(R) Core(TM) i7-7820HQ 4-Core 8-Thread CPU @ 2.90GHz (With LAPACK and Intel oneMKL 2023)
 * Average training time:
 * ERL-GpisMap2D: 10623 us (2.79x slower)
 * GPisMap: 21682 us (5.73x slower)
 * Average testing time:
 * ERL-GpisMap2D: 510 ms (3.85x slower)
 * GPisMap: 398 ms (2.76x slower)
 *
 * On Intel(R) Core(TM) i7-12700K 8P+4E 20-Thread CPU @ 4.90GHz (Without LAPACK nor Intel MKL)
 * Average training time:
 * ERL-GpisMap2D: 4973 us (1.33x slower)
 * GPisMap: 4318 us (1.23x slower)
 * Average testing time:
 * ERL-GpisMap2D: 205 ms (1.55x slower)
 * GPisMap: 166 ms (1.12x slower)
 *
 * On Intel(R) Core(TM) i7-12700K 8P+4E 20-Thread CPU @ 4.90GHz (With LAPACK and Intel oneMKL 2023)
 * Average training time:
 * ERL-GpisMap2D: 3362 us (0.90x slower)
 * GPisMap: 3356 us (0.96x slower)
 * Average testing time:
 * ERL-GpisMap2D: 110 ms (0.83x slower)
 * GPisMap: 111 ms (0.75x slower)
 *
 * On Intel(R) Core(TM) i7-12700K 8P+4E 20-Thread CPU @ 4.90GHz (With LAPACK and Intel oneMKL 2023, Compiled with Intel C++ Compiler 2023)
 * Average training time:
 * ERL-GpisMap2D: 3348 us (0.90x slower)
 * GPisMap: 3436 us (0.98x slower)
 * Average testing time:
 * ERL-GpisMap2D: 98 ms (0.74x slower)
 * GPisMap: 84 ms (0.62x slower) <--- This is the best performance
 */
