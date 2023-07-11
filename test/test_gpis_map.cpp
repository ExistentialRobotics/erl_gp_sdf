#include "GPisMap.h"
#include "erl_common/test_helper.hpp"
#include "erl_gaussian_process/noisy_input_gp.hpp"
#include "erl_gp_sdf/gpis_map_2d.hpp"
#include "erl_env/gazebo_room.hpp"
#include "strct.h"

using namespace erl::common;

void
PrintStatusOfErlGpisMap(const erl::gp_sdf::GpisMap2D &gpis_map, std::ostream &os = std::cout) {
    if (&os == &std::cout) {
        std::cout << PrintInfo("Tree Structure of ERL-GpisMap2D:") << std::endl;
    } else {
        std::cout << PrintInfo("Print tree structure of ERL-GpisMap2D to ostream ...") << std::endl;
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
        std::cout << PrintInfo("Tree Structure of GPisMap:") << std::endl;
    } else {
        std::cout << PrintInfo("Print tree structure of GPisMap to ostream ...") << std::endl;
    }
    os << gpis_map.m_tree_;
}

int
main() {
    const char *train_dat_file = "double/train.dat";
    auto train_data_loader = erl::env::GazeboRoom::TrainDataLoader(train_dat_file);

    GPisMap gpm;
    erl::gp_sdf::GpisMap2D gpis_map;
    gpis_map.m_setting_->update_gp_sdf->offset_distance = GPISMAP_FBIAS;
    std::cout << gpis_map.GetSetting()->AsYamlString() << std::endl;

    std::cout.precision(5);
    std::cout << std::scientific;

    std::size_t t_train_gt = 0;
    std::size_t t_train_ans = 0;
    int cnt = 0;
    std::stringstream ss;
    for (auto df: train_data_loader) {
        ss.str(std::string());
        ss << "GPisMap-Train[" << cnt << ']';
        t_train_gt += ReportTime<std::chrono::microseconds>(ss.str().c_str(), 0, true, [&]() {
            gpm.Update(df.angles.data(), df.distances.data(), int(df.angles.size()), df.pose_matlab);
        });
        ss.str(std::string());
        ss << "GpisMap2D-Train[" << cnt << ']';
        t_train_ans += ReportTime<std::chrono::microseconds>(ss.str().c_str(), 0, true, [&]() { gpis_map.Update(df.angles, df.distances, df.pose_numpy); });
        cnt++;
        std::cout << "===============================================" << std::endl;
    }

    t_train_gt /= train_data_loader.size();
    t_train_ans /= train_data_loader.size();

    std::cout << PrintInfo("Average training time:") << std::endl;
    std::cout << "ERL-GpisMap2D: " << t_train_ans << " us" << std::endl << "GPisMap: " << t_train_gt << " us" << std::endl;

    std::ofstream fs;
    fs.open("GpisMap2D-QuadTree-gt.txt", std::ios::out);
    PrintStatusOfGpisMap(gpm, fs);
    fs.close();

    fs.open("GpisMap2D-QuadTree-ans.txt", std::ios::out);
    PrintStatusOfErlGpisMap(gpis_map, fs);
    fs.close();

    std::cout << PrintInfo("Result compare report:\n", "*********************\n", "Check m_train_buffer_\n", "*********************") << std::endl;
    CheckAnswers("m_gp_theta_->m_train_buffer_.vec_angles", gpis_map.m_gp_theta_->m_train_buffer_.vec_angles, gpm.m_obs_theta_);
    CheckAnswers("m_gp_theta_->m_train_buffer_.vec_ranges", gpis_map.m_gp_theta_->m_train_buffer_.vec_ranges, gpm.m_obs_range_);
    CheckAnswers("m_gp_theta_->m_train_buffer_.vec_mapped_distances", gpis_map.m_gp_theta_->m_train_buffer_.vec_mapped_distances, gpm.m_obs_f_);
    CheckAnswers("m_gp_theta_->m_train_buffer_.position", gpis_map.m_gp_theta_->m_train_buffer_.position, gpm.m_pose_tr_);
    CheckAnswers("m_gp_theta_->m_train_buffer_.rotation", gpis_map.m_gp_theta_->m_train_buffer_.rotation, gpm.m_pose_r_);
    CheckFloatingValue("m_gp_theta_->m_train_buffer_.max_distance", gpis_map.m_gp_theta_->m_train_buffer_.max_distance, gpm.m_range_obs_max_);
    CheckAnswers(
        "m_gp_theta_->m_train_buffer_.mat_xy_local",
        gpis_map.m_gp_theta_->m_train_buffer_.mat_xy_local.reshaped(gpis_map.m_gp_theta_->m_train_buffer_.mat_xy_local.size(), 1),
        gpm.m_obs_xy_local_);
    CheckAnswers(
        "m_gp_theta_->m_train_buffer_.mat_xy_global",
        gpis_map.m_gp_theta_->m_train_buffer_.mat_xy_global.reshaped(gpis_map.m_gp_theta_->m_train_buffer_.mat_xy_global.size(), 1),
        gpm.m_obs_xy_global_);

    std::cout << PrintInfo("*************************\n", "Check regress observation\n", "*************************") << std::endl;
    CheckIntegralValue("m_gp_theta_->m_setting_->overlap_size", gpis_map.m_gp_theta_->m_setting_->overlap_size, gpm.m_gpo_->param.overlap);
    CheckIntegralValue(
        "m_gp_theta_->m_setting_->group_size",
        gpis_map.m_gp_theta_->m_setting_->group_size - gpis_map.m_gp_theta_->m_setting_->overlap_size,
        gpm.m_gpo_->param.group_size);
    CheckFloatingValue("m_gp_theta_->m_setting_->boundary_margin", gpis_map.m_gp_theta_->m_setting_->boundary_margin, gpm.m_gpo_->param.margin);
    CheckFloatingValue("m_gp_theta_->m_setting_->gp->kernel->scale", gpis_map.m_gp_theta_->m_setting_->gp->kernel->scale, gpm.m_gpo_->param.scale);
    CheckFloatingValue("m_setting_->gp_theta->sensor_range_var", gpis_map.m_setting_->gp_theta->sensor_range_var, gpm.m_gpo_->param.noise);

    CheckAnswers("m_gp_theta_.m_partitions_", gpis_map.m_gp_theta_->m_partitions_, gpm.m_gpo_->range);

    std::cout << "m_gps_[i]->m_vec_alpha_:" << std::endl;
    for (size_t i = 0; i < gpis_map.m_gp_theta_->m_gps_.size(); ++i) {
        ss.str(std::string());
        ss << '\t' << std::setw(2) << i;
        CheckAnswers(ss.str().c_str(), gpis_map.m_gp_theta_->m_gps_[i]->m_vec_alpha_, gpm.m_gpo_->gps[i]->alpha);
    }

    std::cout << "m_gps_[i]->m_mat_l_:" << std::endl;
    for (size_t i = 0; i < gpis_map.m_gp_theta_->m_gps_.size(); ++i) {
        ss.str(std::string());
        ss << '\t' << std::setw(2) << i;
        CheckAnswers(ss.str().c_str(), gpis_map.m_gp_theta_->m_gps_[i]->m_mat_l_, gpm.m_gpo_->gps[i]->L);
    }

    std::cout << "m_gps_[i]->m_mat_x_train_:" << std::endl;
    for (size_t i = 0; i < gpis_map.m_gp_theta_->m_gps_.size(); ++i) {
        ss.str(std::string());
        ss << '\t' << std::setw(2) << i;
        CheckAnswers(ss.str().c_str(), gpis_map.m_gp_theta_->m_gps_[i]->m_mat_x_train_, gpm.m_gpo_->gps[i]->x);
    }

    std::cout
        << PrintInfo("*****************************************\n", "Check UpdateSurfacePoints & AddNewSurfaceSamples\n", "*****************************************")
        << std::endl;
    std::vector<std::shared_ptr<erl::geometry::Node>> nodes_ans;
    gpis_map.m_quadtree_->CollectNodesOfTypeInArea(0, gpis_map.m_quadtree_->GetArea(), nodes_ans);  // collect all m_nodes_
    std::vector<std::shared_ptr<Node>> nodes_gt;
    gpm.m_tree_->QueryRange(gpm.m_tree_->m_boundary_, nodes_gt);
    if (CheckIntegralValue("Number of Nodes stored in the QuadTree", nodes_ans.size(), nodes_gt.size())) {
        std::cout << PrintInfo("Start to compare these m_nodes_:") << std::endl;
        for (size_t j = 0; j < nodes_ans.size(); ++j) {
            ss.str(std::string());
            auto &node_ans = nodes_ans[j];
            auto &node_gt = nodes_gt[j];
            auto node_data_ans = node_ans->GetData<erl::gp_sdf::GpisData2D>();
            auto position_ans = node_ans->position;
            auto position_gt = node_gt->GetPos();
            auto &gradient_ans = node_data_ans->gradient;
            auto &gradient_gt = node_gt->m_grad_;

            ss << "\t[" << j << "]position.x";
            CheckFloatingValue(ss.str().c_str(), position_ans.x(), position_gt.x);

            ss.str(std::string());
            ss << "\t[" << j << "]position.y";
            CheckFloatingValue(ss.str().c_str(), position_ans.y(), position_gt.y);

            ss.str(std::string());
            ss << "\t[" << j << "]distance";
            CheckFloatingValue(ss.str().c_str(), node_data_ans->distance, node_gt->m_val_);

            ss.str(std::string());
            ss << "\t[" << j << "]gradient.x";
            CheckFloatingValue(ss.str().c_str(), gradient_ans.x(), gradient_gt.x);

            ss.str(std::string());
            ss << "\t[" << j << "]gradient.y";
            CheckFloatingValue(ss.str().c_str(), gradient_ans.y(), gradient_gt.y);

            ss.str(std::string());
            ss << "\t[" << j << "]var_position";
            CheckFloatingValue(ss.str().c_str(), node_data_ans->var_position, node_gt->GetPosNoise());

            ss.str(std::string());
            ss << "\t[" << j << "]var_gradient";
            CheckFloatingValue(ss.str().c_str(), node_data_ans->var_gradient, node_gt->GetGradNoise());

            std::cout << "\t---------------------------------" << std::endl;
        }
    }

    std::cout << PrintInfo("***************\n", "Check UpdateGpSdf\n", "***************") << std::endl;
    std::vector<std::shared_ptr<const erl::geometry::IncrementalQuadtree>> clusters_ans;
    std::vector<QuadTree *> clusters_gt;
    gpis_map.m_quadtree_->CollectTrees(
        [](const std::shared_ptr<const erl::geometry::IncrementalQuadtree> &tree) -> bool { return tree->GetData<void>() != nullptr; },
        clusters_ans);
    gpm.m_tree_->CollectTrees([](const QuadTree *tree) -> bool { return tree->m_gp_ != nullptr; }, clusters_gt);

    if (CheckIntegralValue("Number of QuadTrees having GP", clusters_ans.size(), clusters_gt.size())) {
        std::cout << "Start to compare these GPs:" << std::endl;
        int i = 0;
        for (auto &cluster: clusters_ans) {
            auto itr = std::find_if(clusters_gt.begin(), clusters_gt.end(), [&](QuadTree *tree) -> bool {
                auto c = tree->GetCenter();
                return (cluster->GetArea().center - Eigen::Vector2d{c.x, c.y}).squaredNorm() < double(1.e-6);
            });
            if (itr == clusters_gt.end()) {
                std::cout << PrintError("[Error]") << ": failed to find the corresponding GP for the cluster at " << cluster->GetArea().center << std::endl;
            } else {
                auto gp_ans = cluster->GetData<erl::gaussian_process::NoisyInputGaussianProcess>();
                auto gp_gt = (*itr)->m_gp_;

                ss.str(std::string());
                ss << "\t[" << i << "]->m_trained_";
                CheckIntegralValue(ss.str().c_str(), gp_ans->m_trained_, gp_gt->m_trained_);

                ss.str(std::string());
                ss << "\t[" << i << "]->m_mat_x_train_";
                CheckAnswers(ss.str().c_str(), gp_ans->m_mat_x_train_, gp_gt->m_x_);

                ss.str(std::string());
                ss << "\t[" << i << "]->m_vec_sigma_x_";
                CheckAnswers(ss.str().c_str(), gp_ans->m_vec_sigma_x_, gp_gt->m_sigx_);

                ss.str(std::string());
                ss << "\t[" << i << "]->m_vec_sigma_grad_";
                CheckAnswers(ss.str().c_str(), gp_ans->m_vec_sigma_grad_, gp_gt->m_siggrad_);

                ss.str(std::string());
                ss << "\t[" << i << "]->m_vec_grad_flag_";
                std::vector<int> grad_flag_gt;
                std::for_each(gp_gt->m_gradflag_.begin(), gp_gt->m_gradflag_.end(), [&](double v) { grad_flag_gt.push_back(v > double(0.5)); });
                if (CheckAnswers(ss.str().c_str(), gp_ans->m_vec_grad_flag_, grad_flag_gt)) {
                    ss.str(std::string());
                    ss << "\t[" << i << "]->m_vec_y_";
                    CheckAnswers(ss.str().c_str(), gp_ans->m_vec_y_, gp_gt->m_y_);

                    ss.str(std::string());
                    ss << "\t[" << i << "]->m_mat_k_train_";
                    CheckAnswers(ss.str().c_str(), gp_ans->m_mat_k_train_, gp_gt->m_k_);

                    ss.str(std::string());
                    ss << "\t[" << i << "]->m_mat_l_";
                    CheckAnswers(ss.str().c_str(), gp_ans->m_mat_l_, gp_gt->m_l_);

                    ss.str(std::string());
                    ss << "\t[" << i << "]->m_vec_alpha_";
                    CheckAnswers(ss.str().c_str(), gp_ans->m_vec_alpha_, gp_gt->m_alpha_);
                }
            }
            std::cout << "\t--------------------------------" << std::endl;
            i++;
        }
    }

    Eigen::VectorXd distance_ans, distance_variance_ans;
    Eigen::Matrix2Xd gradient_ans, gradient_variance_ans;
    const char *test_dat_file;
#if defined(USE_FLOAT32)
    test_dat_file = "float/test.dat";
#else
    test_dat_file = "double/test.dat";
#endif
    auto df = erl::env::GazeboRoom::TestDataFrame(test_dat_file);
    long t_test_ans, t_test_gt;
    t_test_gt = ReportTime<std::chrono::milliseconds>("GPisMap-test", 5, false, [&]() { gpm.Test(df.positions.data(), df.dim, df.num_queries, df.out_buf.data()); });
    std::cout << "\t---------------------------------" << std::endl;
    t_test_ans = ReportTime<std::chrono::milliseconds>("ERL-GpisMap2D-test", 5, false, [&]() {
        gpis_map.Test(df.positions, distance_ans, gradient_ans, distance_variance_ans, gradient_variance_ans);
    });
    std::cout << "\t---------------------------------" << std::endl;

    Eigen::VectorXd distance_gt, distance_variance_gt;
    Eigen::Matrix2Xd gradient_gt, gradient_variance_gt;
    df.Extract(distance_gt, gradient_gt, distance_variance_gt, gradient_variance_gt);
    CheckAnswers("distance", distance_ans, distance_gt);
    CheckAnswers("gradient", gradient_ans, decltype(gradient_ans)(gradient_gt.colwise().normalized()));
    CheckAnswers("distanceVariance", distance_variance_ans, distance_variance_gt);
    CheckAnswers("gradientVariance", gradient_variance_ans, gradient_variance_gt);

    std::cout << PrintInfo("Average training time:") << std::endl;
    std::cout << "ERL-GpisMap2D: " << t_train_ans << " us" << std::endl << "GPisMap: " << t_train_gt << " us" << std::endl;
    std::cout << PrintInfo("Average testing time:") << std::endl;
    std::cout << "ERL-GpisMap2D: " << t_test_ans << " ms" << std::endl << "GPisMap: " << t_test_gt << " ms" << std::endl;

    SaveBinaryFile<double>("GPisMap-xy.dat", df.positions.data(), (std::streamsize) df.positions.size());
    SaveBinaryFile<double>("GPisMap-distances.dat", distance_gt.data(), distance_gt.size());
    SaveBinaryFile<double>("GPisMap-gradients.dat", gradient_gt.data(), gradient_gt.size());
    SaveBinaryFile<double>("GPisMap-distance_variances.dat", distance_variance_gt.data(), distance_variance_gt.size());
    SaveBinaryFile<double>("GPisMap-gradientVariances.dat", gradient_variance_gt.data(), gradient_variance_gt.size());

    return 0;
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
