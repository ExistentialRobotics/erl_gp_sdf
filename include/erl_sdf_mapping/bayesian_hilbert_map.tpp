#pragma once

#include "erl_geometry/intersection.hpp"

namespace erl::sdf_mapping {

    template<typename Dtype, int Dim>
    BayesianHilbertMap<Dtype, Dim>::BayesianHilbertMap(
        std::shared_ptr<BayesianHilbertMapSetting> setting,
        std::shared_ptr<Covariance> kernel,
        MatrixDX hinged_points,
        Aabb map_boundary,
        const uint64_t seed)
        : m_setting_(std::move(setting)),
          m_kernel_(std::move(kernel)),
          m_hinged_points_(std::move(hinged_points)),
          m_map_boundary_(std::move(map_boundary)),
          m_generator_(seed) {
        // TODO: initialize m_sigma_inv_ and m_sigma_ here

        const long m = m_hinged_points_.cols();
        const Dtype sigma = m_setting_->init_sigma;
        const Dtype sigma_inv = 1.0 / sigma;
        if (m_setting_->diagonal_sigma) {
            m_sigma_ = VectorX::Zero(m);
            m_sigma_inv_ = VectorX::Zero(m);
            for (long i = 0; i < m; ++i) {
                m_sigma_(i, 0) = sigma;          // initialize the diagonal of the covariance matrix
                m_sigma_inv_(i, 0) = sigma_inv;  // initialize the inverse covariance matrix
            }
        } else {
            m_sigma_ = MatrixX::Zero(m, m);
            m_sigma_inv_ = MatrixX::Zero(m, m);
            for (long i = 0; i < m; ++i) {
                m_sigma_(i, i) = sigma;          // initialize the diagonal of the covariance matrix
                m_sigma_inv_(i, i) = sigma_inv;  // initialize the inverse covariance matrix
            }
        }
        m_mu_ = VectorX::Zero(m);
    }

    template<typename Dtype, int Dim>
    std::pair<typename BayesianHilbertMap<Dtype, Dim>::MatrixDX, typename BayesianHilbertMap<Dtype, Dim>::VectorX>
    BayesianHilbertMap<Dtype, Dim>::GenerateDataset(const Eigen::Ref<const VectorD> &sensor_position, const Eigen::Ref<const MatrixDX> &points) {

        // 1. check if the ray intersects with the map boundary.
        // 2. compute the range to sample free points and the number of points to sample.
        // 3. sample free points uniformly within the range.
        // 4. return the result.

        const Dtype max_distance = m_setting_->max_distance;
        const Dtype free_points_per_meter = m_setting_->free_points_per_meter;
        const Dtype free_sampling_margin = m_setting_->free_sampling_margin;

        std::vector<std::tuple<long, bool, long, Dtype, Dtype>> infos;  // tuple of (point_index, hit_flag, num_free_points, d1, d2)
        infos.reserve(points.cols());
        long num_total_free_points = 0;
        long num_total_hit_points = 0;

        for (long i = 0; i < points.cols(); ++i) {
            VectorD point = points.col(i);
            VectorD v = sensor_position - point;
            Dtype d1 = 0;
            Dtype d2 = 0;
            bool hit_flag = false;
            bool intersected = false;
            bool is_inside = false;
            // compute intersection between the ray (point -> sensor_position) and the map boundary
            geometry::ComputeIntersectionBetweenRayAndAabb<Dtype, Dim>(
                point,
                v.cwiseInverse(),
                m_map_boundary_.min(),
                m_map_boundary_.max(),
                d1,
                d2,
                intersected,
                is_inside);

            if (!intersected) { continue; }
            if (d1 < 0 && d2 < 0) { continue; }  // the ray hits a point in front of the map

            Dtype v_norm = v.norm();
            if (is_inside) {                                     // the ray hits a point inside the map, d2 < 0 is useless
                if (v_norm < max_distance) { hit_flag = true; }  // a hit
                d1 = v_norm - d1;                                // distance from sensor to the map boundary
                if (d1 < free_sampling_margin) { d1 = free_sampling_margin; }
                // case 1: sensor is outside the map, d1 > 0, we need to make sure d1 > free_sampling_margin
                // case 2: sensor is inside the map, d1 < 0, we need to make sure d1 is at least free_sampling_margin
                d2 = v_norm - free_sampling_margin;  // adjust the distance to avoid sampling too close to the surface
            } else {
                Dtype tmp = d1;
                d1 = v_norm - d2;   // distance from the sensor to the map boundary (first intersection point)
                d2 = v_norm - tmp;  // distance from the sensor to the map boundary (second intersection point)
                if (d1 < free_sampling_margin) { d1 = free_sampling_margin; }
                d2 = std::min(d2, v_norm - free_sampling_margin);  // adjust the distance to avoid sampling too close to the surface
            }
            auto n = static_cast<long>(std::ceil((d2 - d1) * free_points_per_meter));  // number of free points to sample
            if (n < 0) { n = 0; }                                                      // avoid negative number of points
            num_total_free_points += n;                                                // count the number of free points to sample
            num_total_hit_points += static_cast<long>(hit_flag);                       // count the number of hit points
            d1 /= v_norm;
            d2 /= v_norm;
            infos.emplace_back(i, hit_flag, n, d1, d2);
        }

        MatrixDX result_points(Dim, num_total_hit_points + num_total_free_points);
        VectorX result_labels(result_points.cols());
        Dtype *points_ptr = result_points.data();
        Dtype *labels_ptr = result_labels.data();
        for (const auto &[point_index, hit_flag, num_free_points, d1, d2]: infos) {
            const Dtype *point_ptr = points.col(point_index).data();
            if (hit_flag) {
                std::memcpy(points_ptr, point_ptr, sizeof(Dtype) * Dim);  // copy the hit point to the result
                *(labels_ptr++) = 1.0;                                    // label as occupied and move the pointer to the next position
                points_ptr += Dim;                                        // move the pointer to the next position
            }
            // sample free points uniformly within the range [d1, d2]
            std::uniform_real_distribution<Dtype> distribution(d1, d2);
            for (long j = 0; j < num_free_points; ++j) {
                Dtype r = distribution(m_generator_);  // sample a random distance within the range [d1, d2]
                Dtype s = 1 - r;
                for (long k = 0; k < Dim; ++k, ++points_ptr) { *points_ptr = sensor_position[k] * s + point_ptr[k] * r; }  // compute the free point position
                *(labels_ptr++) = 0.0;                                                                                     // label as free
            }
        }

        return {result_points, result_labels};
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertMap<Dtype, Dim>::RunExpectationMaximization(const MatrixDX &points, const VectorX &labels) {
        ERL_DEBUG_ASSERT(points.cols() == labels.size(), "points.cols() = {}, labels.size() = {}.", points.cols(), labels.size());

        if (m_phi_.rows() < points.cols()) { m_phi_.resize(points.cols(), m_hinged_points_.cols()); }
        const auto &[n, m] = m_kernel_->ComputeKtest(points, points.cols(), m_hinged_points_, m_hinged_points_.cols(), m_phi_);
        if (m_xi_.size() < n) {
            m_xi_.resize(n);
            m_lambda_.resize(n);
        }
        m_xi_.setOnes();

        for (int itr = 0; itr < m_setting_->num_em_iterations; ++itr) { RunExpectationMaximizationIteration(labels); }
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertMap<Dtype, Dim>::RunExpectationMaximizationIteration(const VectorX &labels) {
        const long n = labels.size();
        const long m = m_hinged_points_.cols();

        Dtype *xi_ptr = m_xi_.data();
        Dtype *lam_ptr = m_lambda_.data();
        Dtype *alpha_ptr = m_alpha_.data();
        const Dtype *labels_ptr = labels.data();

        // calculate lambda(xi) = (sigmoid(xi) - 0.5) / (2 * xi)
        for (long j = 0; j < n; ++j) { lam_ptr[j] = (1.0 / (1.0 + std::exp(-xi_ptr[j])) - 0.5) / (2 * xi_ptr[j]); }

        if (m_setting_->diagonal_sigma) {  // diagonal sigma
            // E-Step: calculate the posterior
            Dtype *mu_ptr = m_mu_.data();
            Dtype *sigma_inv_ptr = m_sigma_inv_.data();
            Dtype *sigma_ptr = m_sigma_.data();
            for (long i = 0; i < m; ++i) {  // loop over the features
                const Dtype *phi_ptr = m_phi_.col(i).data();
                for (long j = 0; j < n; ++j) {  // loop over the points
                    sigma_inv_ptr[i] += 2 * lam_ptr[j] * phi_ptr[j] * phi_ptr[j];
                    alpha_ptr[i] += (labels_ptr[j] - 0.5) * phi_ptr[j];
                }
                sigma_ptr[i] = 1.0 / sigma_inv_ptr[i];    // sigma_inv' = sigma_inv + 2 * ((phi ** 2).T * lams).sum(dim=1)
                mu_ptr[i] = sigma_ptr[i] * alpha_ptr[i];  // mu' = sigma' * (mu / sigma + phi.T @ (labels - 0.5))
            }
            // M-Step: update xi
            VectorX phi(m);
            for (long j = 0; j < n; ++j) {
                phi = m_phi_.row(j).head(m).transpose();
                const Dtype *phi_ptr = phi.data();
                Dtype a = 0;
                xi_ptr[j] = 0;
                for (long i = 0; i < m; ++i) {  // loop over the features
                    a += phi_ptr[i] * mu_ptr[i];
                    xi_ptr[j] += sigma_ptr[i] * (phi_ptr[i] * phi_ptr[i]);
                }
                xi_ptr[j] += a * a;
                xi_ptr[j] = std::sqrt(xi_ptr[j]);  // xi = sqrt(phi.T @ sigma @ phi + (phi.T @ mu) ** 2)
            }
        } else {  // non-diagonal sigma
            // E-Step: calculate the posterior
            // sigma_inv' = sigma_inv + 2 * (phi.T * lams) @ phi
            // mu' = sigma' @ (sigma_inv @ mu + phi.T @ (labels - 0.5))
            // alpha = sigma_inv @ mu
            VectorX lam_phi(n);
            Dtype *lam_phi_ptr = lam_phi.data();
            for (long c = 0; c < m; ++c) {  // loop over cols
                const Dtype *phi_c_ptr = m_phi_.col(c).data();
                for (long j = 0; j < n; ++j) {
                    lam_phi_ptr[j] = lam_ptr[j] * phi_c_ptr[j];
                    alpha_ptr[c] += (labels_ptr[j] - 0.5) * phi_c_ptr[j];
                }
                for (long r = 0; r < m; ++r) {  // loop over rows
                    const Dtype *phi_r_ptr = m_phi_.col(r).data();
                    Dtype &sigma_inv = m_sigma_inv_(r, c);
                    for (long j = 0; j < n; ++j) { sigma_inv += 2 * lam_phi_ptr[j] * phi_r_ptr[j]; }
                }
            }
            m_sigma_inv_mat_l_ = m_sigma_inv_.llt().matrixL();  // Cholesky decomposition
            m_mu_ = m_sigma_inv_mat_l_.template triangularView<Eigen::Lower>().solve(m_alpha_);
            m_sigma_inv_mat_l_.transpose().template triangularView<Eigen::Upper>().solveInPlace(m_mu_);
            // M-Step: update xi
            VectorX phi(m);
            for (long j = 0; j < n; ++j) {
                phi = m_phi_.row(j).head(m).transpose();
                const Dtype a = phi.dot(m_mu_);
                m_sigma_inv_mat_l_.template triangularView<Eigen::Lower>().solveInPlace(phi);
                xi_ptr[j] = std::sqrt(phi.squaredNorm() + a * a);
            }
        }
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertMap<Dtype, Dim>::Update(const Eigen::Ref<const VectorD> &sensor_position, const Eigen::Ref<const MatrixDX> &points) {
        auto [dataset_points, dataset_labels] = GenerateDataset(sensor_position, points);
        if (dataset_points.cols() == 0) {
            ERL_WARN("No valid points generated for update. Skipping update.");
            return;
        }
        RunExpectationMaximization(dataset_points, dataset_labels);
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertMap<Dtype, Dim>::Predict(
        const Eigen::Ref<const MatrixDX> &points,
        const bool faster,
        const bool compute_gradient,
        VectorX &prob_occupied,
        MatrixDX &gradient) const {
        const long n = points.cols();
        const long m = compute_gradient ? n * (Dim + 1) : n;
        if (m_phi_.rows() < n || m_phi_.cols() < m) { m_phi_.resize(n, m); }
        m_kernel_->ComputeKtestWithGradient(
            m_hinged_points_,
            m_hinged_points_.cols(),
            Eigen::VectorXi::Constant(m_hinged_points_.cols(), -1),
            points,
            n,
            compute_gradient,
            m_phi_);

        prob_occupied.resize(n);
        Dtype *prob_occupied_ptr = prob_occupied.data();
        if (faster) {  // assume sigma is very small, we can use the mean directly
            for (long i = 0; i < n; ++i) {
                Dtype t1 = m_phi_.col(i).dot(m_mu_);
                prob_occupied_ptr[i] = 1.0 / (1.0 + std::exp(-t1));
            }
        } else {
            if (m_setting_->diagonal_sigma) {
                for (long i = 0; i < n; ++i) {
                    Dtype t1 = m_phi_.col(i).dot(m_mu_);
                    Dtype t2 = std::sqrt(1.0 + m_phi_.col(i).cwiseAbs2().dot(m_mu_) * (M_PI / 8.0));
                    prob_occupied_ptr[i] = 1.0 / (1.0 + std::exp(-t1 / t2));  // sigmoid function
                }
            } else {
                for (long i = 0; i < n; ++i) {
                    Dtype t1 = m_phi_.col(i).dot(m_mu_);
                    Dtype t2 = std::sqrt(1.0 + m_sigma_inv_mat_l_.template triangularView<Eigen::Lower>().solve(m_phi_.col(i)).squaredNorm() * (M_PI / 8.0));
                    prob_occupied_ptr[i] = 1.0 / (1.0 + std::exp(-t1 / t2));  // sigmoid function
                }
            }
        }

        if (compute_gradient) {
            gradient.resize(Dim, n);
            Dtype *gradient_ptr = gradient.data();
        }
    }

}  // namespace erl::sdf_mapping
