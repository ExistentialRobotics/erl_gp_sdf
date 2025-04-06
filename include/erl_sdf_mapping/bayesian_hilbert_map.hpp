#pragma once

#include "erl_common/random.hpp"
#include "erl_common/yaml.hpp"
#include "erl_covariance/covariance.hpp"
#include "erl_geometry/aabb.hpp"

namespace erl::sdf_mapping {

    struct BayesianHilbertMapSetting : common::Yamlable<BayesianHilbertMapSetting> {
        bool diagonal_sigma = false;         // if true, the covariance matrix will be assumed to be diagonal to speed up the computation.
        float max_distance = 30.0f;          // maximum distance from the sensor to consider a point as occupied.
        float free_points_per_meter = 2.0f;  // number of free points to sample per meter from the sensor.
        float free_sampling_margin = 0.1f;   // margin to use when sampling free points to avoid sampling too close to the surface or the sensor.
        float init_sigma = 1.0e6f;           // initial value for initializing the covariance matrix.
        int num_em_iterations = 3;           // number of iterations for the Expectation-Maximization (EM) algorithm to optimize the mean and covariance.
    };

    template<typename Dtype, int Dim>
    class BayesianHilbertMap {
    public:
        using Covariance = covariance::Covariance<Dtype>;
        using MatrixDX = Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorD = Eigen::Vector<Dtype, Dim>;
        using VectorX = Eigen::VectorX<Dtype>;
        using Aabb = geometry::Aabb<Dtype, Dim>;

    private:
        std::shared_ptr<BayesianHilbertMapSetting> m_setting_ = nullptr;  // settings for the Bayesian Hilbert Map
        std::shared_ptr<Covariance> m_kernel_ = nullptr;
        MatrixDX m_hinged_points_{};  // [Dim, N] hinged points for computing the hilbert space features
        Aabb m_map_boundary_{};
        std::mt19937_64 m_generator_;
        MatrixX m_sigma_inv_{};
        MatrixX m_sigma_{};            // posterior covariance of the weights
        MatrixX m_sigma_inv_mat_l_{};  // matrix L for Cholesky decomposition of m_sigma_inv_, used when diagonal_sigma is false
        VectorX m_alpha_{};            // alpha = sigma_inv * mu
        VectorX m_mu_{};               // posterior mean vector of weights
        MatrixX m_phi_{};              // [N, M] feature matrix
        VectorX m_xi_{};               // EM xi vector
        VectorX m_lambda_{};           // EM lambda vector

    public:
        /**
         * Constructor for BayesianHilbertMap.
         * @param setting settings for the Bayesian Hilbert Map
         * @param kernel kernel function for computing the features
         * @param hinged_points points in the world frame that will be used to compute the Hilbert space features.
         * @param map_boundary the boundary of the map in the world frame. This is used to generate the dataset and to check if a point is inside the map.
         * @param seed random seed for sampling free points. This is to ensure reproducibility of the dataset.
         */
        BayesianHilbertMap(
            std::shared_ptr<BayesianHilbertMapSetting> setting,
            std::shared_ptr<Covariance> kernel,
            MatrixDX hinged_points,
            Aabb map_boundary,
            uint64_t seed);

        /**
         * Generate a dataset of {x, y} where x is the position and y is the occupancy label (1 for occupied, 0 for free).
         * @param sensor_position the position of the sensor in the world frame.
         * @param points point cloud in the world frame of the sensor measurement.
         * @return
         */
        std::pair<MatrixDX, VectorX>
        GenerateDataset(const Eigen::Ref<const VectorD> &sensor_position, const Eigen::Ref<const MatrixDX> &points);

        void
        RunExpectationMaximization(const MatrixDX &points, const VectorX &labels);

        void
        RunExpectationMaximizationIteration(const VectorX &labels);

        void
        Update(const Eigen::Ref<const VectorD> &sensor_position, const Eigen::Ref<const MatrixDX> &points);

        void
        Predict(const Eigen::Ref<const MatrixDX> &points, bool faster, bool compute_gradient, VectorX &prob_occupied, MatrixDX &gradient) const;
    };

}  // namespace erl::sdf_mapping

#include "bayesian_hilbert_map.tpp"

template<>
struct YAML::convert<erl::sdf_mapping::BayesianHilbertMapSetting> {
    static Node
    encode(const erl::sdf_mapping::BayesianHilbertMapSetting &setting) {
        Node node;
        node["diagonal_sigma"] = setting.diagonal_sigma;
        node["max_distance"] = setting.max_distance;
        node["free_points_per_meter"] = setting.free_points_per_meter;
        node["free_sampling_margin"] = setting.free_sampling_margin;
        return node;
    }

    static bool
    decode(const Node &node, erl::sdf_mapping::BayesianHilbertMapSetting &setting) {
        if (!node.IsMap()) { return false; }
        setting.diagonal_sigma = node["diagonal_sigma"].as<bool>();
        setting.max_distance = node["max_distance"].as<float>();
        setting.free_points_per_meter = node["free_points_per_meter"].as<float>();
        setting.free_sampling_margin = node["free_sampling_margin"].as<float>();
        return true;
    }
};
