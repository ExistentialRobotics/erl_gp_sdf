#pragma once

#include "sdf_gp.hpp"

#include "erl_common/yaml.hpp"

namespace erl::gp_sdf {

    template<typename Dtype, int Dim>
    struct GpSdfMappingSetting : common::Yamlable<GpSdfMappingSetting<Dtype, Dim>> {

        using SdfGp = SdfGaussianProcess<Dtype, Dim>;
        using SdfGpSetting = typename SdfGp::Setting;

        struct TestQuery : common::Yamlable<TestQuery> {
            Dtype default_invalid_sdf = -0.001f;       // default sdf value for invalid prediction.
            Dtype max_test_valid_distance_var = 0.4f;  // maximum distance variance of prediction.
            Dtype search_area_half_size = 4.8f;        // the half-size of the search area.
            int num_neighbor_gps = 4;                  // Number of neighbors used for prediction.
            bool use_smallest = false;              // If true, use the smallest sdf for prediction.
            bool compute_gradient = true;           // If true, compute the sdf gradient.
            bool compute_gradient_variance = true;  // If true, compute the gradient variance.
            bool compute_covariance = true;       // If true, compute the covariance of prediction.
            bool use_gp_covariance = false;       // If true, compute variance with the GP.
            bool retrain_outdated = false;        // If true, retrain the trained GPs if outdated.
            std::size_t max_num_retrain_gps = 0;  // max number of GPs to retrain, -1 for unlimited.
            bool use_global_buffer = false;       // If true, use the global buffer.

            ERL_REFLECT_SCHEMA(
                TestQuery,
                ERL_REFLECT_MEMBER(TestQuery, default_invalid_sdf),
                ERL_REFLECT_MEMBER(TestQuery, max_test_valid_distance_var),
                ERL_REFLECT_MEMBER(TestQuery, search_area_half_size),
                ERL_REFLECT_MEMBER(TestQuery, num_neighbor_gps),
                ERL_REFLECT_MEMBER(TestQuery, use_smallest),
                ERL_REFLECT_MEMBER(TestQuery, compute_gradient),
                ERL_REFLECT_MEMBER(TestQuery, compute_gradient_variance),
                ERL_REFLECT_MEMBER(TestQuery, compute_covariance),
                ERL_REFLECT_MEMBER(TestQuery, use_gp_covariance),
                ERL_REFLECT_MEMBER(TestQuery, retrain_outdated),
                ERL_REFLECT_MEMBER(TestQuery, max_num_retrain_gps),
                ERL_REFLECT_MEMBER(TestQuery, use_global_buffer));
        };

        struct QueuePriority : common::Yamlable<QueuePriority> {
            long max_buf_outdated_count = 1e4;      // max buffer outdated count
            Dtype distance_weight = 0.1f;           // weight for distance in priority calculation
            Dtype query_weight_for_loading = 1.0f;  // weight of query count for loading priority
            Dtype query_weight_for_retrain = 1.0f;  // weight of query count for retrain priority

            ERL_REFLECT_SCHEMA(
                QueuePriority,
                ERL_REFLECT_MEMBER(QueuePriority, max_buf_outdated_count),
                ERL_REFLECT_MEMBER(QueuePriority, distance_weight),
                ERL_REFLECT_MEMBER(QueuePriority, query_weight_for_loading),
                ERL_REFLECT_MEMBER(QueuePriority, query_weight_for_retrain));
        };

        TestQuery test_query;                 // parameters used by Test.
        QueuePriority queue_priority;         // parameters for GP loading and retrain priority.
        uint32_t num_threads = 64;            // number of threads for testing.
        long min_num_gps_to_update = 256;     // min number of GPs to trigger an update.
        Dtype update_hz = 20.0f;              // update frequency in Hz.
        Dtype sensor_noise = 0.01f;           // sensor noise for surface data.
        Dtype gp_sdf_area_scale = 4.0f;       // ratio between GP area and cluster area
        Dtype max_valid_gradient_var = 0.1f;  // max gradient variance valid for training.
        Dtype invalid_position_var = 2.0f;    // position var when > max_valid_gradient_var.
        std::shared_ptr<SdfGpSetting> sdf_gp = std::make_shared<SdfGpSetting>();

        ERL_REFLECT_SCHEMA(
            GpSdfMappingSetting,
            ERL_REFLECT_MEMBER(GpSdfMappingSetting, test_query),
            ERL_REFLECT_MEMBER(GpSdfMappingSetting, queue_priority),
            ERL_REFLECT_MEMBER(GpSdfMappingSetting, num_threads),
            ERL_REFLECT_MEMBER(GpSdfMappingSetting, min_num_gps_to_update),
            ERL_REFLECT_MEMBER(GpSdfMappingSetting, update_hz),
            ERL_REFLECT_MEMBER(GpSdfMappingSetting, sensor_noise),
            ERL_REFLECT_MEMBER(GpSdfMappingSetting, gp_sdf_area_scale),
            ERL_REFLECT_MEMBER(GpSdfMappingSetting, max_valid_gradient_var),
            ERL_REFLECT_MEMBER(GpSdfMappingSetting, invalid_position_var),
            ERL_REFLECT_MEMBER(GpSdfMappingSetting, sdf_gp));

        bool
        PostDeserialization() override;
    };

    using GpSdfMappingSetting2Df = GpSdfMappingSetting<float, 2>;
    using GpSdfMappingSetting2Dd = GpSdfMappingSetting<double, 2>;
    using GpSdfMappingSetting3Df = GpSdfMappingSetting<float, 3>;
    using GpSdfMappingSetting3Dd = GpSdfMappingSetting<double, 3>;

    extern template struct GpSdfMappingSetting<float, 2>;
    extern template struct GpSdfMappingSetting<double, 2>;
    extern template struct GpSdfMappingSetting<float, 3>;
    extern template struct GpSdfMappingSetting<double, 3>;
}  // namespace erl::gp_sdf
