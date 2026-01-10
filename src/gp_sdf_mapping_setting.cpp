#include "erl_gp_sdf/gp_sdf_mapping_setting.hpp"

#include <thread>

namespace erl::gp_sdf {

    template<typename Dtype, int Dim>
    bool
    GpSdfMappingSetting<Dtype, Dim>::PostDeserialization() {
        ERL_ASSERT_GT(queue_priority.max_buf_outdated_count, 0);
        ERL_ASSERT_GE(queue_priority.distance_weight, 0.0f);
        ERL_ASSERT_GE(queue_priority.query_weight_for_loading, 0.0f);
        ERL_ASSERT_GE(queue_priority.query_weight_for_retrain, 0.0f);

        // determine num_threads

#pragma omp parallel default(none)
#pragma omp critical
        {
            // do it on the main thread only
            if (omp_get_thread_num() == 0) {
                // get the number of available threads from OpenMP
                num_threads = std::min<uint32_t>(num_threads, omp_get_num_threads());
            }
        }

        // cap num_threads to hardware concurrency
        if (num_threads == 0 || num_threads > std::thread::hardware_concurrency()) {
            num_threads = std::thread::hardware_concurrency();
        }
        return true;
    }

    template struct GpSdfMappingSetting<float, 2>;
    template struct GpSdfMappingSetting<double, 2>;
    template struct GpSdfMappingSetting<float, 3>;
    template struct GpSdfMappingSetting<double, 3>;
}  // namespace erl::gp_sdf
