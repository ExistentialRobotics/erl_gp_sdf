#include "erl_gp_sdf/gp_sdf_mapping_setting.hpp"

namespace erl::gp_sdf {

    template struct GpSdfMappingSetting<float, 2>;
    template struct GpSdfMappingSetting<double, 2>;
    template struct GpSdfMappingSetting<float, 3>;
    template struct GpSdfMappingSetting<double, 3>;
}  // namespace erl::gp_sdf
