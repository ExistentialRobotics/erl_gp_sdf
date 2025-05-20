#include "erl_sdf_mapping/gp_sdf_mapping.hpp"

namespace erl::sdf_mapping {
    std::string
    AbstractGpSdfMapping::GetSdfMappingId(const std::string &surface_mapping_id) {
        const auto it = s_surface_mapping_to_sdf_mapping_.find(surface_mapping_id);
        if (it == s_surface_mapping_to_sdf_mapping_.end()) {
            ERL_WARN(
                "Failed to find sdf mapping for surface mapping: {}. Do you register it?",
                surface_mapping_id);
            return "";
        }
        return it->second;
    }

    std::shared_ptr<AbstractGpSdfMapping>
    AbstractGpSdfMapping::Create(
        const std::string &mapping_type,
        const std::shared_ptr<common::YamlableBase> &surface_mapping_setting,
        const std::shared_ptr<common::YamlableBase> &sdf_mapping_setting) {
        return Factory::GetInstance().Create(
            mapping_type,
            surface_mapping_setting,
            sdf_mapping_setting);
    }
}  // namespace erl::sdf_mapping
