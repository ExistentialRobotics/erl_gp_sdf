#include "erl_gp_sdf/abstract_surface_mapping.hpp"

namespace erl::gp_sdf {
    template<typename Dtype, int Dim>
    std::lock_guard<std::mutex>
    AbstractSurfaceMapping<Dtype, Dim>::GetLockGuard() {
        return std::lock_guard<std::mutex>(m_mutex_);
    }

    template<typename Dtype, int Dim>
    const typename AbstractSurfaceMapping<Dtype, Dim>::SurfDataManager &
    AbstractSurfaceMapping<Dtype, Dim>::GetSurfaceDataManager() const {
        return m_surf_data_manager_;
    }

    template class AbstractSurfaceMapping<float, 2>;
    template class AbstractSurfaceMapping<double, 2>;
    template class AbstractSurfaceMapping<float, 3>;
    template class AbstractSurfaceMapping<double, 3>;
}  // namespace erl::gp_sdf
