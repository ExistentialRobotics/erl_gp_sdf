#pragma once

#include "erl_common/data_buffer_manager.hpp"
#include "erl_common/eigen.hpp"

namespace erl::sdf_mapping {

    template<typename Dtype, int Dim>
    struct SurfaceData {
        using Vector = Eigen::Vector<Dtype, Dim>;

        Vector position = Vector::Zero();
        Vector normal = Vector::Zero();
        Dtype var_position = 0.0;
        Dtype var_normal = 0.0;

        SurfaceData() = default;

        SurfaceData(Vector position, Vector normal, const Dtype var_position, const Dtype var_normal)
            : position(std::move(position)),
              normal(std::move(normal)),
              var_position(var_position),
              var_normal(var_normal) {}

        SurfaceData(const SurfaceData &other) = default;
        SurfaceData &
        operator=(const SurfaceData &other) = default;
        SurfaceData(SurfaceData &&other) noexcept = default;
        SurfaceData &
        operator=(SurfaceData &&other) noexcept = default;

        [[nodiscard]] bool
        operator==(const SurfaceData &other) const {
            return position == other.position && normal == other.normal && var_position == other.var_position && var_normal == other.var_normal;
        }

        [[nodiscard]] bool
        operator!=(const SurfaceData &other) const {
            return !(*this == other);
        }
    };

    template<typename Dtype, int Dim>
    class SurfaceDataManager : public common::DataBufferManager<SurfaceData<Dtype, Dim>> {
    public:
        using Data = SurfaceData<Dtype, Dim>;

        SurfaceDataManager() = default;

        SurfaceDataManager(const SurfaceDataManager &) = default;
        SurfaceDataManager &
        operator=(const SurfaceDataManager &) = default;
        SurfaceDataManager(SurfaceDataManager &&) = default;
        SurfaceDataManager &
        operator=(SurfaceDataManager &&) = default;
    };

}  // namespace erl::sdf_mapping
