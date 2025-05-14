#pragma once

#include "erl_common/data_buffer_manager.hpp"
#include "erl_common/eigen.hpp"

namespace erl::sdf_mapping {

    template<typename Dtype, int Dim>
    struct SurfaceData {
        using VectorD = Eigen::Vector<Dtype, Dim>;

        VectorD position = VectorD::Zero();
        VectorD normal = VectorD::Zero();
        Dtype var_position = 0.0f;
        Dtype var_normal = 0.0f;

        SurfaceData() = default;

        SurfaceData(VectorD position, VectorD normal, const Dtype var_position, const Dtype var_normal)
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
        operator==(const SurfaceData &other) const;

        [[nodiscard]] bool
        operator!=(const SurfaceData &other) const;

        [[nodiscard]] bool
        Write(std::ostream &s) const;  // TODO: check implementation

        [[nodiscard]] bool
        Read(std::istream &s);
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

#include "surface_data_manager.tpp"
