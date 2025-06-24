#pragma once

#include "abstract_surface_mapping.hpp"
#include "surface_data_manager.hpp"
#include "surface_mapping_quadtree.hpp"

#include "erl_common/exception.hpp"

namespace erl::gp_sdf {

    template<typename Dtype>
    class AbstractSurfaceMapping2D : public AbstractSurfaceMapping {

    public:
        [[nodiscard]] virtual Dtype
        GetSensorNoise() const = 0;

        [[nodiscard]] virtual unsigned int
        GetClusterLevel() const = 0;

        virtual geometry::QuadtreeKeySet
        GetChangedClusters() = 0;

        virtual std::shared_ptr<SurfaceMappingQuadtree<Dtype>>
        GetQuadtree() = 0;

        [[nodiscard]] virtual const SurfaceDataManager<Dtype, 2> &
        GetSurfaceDataManager() const = 0;

        virtual bool
        Update(
            const Eigen::Ref<const Eigen::Matrix2<Dtype>> &rotation,
            const Eigen::Ref<const Eigen::Vector2<Dtype>> &translation,
            const Eigen::Ref<const Eigen::MatrixX<Dtype>> &ranges) = 0;

        [[nodiscard]] virtual bool
        operator==(const AbstractSurfaceMapping2D & /*other*/) const {
            throw NotImplemented(__PRETTY_FUNCTION__);
        }

        [[nodiscard]] virtual bool
        operator!=(const AbstractSurfaceMapping2D &other) const {
            return !(*this == other);
        }
    };
}  // namespace erl::gp_sdf
