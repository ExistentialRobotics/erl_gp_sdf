#pragma once

#include "abstract_surface_mapping.hpp"
#include "surface_data_manager.hpp"
#include "surface_mapping_octree.hpp"

#include "erl_common/exception.hpp"

namespace erl::sdf_mapping {

    template<typename Dtype>
    class AbstractSurfaceMapping3D : public AbstractSurfaceMapping {

    public:
        [[nodiscard]] virtual Dtype
        GetSensorNoise() const = 0;

        [[nodiscard]] virtual unsigned int
        GetClusterLevel() const = 0;

        virtual geometry::OctreeKeySet
        GetChangedClusters() = 0;

        virtual std::shared_ptr<SurfaceMappingOctree<Dtype>>
        GetOctree() = 0;

        [[nodiscard]] virtual const SurfaceDataManager<Dtype, 3> &
        GetSurfaceDataManager() const = 0;

        virtual bool
        Update(
            const Eigen::Ref<const Eigen::Matrix3<Dtype>> &rotation,
            const Eigen::Ref<const Eigen::Vector3<Dtype>> &translation,
            const Eigen::Ref<const Eigen::MatrixX<Dtype>> &ranges) = 0;

        [[nodiscard]] virtual bool
        operator==(const AbstractSurfaceMapping3D & /*other*/) const {
            throw NotImplemented(__PRETTY_FUNCTION__);
        }

        [[nodiscard]] virtual bool
        operator!=(const AbstractSurfaceMapping3D &other) const {
            return !(*this == other);
        }
    };
}  // namespace erl::sdf_mapping
