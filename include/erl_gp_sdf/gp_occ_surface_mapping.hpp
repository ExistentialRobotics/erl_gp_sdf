#pragma once

#include "abstract_surface_mapping.hpp"
#include "surface_data_manager.hpp"

#include "erl_gaussian_process/lidar_gp_2d.hpp"
#include "erl_gaussian_process/range_sensor_gp_3d.hpp"
#include "erl_geometry/occupancy_octree.hpp"
#include "erl_geometry/occupancy_quadtree.hpp"

namespace erl::gp_sdf {

    template<typename Dtype, int Dim>
    class GpOccSurfaceMapping : public AbstractSurfaceMapping<Dtype, Dim> {
        static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3.");

    public:
        using Super = AbstractSurfaceMapping<Dtype, Dim>;
        using typename Super::Aabb;
        using typename Super::Key;
        using typename Super::KeySet;
        using typename Super::KeyVector;
        using typename Super::MatrixDX;
        using typename Super::MatrixX;
        using typename Super::Ranges;
        using typename Super::Rotation;
        using typename Super::SurfData;
        using typename Super::SurfDataManager;
        using typename Super::Translation;
        using typename Super::VectorD;
        using typename Super::VectorX;

        using Tree = std::conditional_t<
            Dim == 2,
            geometry::OccupancyQuadtree<Dtype>,
            geometry::OccupancyOctree<Dtype>>;
        using TreeNode = std::conditional_t<  //
            Dim == 2,
            geometry::OccupancyQuadtreeNode,
            geometry::OccupancyOctreeNode>;

        // other types
        using SensorGp = std::conditional_t<
            Dim == 2,
            gaussian_process::LidarGaussianProcess2D<Dtype>,
            gaussian_process::RangeSensorGaussianProcess3D<Dtype>>;
        using SensorGpSetting = typename SensorGp::Setting;
        using TreeSetting = typename Tree::Setting;
        using SurfIndices0Type = absl::flat_hash_map<Key, std::size_t>;
        using SurfIndices1Type = absl::flat_hash_map<Key, absl::flat_hash_map<int, std::size_t>>;

        // eigen types
        using Scalar = Eigen::Matrix<Dtype, 1, 1>;

        struct Setting : public common::Yamlable<Setting> {
            struct ComputeVariance : common::Yamlable<ComputeVariance> {
                // position variance to set when the estimated gradient is almost zero.
                Dtype zero_gradient_position_var = 1.;
                // gradient variance to set when the estimated gradient is almost zero.
                Dtype zero_gradient_gradient_var = 1.;
                Dtype position_var_alpha = 0.01f;   // scaling of the position variance.
                Dtype direction_var_alpha = 0.01f;  // scaling of the direction variance.
                Dtype min_distance_var = 1.0f;      // allowed minimum distance variance.
                Dtype max_distance_var = 100.0f;    // allowed maximum distance variance.
                Dtype min_gradient_var = 0.01f;     // allowed minimum gradient variance.
                Dtype max_gradient_var = 1.0f;      // allowed maximum gradient variance.

                ERL_REFLECT_SCHEMA(
                    ComputeVariance,
                    ERL_REFLECT_MEMBER(ComputeVariance, zero_gradient_position_var),
                    ERL_REFLECT_MEMBER(ComputeVariance, zero_gradient_gradient_var),
                    ERL_REFLECT_MEMBER(ComputeVariance, position_var_alpha),
                    ERL_REFLECT_MEMBER(ComputeVariance, direction_var_alpha),
                    ERL_REFLECT_MEMBER(ComputeVariance, min_distance_var),
                    ERL_REFLECT_MEMBER(ComputeVariance, max_distance_var),
                    ERL_REFLECT_MEMBER(ComputeVariance, min_gradient_var),
                    ERL_REFLECT_MEMBER(ComputeVariance, max_gradient_var));
            };

            struct UpdateTree : common::Yamlable<UpdateTree> {
                bool with_count = false;
                bool parallel = true;
                bool lazy_eval = true;
                bool discrete = true;

                ERL_REFLECT_SCHEMA(
                    UpdateTree,
                    ERL_REFLECT_MEMBER(UpdateTree, with_count),
                    ERL_REFLECT_MEMBER(UpdateTree, parallel),
                    ERL_REFLECT_MEMBER(UpdateTree, lazy_eval),
                    ERL_REFLECT_MEMBER(UpdateTree, discrete));
            };

            struct UpdateMapPoints : common::Yamlable<UpdateMapPoints> {
                int max_adjust_tries = 10;
                // points of OCC smaller than this value are considered unobservable.
                // i.e., inside the object.
                Dtype min_observable_occ = -0.1f;
                Dtype min_position_var = 0.001f;  // minimum position variance.
                Dtype min_gradient_var = 0.001f;  // minimum gradient variance.

                // maximum absolute value of surface points' OCC, which should be zero ideally.
                Dtype max_surface_abs_occ = 0.02f;
                // no Bayes Update if the maximum valid gradient variance is above this threshold.
                Dtype max_valid_gradient_var = 0.5f;
                // discard it if the position variance by Bayes Update is above this threshold.
                Dtype max_bayes_position_var = 1.0f;
                // discard it if the gradient variance by Bayes Update is above this threshold.
                Dtype max_bayes_gradient_var = 0.6f;
                // maximum number of points to update in one map update.
                long max_num_points = 100000;

                ERL_REFLECT_SCHEMA(
                    UpdateMapPoints,
                    ERL_REFLECT_MEMBER(UpdateMapPoints, max_adjust_tries),
                    ERL_REFLECT_MEMBER(UpdateMapPoints, min_observable_occ),
                    ERL_REFLECT_MEMBER(UpdateMapPoints, min_position_var),
                    ERL_REFLECT_MEMBER(UpdateMapPoints, min_gradient_var),
                    ERL_REFLECT_MEMBER(UpdateMapPoints, max_surface_abs_occ),
                    ERL_REFLECT_MEMBER(UpdateMapPoints, max_valid_gradient_var),
                    ERL_REFLECT_MEMBER(UpdateMapPoints, max_bayes_position_var),
                    ERL_REFLECT_MEMBER(UpdateMapPoints, max_bayes_gradient_var),
                    ERL_REFLECT_MEMBER(UpdateMapPoints, max_num_points));
            };

            ComputeVariance compute_variance;
            UpdateTree update_tree;
            UpdateMapPoints update_map_points;
            std::shared_ptr<SensorGpSetting> sensor_gp = std::make_shared<SensorGpSetting>();
            std::shared_ptr<TreeSetting> tree = std::make_shared<TreeSetting>();
            // resolution to track the surface points; when <= 0, each leaf node contains only one
            // surface point.
            Dtype surface_resolution = 0.01f;
            Dtype scaling = 1.0f;         // internal scaling factor.
            Dtype perturb_delta = 0.01f;  // perturbation delta for gradient estimation.

            // a gradient with norm below this threshold is considered zero.
            Dtype zero_gradient_threshold = 1.e-15f;
            bool update_occupancy = true;  // whether to update the occupancy of the occupancy tree.
            uint32_t cluster_depth = 14;

            ERL_REFLECT_SCHEMA(
                Setting,
                ERL_REFLECT_MEMBER(Setting, compute_variance),
                ERL_REFLECT_MEMBER(Setting, update_tree),
                ERL_REFLECT_MEMBER(Setting, update_map_points),
                ERL_REFLECT_MEMBER(Setting, sensor_gp),
                ERL_REFLECT_MEMBER(Setting, tree),
                ERL_REFLECT_MEMBER(Setting, surface_resolution),
                ERL_REFLECT_MEMBER(Setting, scaling),
                ERL_REFLECT_MEMBER(Setting, perturb_delta),
                ERL_REFLECT_MEMBER(Setting, zero_gradient_threshold),
                ERL_REFLECT_MEMBER(Setting, update_occupancy),
                ERL_REFLECT_MEMBER(Setting, cluster_depth));
        };

        class SurfaceDataIterator {
            GpOccSurfaceMapping *m_mapping_;
            bool m_use0_ = true;
            typename SurfIndices0Type::iterator m_it0_;
            typename SurfIndices1Type::iterator m_it1_;
            absl::flat_hash_map<int, std::size_t>::iterator m_it2_;

        public:
            explicit SurfaceDataIterator(GpOccSurfaceMapping *mapping);

            SurfaceDataIterator(const SurfaceDataIterator &other) = default;
            SurfaceDataIterator &
            operator=(const SurfaceDataIterator &other) = default;
            SurfaceDataIterator(SurfaceDataIterator &&other) = default;
            SurfaceDataIterator &
            operator=(SurfaceDataIterator &&other) = default;

            [[nodiscard]] bool
            operator==(const SurfaceDataIterator &other) const;

            [[nodiscard]] bool
            operator!=(const SurfaceDataIterator &other) const;

            SurfData &
            operator*();

            SurfData *
            operator->();

            SurfaceDataIterator &
            operator++();

            SurfaceDataIterator
            operator++(int);
        };

    private:
        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::shared_ptr<SensorGp> m_sensor_gp_ = nullptr;
        std::shared_ptr<Tree> m_tree_ = nullptr;
        // strides for the grid indices
        Eigen::Vector<int, Dim> m_strides_ = Eigen::Vector<int, Dim>::Zero();

        struct Index {
            Key key;
            int grid_index = 0;
            std::size_t surf_index = 0;
        };

        // index, updated, to_remove, new_index
        std::vector<std::tuple<Index, bool, bool, std::optional<Index>>> m_surf_in_aabb_;

        // key -> surface data index (used when the surface resolution is <= 0)
        SurfIndices0Type m_surf_indices0_;
        // key -> [grid_min, (grid index -> surface data index)]
        SurfIndices1Type m_surf_indices1_;

        Eigen::Matrix<Dtype, Dim, 2 * Dim> m_pos_perturb_ = {};
        Dtype m_surface_resolution_inv_ = 0.0f;  // inverse of the tree resolution
        KeySet m_changed_keys_ = {};

    public:
        explicit GpOccSurfaceMapping(std::shared_ptr<Setting> setting);

        [[nodiscard]] std::shared_ptr<const Setting>
        GetSetting() const;

        [[nodiscard]] std::shared_ptr<const SensorGp>
        GetSensorGp() const;

        [[nodiscard]] std::shared_ptr<const Tree>
        GetTree() const;

        /**
         * Update the surface mapping.
         * @param rotation The rotation of the sensor. For 2D, it is a 2x2 matrix. For 3D, it is a
         * 3x3 matrix.
         * @param translation The translation of the sensor. For 2D, it is a 2x1 vector. For 3D, it
         * is a 3x1 vector.
         * @param ranges The observation of ranges assumed organized in the order of ray angles. For
         * 2D, it is a vector of ranges. For 3D, it is a matrix of ranges.
         * @return true if the update is successful.
         */
        bool
        Update(
            const Eigen::Ref<const Rotation> &rotation,
            const Eigen::Ref<const Translation> &translation,
            const Eigen::Ref<const Ranges> &ranges);

        SurfaceDataIterator
        BeginSurfaceData();

        SurfaceDataIterator
        EndSurfaceData();

        // implement the methods required by AbstractSurfaceMapping

        bool
        Update(
            const Eigen::Ref<const Rotation> &rotation,
            const Eigen::Ref<const Translation> &translation,
            const Eigen::Ref<const Ranges> &scan,
            bool are_points,
            bool are_local) override;

        [[nodiscard]] Dtype
        GetScaling() const override;

        [[nodiscard]] Dtype
        GetClusterSize() const override;

        [[nodiscard]] long
        GetClusterKeySize() const override;

        [[nodiscard]] bool
        HasCluster(const Key &key) const override;

        [[nodiscard]] VectorD
        GetClusterCenter(const Key &key) const override;

        [[nodiscard]] const KeySet &
        GetChangedClusters() const override;

        void
        ClearChangedClusters() override;

        [[nodiscard]] KeySet
        GetAllClusters() const override;

        [[nodiscard]] Key
        GetClusterKey(const Eigen::Ref<const VectorD> &pos) const override;

        void
        IterateClustersInAabb(const Aabb &aabb, std::function<void(const Key &)> callback)
            const override;

        [[nodiscard]] const std::vector<std::size_t> &
        GetUnusedSurfaceDataIndices() const override;

        [[nodiscard]] const std::vector<SurfData> &
        GetSurfaceDataBuffer() const override;

        [[nodiscard]] std::size_t
        CollectSurfaceDataInAabb(
            const Aabb &aabb,
            std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices) const override;

        [[nodiscard]] std::size_t
        CollectSurfaceDataFromCluster(
            const Key &key,
            std::vector<std::size_t> &surface_data_indices) const override;

        void
        FlushSurfaceDataCache() override {}

        [[nodiscard]] Aabb
        GetMapBoundary() const override;

        [[nodiscard]] bool
        IsInFreeSpace(const MatrixDX &positions, Eigen::VectorXb &in_free_space) const override;

        [[nodiscard]] bool
        operator==(const Super &other) const override;

        [[nodiscard]] bool
        Write(std::ostream &stream) const override;

        [[nodiscard]] bool
        Read(std::istream &stream) override;

    private:
        static std::pair<Dtype, Dtype>
        Cartesian2Polar(Dtype x, Dtype y);

        void
        UpdateMapPoints0();

        void
        UpdateMapPoints1();

        Dtype
        UpdateMapPoint(SurfData &surface_data, bool &updated, bool &to_remove);

        [[nodiscard]] std::pair<Key, int>
        ComputeSurfaceIndex1(const VectorD &pos_global) const;

        template<int D = Dim>
        std::enable_if_t<D == 2>
        UpdateGradient(Dtype var_new, Dtype var_sum, const VectorD &grad_old, VectorD &grad_new);

        template<int D = Dim>
        std::enable_if_t<D == 3>
        UpdateGradient(Dtype var_new, Dtype var_sum, const VectorD &grad_old, VectorD &grad_new);

        void
        UpdateOccupancy();

        void
        AddNewMeasurement0();

        void
        AddNewMeasurement1();

        void
        RecordChangedKey(const Key &key);

        bool
        ComputeGradient1(
            const VectorD &pos_local,
            VectorD &gradient,
            Dtype &occ_mean,
            Dtype &distance_var);

        bool
        ComputeGradient2(
            const Eigen::Ref<const VectorD> &pos_local,
            VectorD &gradient,
            Dtype &occ_mean);

        void
        ComputeVariance(
            const Eigen::Ref<const VectorD> &pos_local,
            const VectorD &grad_local,
            const Dtype &distance,
            const Dtype &distance_var,
            const Dtype &occ_mean_abs,
            const Dtype &occ_abs,
            bool new_point,
            Dtype &var_position,
            Dtype &var_gradient) const;
    };

    using GpOccSurfaceMapping3Dd = GpOccSurfaceMapping<double, 3>;
    using GpOccSurfaceMapping3Df = GpOccSurfaceMapping<float, 3>;
    using GpOccSurfaceMapping2Dd = GpOccSurfaceMapping<double, 2>;
    using GpOccSurfaceMapping2Df = GpOccSurfaceMapping<float, 2>;

    extern template class GpOccSurfaceMapping<double, 3>;
    extern template class GpOccSurfaceMapping<float, 3>;
    extern template class GpOccSurfaceMapping<double, 2>;
    extern template class GpOccSurfaceMapping<float, 2>;

}  // namespace erl::gp_sdf
