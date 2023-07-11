#pragma once
#include "erl_geometry/occupancy_quadtree_node.hpp"

namespace erl::sdf_mapping {

    class SurfaceMappingQuadtreeNode : public geometry::OccupancyQuadtreeNode {

    public:
        struct SurfaceData {
            Eigen::Vector2d position = {0., 0.};
            Eigen::Vector2d normal = {0., 0.};
            double var_position = 0.;
            double var_normal = 0.;
            long num_hit_rays = 0;
            double hit_ray_direction_angle_resolution = M_PI / 180.0;
            long max_num_hit_rays = long(2 * M_PI / hit_ray_direction_angle_resolution);
            Eigen::VectorXb hit_ray_mask = {};
            Eigen::Matrix2d hit_ray_start_pts = {};

            SurfaceData() = default;

            SurfaceData(Eigen::Vector2d position, Eigen::Vector2d normal, double var_position, double var_normal)
                : position(std::move(position)),
                  normal(std::move(normal)),
                  var_position(var_position),
                  var_normal(var_normal) {}

            SurfaceData(
                Eigen::Vector2d position,
                Eigen::Vector2d normal,
                double var_position,
                double var_normal,
                const Eigen::Ref<const Eigen::Vector2d> &hit_ray_start_point)
                : SurfaceData(std::move(position), std::move(normal), var_position, var_normal) {
                hit_ray_mask.setConstant(max_num_hit_rays, false);
                hit_ray_start_pts.setConstant(2, max_num_hit_rays, 0.);
                RecordHitRay(hit_ray_start_point);
            }

            void
            RecordHitRay(const Eigen::Ref<const Eigen::Vector2d> &ray_start_point, bool replace = false) {
                Eigen::Vector2d ray_direction = ray_start_point - position;
                double ray_direction_angle = std::atan2(ray_direction.y(), ray_direction.x());
                auto ray_direction_angle_idx = long((ray_direction_angle + M_PI) / hit_ray_direction_angle_resolution);
                long n_directions = hit_ray_mask.size();
                if (ray_direction_angle_idx == n_directions) { ray_direction_angle_idx = 0; }
                if (!hit_ray_mask[ray_direction_angle_idx]) {
                    hit_ray_start_pts.col(ray_direction_angle_idx) = ray_start_point;
                    hit_ray_mask[ray_direction_angle_idx] = true;
                    num_hit_rays++;
                } else if (replace) {
                    hit_ray_start_pts.col(ray_direction_angle_idx) = ray_start_point;
                }
            }
        };

        SurfaceMappingQuadtreeNode()
            : OccupancyQuadtreeNode() {}

        void
        SetSurfaceData(Eigen::Vector2d position, Eigen::Vector2d normal, double var_position, double var_normal) {
            m_data_ = std::make_shared<SurfaceData>(std::move(position), std::move(normal), var_position, var_normal);
        }

        void
        SetSurfaceData(
            Eigen::Vector2d position,
            Eigen::Vector2d normal,
            double var_position,
            double var_normal,
            const Eigen::Ref<const Eigen::Vector2d> &hit_ray_start_point) {
            m_data_ = std::make_shared<SurfaceData>(std::move(position), std::move(normal), var_position, var_normal, hit_ray_start_point);
        }

        void
        SetSurfaceData(const std::shared_ptr<SurfaceData> &data) {
            m_data_ = data;
        }

        std::shared_ptr<SurfaceData>
        GetSurfaceData() {
            return m_data_;
        }

        [[nodiscard]] std::shared_ptr<const SurfaceData>
        GetSurfaceData() const {
            return m_data_;
        }

        void
        ResetSurfaceData() {
            m_data_.reset();
        }

        [[nodiscard]] bool
        AllowUpdateLogOdds(double delta) const override {
            return delta > 0 || m_data_ == nullptr;
        }

        // [[nodiscard]] bool
        // LogOddsLocked() const override {
        //     return m_data_ != nullptr;  // if m_data_ is not null, then the node's log odds should be locked such that m_data_ is not deleted.
        // }

        std::istream &
        ReadData(std::istream &s) override {
            geometry::OccupancyQuadtreeNode::ReadData(s);
            char has_data;
            s.read(&has_data, sizeof(char));
            if (has_data == 0) {
                m_data_.reset();
                return s;
            }
            m_data_ = std::make_shared<SurfaceData>();
            s.read(reinterpret_cast<char *>(m_data_->position.data()), sizeof(double) * 2);
            s.read(reinterpret_cast<char *>(m_data_->normal.data()), sizeof(double) * 2);
            s.read(reinterpret_cast<char *>(&m_data_->var_position), sizeof(double));
            s.read(reinterpret_cast<char *>(&m_data_->var_normal), sizeof(double));
            s.read(reinterpret_cast<char *>(&m_data_->num_hit_rays), sizeof(long));
            s.read(reinterpret_cast<char *>(&m_data_->hit_ray_direction_angle_resolution), sizeof(double));
            s.read(reinterpret_cast<char *>(&m_data_->max_num_hit_rays), sizeof(long));
            ERL_ASSERTM(m_data_->num_hit_rays <= m_data_->max_num_hit_rays, "num_hit_rays > max_num_hit_rays");
            ERL_ASSERTM(m_data_->max_num_hit_rays > 0, "max_num_hit_rays <= 0");
            if (m_data_->num_hit_rays <= 0) { return s; }
            m_data_->hit_ray_mask.setConstant(m_data_->max_num_hit_rays, false);
            m_data_->hit_ray_start_pts.setConstant(2, m_data_->max_num_hit_rays, 0);
            s.read(reinterpret_cast<char *>(m_data_->hit_ray_mask.data()), std::streamsize(sizeof(bool) * m_data_->max_num_hit_rays));
            s.read(reinterpret_cast<char *>(m_data_->hit_ray_start_pts.data()), std::streamsize(sizeof(double) * 2 * m_data_->max_num_hit_rays));
            return s;
        }

        std::ostream &
        WriteData(std::ostream &s) const override {
            geometry::OccupancyQuadtreeNode::WriteData(s);
            if (m_data_ == nullptr) {
                s << char(0);
                return s;
            } else {
                s << char(1);
            }
            s.write(reinterpret_cast<const char *>(m_data_->position.data()), sizeof(double) * 2);
            s.write(reinterpret_cast<const char *>(m_data_->normal.data()), sizeof(double) * 2);
            s.write(reinterpret_cast<const char *>(&m_data_->var_position), sizeof(double));
            s.write(reinterpret_cast<const char *>(&m_data_->var_normal), sizeof(double));
            s.write(reinterpret_cast<const char *>(&m_data_->num_hit_rays), sizeof(long));
            s.write(reinterpret_cast<const char *>(&m_data_->hit_ray_direction_angle_resolution), sizeof(double));
            s.write(reinterpret_cast<const char *>(&m_data_->max_num_hit_rays), sizeof(long));
            if (m_data_->num_hit_rays <= 0) { return s; }
            ERL_ASSERTM(m_data_->max_num_hit_rays > 0, "max_num_hit_rays <= 0");
            ERL_ASSERTM(m_data_->num_hit_rays <= m_data_->max_num_hit_rays, "num_hit_rays > max_num_hit_rays");
            ERL_ASSERTM(m_data_->hit_ray_mask.size() == m_data_->max_num_hit_rays, "hit_ray_mask.size() != max_num_hit_rays");
            ERL_ASSERTM(m_data_->hit_ray_start_pts.cols() == m_data_->max_num_hit_rays, "hit_ray_start_pts.cols() != max_num_hit_rays");
            s.write(reinterpret_cast<const char *>(m_data_->hit_ray_mask.data()), std::streamsize(sizeof(bool) * m_data_->max_num_hit_rays));
            s.write(reinterpret_cast<const char *>(m_data_->hit_ray_start_pts.data()), std::streamsize(sizeof(double) * 2 * m_data_->max_num_hit_rays));
            return s;
        }

    private:
        std::shared_ptr<SurfaceData> m_data_ = nullptr;
    };
}  // namespace erl::sdf_mapping
