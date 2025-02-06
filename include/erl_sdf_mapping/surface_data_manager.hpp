#pragma once

#include "erl_common/eigen.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace erl::sdf_mapping {

    template<int Dim>
    class SurfaceDataManager {

    public:
        using Vector = Eigen::Vector<double, Dim>;

        struct SurfaceData {

            Vector position = {0.0, 0.0, 0.0};
            Vector normal = {0.0, 0.0, 0.0};
            double var_position = 0.0;
            double var_normal = 0.0;

            SurfaceData() = default;

            SurfaceData(Vector position, Vector normal, const double var_position, const double var_normal)
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

    private:
        std::vector<SurfaceData> m_entries_;
        std::unordered_set<std::size_t> m_available_indices_;
        std::size_t m_size_ = 0;

    public:
        SurfaceDataManager() = default;

        SurfaceDataManager(const SurfaceDataManager &) = default;
        SurfaceDataManager &
        operator=(const SurfaceDataManager &) = default;
        SurfaceDataManager(SurfaceDataManager &&) = default;
        SurfaceDataManager &
        operator=(SurfaceDataManager &&) = default;

        [[nodiscard]] std::size_t
        Size() const {
            return m_size_;
        }

        std::size_t
        AddEntry(const SurfaceData &entry) {
            if (m_available_indices_.empty()) {
                m_entries_.emplace_back(entry);
                ++m_size_;
                return m_size_ - 1;
            }

            const auto index = *m_available_indices_.begin();
            m_available_indices_.erase(index);
            m_entries_[index] = entry;
            ++m_size_;
            return index;
        }

        std::size_t
        AddEntry(const SurfaceData &&entry) {
            if (m_available_indices_.empty()) {
                m_entries_.emplace_back(entry);
                ++m_size_;
                return m_size_ - 1;
            }

            const auto index = *m_available_indices_.begin();
            m_available_indices_.erase(index);
            m_entries_[index] = entry;
            ++m_size_;
            return index;
        }

        void
        RemoveEntry(const std::size_t index) {
            ERL_ASSERTM(m_available_indices_.insert(index).second, "Index {} is already removed.", index);
            --m_size_;
        }

        SurfaceData &
        operator[](const std::size_t index) {
            ERL_DEBUG_ASSERT(m_available_indices_.find(index) == m_available_indices_.end(), "Index {} is not available.", index);
            return m_entries_[index];
        }

        const SurfaceData &
        operator[](const std::size_t index) const {
            ERL_DEBUG_ASSERT(m_available_indices_.find(index) == m_available_indices_.end(), "Index {} is not available.", index);
            return m_entries_[index];
        }

        const std::vector<SurfaceData> &
        GetEntries() const {
            return m_entries_;
        }

        void
        Clear() {
            m_entries_.clear();
            m_available_indices_.clear();
            m_size_ = 0;
        }

        std::unordered_map<std::size_t, std::size_t>
        Compact() {
            std::unordered_map<std::size_t, std::size_t> index_mapping;
            std::size_t new_index = 0;
            for (std::size_t i = 0; i < m_entries_.size(); ++i) {
                if (m_available_indices_.find(i) == m_available_indices_.end()) {
                    index_mapping[i] = new_index;
                    m_entries_[new_index] = m_entries_[i];
                    ++new_index;
                }
            }
            m_entries_.resize(new_index);
            m_available_indices_.clear();
            m_size_ = new_index;
            return index_mapping;
        }
    };

}  // namespace erl::sdf_mapping
