#pragma once

#include "node.hpp"

#include "erl_common/yaml.hpp"
#include "erl_geometry/aabb.hpp"

namespace erl::sdf_mapping::gpis {

    template<int Dim>
    class GpisNodeContainer {

    public:
        using Node = GpisNode<Dim>;

        struct Setting : public common::Yamlable<Setting> {
            int capacity = 1;
            double min_squared_distance = 0.04;
        };

    private:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        std::vector<std::shared_ptr<Node>> m_nodes_ = {};

    public:
        explicit GpisNodeContainer(std::shared_ptr<Setting> setting)
            : m_setting_(std::move(setting)) {
            Reset();
        }

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const {
            return m_setting_;
        }

        [[nodiscard]] std::vector<int>
        GetNodeTypes() const {
            return std::vector<int>{0};  // vector of one element 0
        }

        [[nodiscard]] std::string
        GetNodeTypeName(const int type) const {
            static const char *names[] = {"kSurface"};
            ERL_DEBUG_ASSERT(type >= 0 && type < 1, "Invalid node type.");
            return names[type];
        }

        [[nodiscard]] std::size_t
        Capacity() const {
            return m_setting_->capacity;
        }

        [[nodiscard]] std::size_t
        Capacity(int /*type*/) const {
            return m_setting_->capacity;
        }

        [[nodiscard]] bool
        Empty() const {
            return Size() == 0;
        }

        [[nodiscard]] bool
        Empty(const int type) const {
            return Size(type) == 0;
        }

        [[nodiscard]] bool
        Full() const {
            return Size() >= Capacity();
        }

        [[nodiscard]] bool
        Full(const int type) const {
            return Size(type) >= Capacity(type);
        }

        [[nodiscard]] std::size_t
        Size() const {
            return m_nodes_.size();
        }

        [[nodiscard]] std::size_t
        Size(int /*type*/) const {
            return m_nodes_.size();
        }

        void
        Clear() {
            Reset();
        }

        typename std::vector<std::shared_ptr<Node>>::iterator
        Begin(int /*type*/) {
            return m_nodes_.begin();
        }

        [[nodiscard]] typename std::vector<std::shared_ptr<Node>>::const_iterator
        Begin(int /*type*/) const {
            return m_nodes_.begin();
        }

        typename std::vector<std::shared_ptr<Node>>::iterator
        End(int /*type*/) {
            return m_nodes_.end();
        }

        [[nodiscard]] typename std::vector<std::shared_ptr<Node>>::const_iterator
        End(int /*type*/) const {
            return m_nodes_.end();
        }

        [[nodiscard]] std::vector<std::shared_ptr<Node>>
        CollectNodes() const {
            std::vector<std::shared_ptr<Node>> out;
            CollectNodes(out);
            return out;
        }

        void
        CollectNodes(std::vector<std::shared_ptr<Node>> &out) const {
            out.insert(out.end(), m_nodes_.begin(), m_nodes_.end());
        }

        [[nodiscard]] std::vector<std::shared_ptr<Node>>
        CollectNodesOfType(const int type) const {
            std::vector<std::shared_ptr<Node>> out;
            CollectNodesOfType(type, out);
            return out;
        }

        void
        CollectNodesOfType(int /*type*/, std::vector<std::shared_ptr<Node>> &out) const {
            out.insert(out.end(), m_nodes_.begin(), m_nodes_.end());
        }

        void
        CollectNodesOfTypeInAabb2D(int /*type*/, const geometry::Aabb2D &area, std::vector<std::shared_ptr<Node>> &nodes) const {
            if (Dim == 3) { throw std::runtime_error("GpisNodeContainer::CollectNodesOfTypeInAabb2D: 3D not implemented"); }

            for (auto &node: m_nodes_) {
                if (area.contains(node->position)) { nodes.push_back(node); }
            }
        }

        void
        CollectNodesOfTypeInAabb3D(int /*type*/, const geometry::Aabb3D &area, std::vector<std::shared_ptr<Node>> &nodes) const {
            if (Dim == 2) { throw std::runtime_error("GpisNodeContainer::CollectNodesOfTypeInAabb2D: 2D not implemented"); }

            for (auto &node: m_nodes_) {
                if (area.contains(node->position)) { nodes.push_back(node); }
            }
        }

        bool
        Insert(const std::shared_ptr<Node> &node, bool &too_close) {
            too_close = false;

            // must compute too_close before checking capacity, otherwise the node may be inserted into another container.
            if (!m_nodes_.empty()) {  // not empty
                too_close = std::any_of(m_nodes_.begin(), m_nodes_.end(), [&node, this](const std::shared_ptr<Node> &stored_node) {
                    return (stored_node->position - node->position).squaredNorm() < m_setting_->min_squared_distance;
                });
                if (too_close) { return false; }
            }

            if (m_nodes_.size() >= static_cast<std::size_t>(m_setting_->capacity)) { return false; }
            m_nodes_.push_back(node);
            return true;
        }

        bool
        Remove(const std::shared_ptr<const Node> &node) {
            if (m_nodes_.empty()) { return false; }
            auto end = m_nodes_.end();
            for (auto itr = m_nodes_.begin(); itr < end; ++itr) {
                if (**itr != *node) { continue; }
                *itr = *(--end);
                m_nodes_.pop_back();
                return true;
            }

            return false;
        }

    private:
        void
        Reset() {
            ERL_ASSERTM(m_setting_ != nullptr, "Setting is null");
            ERL_ASSERTM(m_setting_->capacity > 0, "Capacity must be greater than 0");
            ERL_ASSERTM(m_setting_->min_squared_distance > 0, "Min squared distance must be greater than 0");
            m_nodes_.clear();
            m_nodes_.reserve(this->m_setting_->capacity);
        }
    };

    using GpisNodeContainer2D = GpisNodeContainer<2>;
    using GpisNodeContainer3D = GpisNodeContainer<3>;
}  // namespace erl::sdf_mapping::gpis

// ReSharper disable CppInconsistentNaming
namespace YAML {

    using namespace erl::sdf_mapping::gpis;

    template<int Dim>
    struct ConvertGpisNodeContainerSetting {
        static Node
        encode(const typename GpisNodeContainer<Dim>::Setting &setting) {
            Node node;
            node["capacity"] = setting.capacity;
            node["min_squared_distance"] = setting.min_squared_distance;
            return node;
        }

        static bool
        decode(const Node &node, typename GpisNodeContainer<Dim>::Setting &setting) {
            if (!node.IsMap()) { return false; }
            setting.capacity = node["capacity"].as<int>();
            setting.min_squared_distance = node["min_squared_distance"].as<double>();
            return true;
        }
    };

    template<>
    struct convert<GpisNodeContainer2D::Setting> : public ConvertGpisNodeContainerSetting<2> {};

    template<>
    struct convert<GpisNodeContainer3D::Setting> : public ConvertGpisNodeContainerSetting<3> {};
}  // namespace YAML

// ReSharper restore CppInconsistentNaming
