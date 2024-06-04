#pragma once

#include "node.hpp"

#include "erl_geometry/node_container.hpp"

namespace erl::sdf_mapping::gpis {

    template<int Dim>
    class GpisNodeContainer : public geometry::NodeContainer {

    public:
        using Node = GpisNode<Dim>;

        struct Setting : public common::Yamlable<Setting> {
            int capacity = 1;
            double min_squared_distance = 0.04;
        };

    private:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        std::vector<std::shared_ptr<geometry::Node>> m_nodes_ = {};

    public:
        static std::shared_ptr<GpisNodeContainer>
        Create(const std::shared_ptr<Setting> &setting) {
            return std::shared_ptr<GpisNodeContainer>(new GpisNodeContainer(setting));
        }

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const {
            return m_setting_;
        }

        [[nodiscard]] std::vector<int>
        GetNodeTypes() const override {
            return std::vector<int>{0};  // vector of one element 0
        }

        [[nodiscard]] std::string
        GetNodeTypeName(const int type) const override {
            static const char *names[] = {"kSurface"};
            ERL_DEBUG_ASSERT(type >= 0 && type < 1, "Invalid node type.");
            return names[type];
        }

        [[nodiscard]] std::size_t
        Capacity() const override {
            return m_setting_->capacity;
        }

        [[nodiscard]] std::size_t
        Capacity(int /*type*/) const override {
            return m_setting_->capacity;
        }

        [[nodiscard]] std::size_t
        Size() const override {
            return m_nodes_.size();
        }

        [[nodiscard]] std::size_t
        Size(int /*type*/) const override {
            return m_nodes_.size();
        }

        void
        Clear() override {
            Reset();
        }

        std::vector<std::shared_ptr<geometry::Node>>::iterator
        Begin(int /*type*/) override {
            return m_nodes_.begin();
        }

        [[nodiscard]] std::vector<std::shared_ptr<geometry::Node>>::const_iterator
        Begin(int /*type*/) const override {
            return m_nodes_.begin();
        }

        std::vector<std::shared_ptr<geometry::Node>>::iterator
        End(int /*type*/) override {
            return m_nodes_.end();
        }

        [[nodiscard]] std::vector<std::shared_ptr<geometry::Node>>::const_iterator
        End(int /*type*/) const override {
            return m_nodes_.end();
        }

        void
        CollectNodes(std::vector<std::shared_ptr<geometry::Node>> &out) const override {
            out.insert(out.end(), m_nodes_.begin(), m_nodes_.end());
        }

        void
        CollectNodesOfType(int /*type*/, std::vector<std::shared_ptr<geometry::Node>> &out) const override {
            out.insert(out.end(), m_nodes_.begin(), m_nodes_.end());
        }

        void
        CollectNodesOfTypeInAabb2D(int /*type*/, const geometry::Aabb2D &area, std::vector<std::shared_ptr<geometry::Node>> &nodes) const override {
            if (Dim == 3) { throw std::runtime_error("GpisNodeContainer::CollectNodesOfTypeInAabb2D: 3D not implemented"); }

            for (auto &node: m_nodes_) {
                if (area.contains(node->position)) { nodes.push_back(node); }
            }
        }

        void
        CollectNodesOfTypeInAabb3D(int /*type*/, const geometry::Aabb3D &area, std::vector<std::shared_ptr<geometry::Node>> &nodes) const override {
            if (Dim == 2) { throw std::runtime_error("GpisNodeContainer::CollectNodesOfTypeInAabb2D: 2D not implemented"); }

            for (auto &node: m_nodes_) {
                if (area.contains(node->position)) { nodes.push_back(node); }
            }
        }

        bool
        Insert(const std::shared_ptr<geometry::Node> &node, bool &too_close) override {
            too_close = false;

            // must compute too_close before checking capacity, otherwise the node may be inserted into another container.
            if (!m_nodes_.empty()) {  // not empty
                too_close = std::any_of(m_nodes_.begin(), m_nodes_.end(), [&node, this](const std::shared_ptr<geometry::Node> &stored_node) {
                    return (stored_node->position - node->position).squaredNorm() < m_setting_->min_squared_distance;
                });
                if (too_close) { return false; }
            }

            if (m_nodes_.size() >= static_cast<std::size_t>(m_setting_->capacity)) { return false; }
            m_nodes_.push_back(node);
            return true;
        }

        bool
        Remove(const std::shared_ptr<const geometry::Node> &node) override {
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
        explicit GpisNodeContainer(std::shared_ptr<Setting> setting)
            : m_setting_(std::move(setting)) {
            Reset();
        }

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
