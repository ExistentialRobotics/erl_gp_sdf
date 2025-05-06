#pragma once

#include "erl_common/exception.hpp"
#include "erl_common/factory_pattern.hpp"
#include "erl_common/yaml.hpp"

namespace erl::sdf_mapping {

    class AbstractSurfaceMapping {
    public:
        using Factory = common::FactoryPattern<AbstractSurfaceMapping, false, false, const std::shared_ptr<common::YamlableBase> &>;
        virtual ~AbstractSurfaceMapping() = default;

        // Comparison
        [[nodiscard]] virtual bool
        operator==(const AbstractSurfaceMapping & /*other*/) const {
            throw NotImplemented(__PRETTY_FUNCTION__);
        }

        [[nodiscard]] bool
        operator!=(const AbstractSurfaceMapping &other) const {
            return !(*this == other);
        }

        // IO

        [[nodiscard]] virtual bool
        Write(std::ostream &s) const = 0;

        [[nodiscard]] virtual bool
        Read(std::istream &s) = 0;

        // Factory pattern
        template<typename Derived = AbstractSurfaceMapping>
        static std::shared_ptr<Derived>
        Create(const std::string &mapping_type, const std::shared_ptr<common::YamlableBase> &setting) {
            return Factory::GetInstance().Create(mapping_type, setting);
        }

        template<typename Derived>
        static std::enable_if_t<std::is_base_of_v<AbstractSurfaceMapping, Derived>, bool>
        Register(std::string mapping_type = "") {
            return Factory::GetInstance().Register<Derived>(
                mapping_type,
                [](const std::shared_ptr<common::YamlableBase> &setting) -> std::shared_ptr<AbstractSurfaceMapping> {
                    auto mapping_setting = std::dynamic_pointer_cast<typename Derived::Setting>(setting);
                    if (setting == nullptr) { mapping_setting = std::make_shared<typename Derived::Setting>(); }
                    if (mapping_setting == nullptr) {
                        ERL_WARN("Failed to cast setting to {}", type_name<typename Derived::Setting>());
                        return nullptr;
                    }
                    return std::make_shared<Derived>(mapping_setting);
                });
        }
    };
}  // namespace erl::sdf_mapping
