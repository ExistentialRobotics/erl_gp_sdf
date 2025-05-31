#pragma once

#include "sdf_gp.hpp"

template<>
struct YAML::convert<erl::sdf_mapping::SignMethod> {
    static Node
    encode(const erl::sdf_mapping::SignMethod &method) {
        Node node;
        switch (method) {
            case erl::sdf_mapping::kNone:
                node = "kNone";
                break;
            case erl::sdf_mapping::kSignGp:
                node = "kSignGp";
                break;
            case erl::sdf_mapping::kNormalGp:
                node = "kNormalGp";
                break;
            case erl::sdf_mapping::kExternal:
                node = "kExternal";
                break;
            case erl::sdf_mapping::kHybrid:
                node = "kHybrid";
                break;
            default:
                ERL_FATAL("Unknown SignMethod: {}", static_cast<int>(method));
        }
        return node;
    }

    static bool
    decode(const Node &node, erl::sdf_mapping::SignMethod &method) {
        if (const std::string method_str = node.as<std::string>();  //
            method_str == "kNone") {
            method = erl::sdf_mapping::kNone;
        } else if (method_str == "kSignGp") {
            method = erl::sdf_mapping::kSignGp;
        } else if (method_str == "kNormalGp") {
            method = erl::sdf_mapping::kNormalGp;
        } else if (method_str == "kExternal") {
            method = erl::sdf_mapping::kExternal;
        } else if (method_str == "kHybrid") {
            method = erl::sdf_mapping::kHybrid;
        } else {
            ERL_FATAL("Unknown SignMethod: {}", method_str);
        }
        return true;
    }
};

namespace erl::sdf_mapping {

    template<typename Dtype>
    YAML::Node
    SdfGaussianProcessSetting<Dtype>::YamlConvertImpl::encode(
        const SdfGaussianProcessSetting &setting) {
        YAML::Node node;
        ERL_YAML_SAVE_ATTR(node, setting, sign_method);
        ERL_YAML_SAVE_ATTR(node, setting, hybrid_sign_methods);
        ERL_YAML_SAVE_ATTR(node, setting, hybrid_sign_threshold);
        ERL_YAML_SAVE_ATTR(node, setting, normal_scale);
        ERL_YAML_SAVE_ATTR(node, setting, softmin_temperature);
        ERL_YAML_SAVE_ATTR(node, setting, sign_gp_offset_distance);
        ERL_YAML_SAVE_ATTR(node, setting, edf_gp_offset_distance);
        ERL_YAML_SAVE_ATTR(node, setting, sign_gp);
        ERL_YAML_SAVE_ATTR(node, setting, edf_gp);
        return node;
    }

    template<typename Dtype>
    bool
    SdfGaussianProcessSetting<Dtype>::YamlConvertImpl::decode(
        const YAML::Node &node,
        SdfGaussianProcessSetting &setting) {
        if (!node.IsMap()) { return false; }
        ERL_YAML_LOAD_ATTR(node, setting, sign_method);
        ERL_YAML_LOAD_ATTR(node, setting, hybrid_sign_methods);
        ERL_YAML_LOAD_ATTR(node, setting, hybrid_sign_threshold);
        ERL_YAML_LOAD_ATTR(node, setting, normal_scale);
        ERL_YAML_LOAD_ATTR(node, setting, softmin_temperature);
        ERL_YAML_LOAD_ATTR(node, setting, sign_gp_offset_distance);
        ERL_YAML_LOAD_ATTR(node, setting, edf_gp_offset_distance);
        ERL_YAML_LOAD_ATTR(node, setting, sign_gp);
        ERL_YAML_LOAD_ATTR(node, setting, edf_gp);
        return true;
    }

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim>::SdfGaussianProcess(std::shared_ptr<Setting> setting_)
        : setting(std::move(setting_)) {
        ERL_ASSERTM(setting != nullptr, "Setting is null.");
        use_normal_gp =
            setting->sign_method == kNormalGp ||
            (setting->sign_method == kHybrid && (setting->hybrid_sign_methods.first == kNormalGp ||
                                                 setting->hybrid_sign_methods.second == kNormalGp));
    }

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim>::SdfGaussianProcess(const SdfGaussianProcess &other)
        : setting(other.setting),
          active(other.active),
          outdated(other.outdated),
          use_normal_gp(other.use_normal_gp),
          offset_distance(other.offset_distance),
          locked_for_test(other.locked_for_test.load()),
          position(other.position),
          half_size(other.half_size) {
        if (other.sign_gp != nullptr) { sign_gp = std::make_shared<SignGp>(*other.sign_gp); }
        if (other.edf_gp != nullptr) { edf_gp = std::make_shared<EdfGp>(*other.edf_gp); }
    }

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim>::SdfGaussianProcess(SdfGaussianProcess &&other) noexcept
        : setting(other.setting),
          active(other.active),
          outdated(other.outdated),
          use_normal_gp(other.use_normal_gp),
          offset_distance(other.offset_distance),
          locked_for_test(other.locked_for_test.load()),
          position(std::move(other.position)),
          half_size(other.half_size),
          sign_gp(std::move(other.sign_gp)),
          edf_gp(std::move(other.edf_gp)) {}

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim> &
    SdfGaussianProcess<Dtype, Dim>::operator=(const SdfGaussianProcess &other) {
        if (this == &other) { return *this; }
        setting = other.setting;
        active = other.active;
        outdated = other.outdated;
        use_normal_gp = other.use_normal_gp;
        offset_distance = other.offset_distance;
        locked_for_test = other.locked_for_test.load();
        position = other.position;
        half_size = other.half_size;
        if (other.sign_gp != nullptr) {
            sign_gp = std::make_shared<SignGp>(*other.sign_gp);
        } else {
            sign_gp = nullptr;
        }
        if (other.edf_gp != nullptr) {
            edf_gp = std::make_shared<EdfGp>(*other.edf_gp);
        } else {
            edf_gp = nullptr;
        }
        return *this;
    }

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim> &
    SdfGaussianProcess<Dtype, Dim>::operator=(SdfGaussianProcess &&other) noexcept {
        if (this == &other) { return *this; }
        setting = other.setting;
        active = other.active;
        outdated = other.outdated;
        use_normal_gp = other.use_normal_gp;
        offset_distance = other.offset_distance;
        locked_for_test = other.locked_for_test.load();
        position = other.position;
        half_size = other.half_size;
        sign_gp = std::move(other.sign_gp);
        edf_gp = std::move(other.edf_gp);
        return *this;
    }

    template<typename Dtype, int Dim>
    void
    SdfGaussianProcess<Dtype, Dim>::Activate() {
        if (sign_gp == nullptr && (setting->sign_method == kSignGp ||
                                   (setting->sign_method == kHybrid &&
                                    (setting->hybrid_sign_methods.first == kSignGp ||
                                     setting->hybrid_sign_methods.second == kSignGp)))) {
            sign_gp = std::make_shared<SignGp>(setting->sign_gp);
        }
        if (edf_gp == nullptr) { edf_gp = std::make_shared<EdfGp>(setting->edf_gp); }
        active = true;
    }

    template<typename Dtype, int Dim>
    void
    SdfGaussianProcess<Dtype, Dim>::Deactivate() {
        active = false;
    }

    template<typename Dtype, int Dim>
    void
    SdfGaussianProcess<Dtype, Dim>::MarkOutdated() {
        outdated = true;
    }

    template<typename Dtype, int Dim>
    std::size_t
    SdfGaussianProcess<Dtype, Dim>::GetMemoryUsage() const {
        std::size_t memory_usage = sizeof(SdfGaussianProcess);
        if (edf_gp != nullptr) { memory_usage += edf_gp->GetMemoryUsage(); }
        if (sign_gp != nullptr) { memory_usage += sign_gp->GetMemoryUsage(); }
        return memory_usage;
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::Intersects(
        const VectorD &other_position,
        const Dtype other_half_size) const {
        for (int i = 0; i < Dim; ++i) {
            if (std::abs(position[i] - other_position[i]) > half_size + other_half_size) {
                return false;
            }
        }
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::Intersects(
        const VectorD &other_position,
        const VectorD &other_half_sizes) const {
        for (int i = 0; i < Dim; ++i) {
            if (std::abs(position[i] - other_position[i]) > half_size + other_half_sizes[i]) {
                return false;
            }
        }
        return true;
    }

    template<typename Dtype, int Dim>
    void
    SdfGaussianProcess<Dtype, Dim>::LoadSurfaceData(
        std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices,
        const std::vector<SurfaceData<Dtype, Dim>> &surface_data_vec,
        Dtype sensor_noise,
        Dtype max_valid_gradient_var,
        Dtype invalid_position_var) {
        this->offset_distance = offset_distance;
        if (sign_gp != nullptr) {
            sign_gp->template LoadSurfaceData<Dim>(
                surface_data_indices,
                surface_data_vec,
                position,
                setting->sign_gp_offset_distance,
                sensor_noise,
                max_valid_gradient_var,
                invalid_position_var);
        }
        if (edf_gp != nullptr) {
            edf_gp->template LoadSurfaceData<Dim>(
                surface_data_indices,
                surface_data_vec,
                position,
                use_normal_gp,
                setting->normal_scale,
                setting->edf_gp_offset_distance,
                sensor_noise,
                max_valid_gradient_var,
                invalid_position_var);
        }
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::IsTrained() const {
        if (edf_gp != nullptr) { return edf_gp->IsTrained(); }
        return false;
    }

    template<typename Dtype, int Dim>
    void
    SdfGaussianProcess<Dtype, Dim>::Train() {
        ERL_DEBUG_ASSERT(active, "SdfGaussianProcess is not active.");
        if (sign_gp != nullptr) { sign_gp->Train(); }
        if (edf_gp != nullptr) { edf_gp->Train(); }
        outdated = false;  // mark as not outdated after training
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::Test(
        const VectorD &test_position,                     // single position to test
        Eigen::Ref<Eigen::Vector<Dtype, 2 * Dim + 1>> f,  // sdf, sdf_gradient, normal
        Eigen::Ref<Eigen::Vector<Dtype, Dim + 1>> var,
        Eigen::Ref<Eigen::Vector<Dtype, Dim *(Dim + 1) / 2>> covariance,
        const Dtype external_sign,
        const bool compute_gradient,
        const bool compute_gradient_variance,
        const bool compute_covariance,
        const bool use_gp_covariance) const {

        // edf, sign and variance of sdf are always computed.
        // when compute_gradient is true, the gradient of sdf is computed.
        // when compute_gradient_variance is true, the variance of sdf gradient is computed.
        // when compute_covariance is true, the covariance of sdf and sdf gradient is computed.
        // when use_gp_covariance is true, edf_gp is used to compute the covariance.
        // otherwise, the covariance is computed using the training samples.

        ERL_DEBUG_ASSERT(active, "SdfGaussianProcess is not active.");

        // compute edf
        Dtype &sdf = f[0];
        auto edf_result = *std::reinterpret_pointer_cast<typename EdfGp::TestResult>(
            edf_gp->Test(test_position, compute_gradient || use_normal_gp));
        edf_result.GetMean(0, 0, sdf);
        if (!std::isfinite(sdf)) {  // invalid sdf
            sdf = 0.0f;
            var[0] = 1e6f;  // set a large variance if sdf is invalid
            return false;
        }

        // compute sign
        SignMethod sign_method = setting->sign_method;
        if (sign_method == kHybrid) {
            if (setting->hybrid_sign_threshold < sdf) {
                sign_method = setting->hybrid_sign_methods.first;
            } else {
                sign_method = setting->hybrid_sign_methods.second;
            }
        }
        Dtype sign = 1.0f;
        bool sdf_gradient_computed = false;
        auto sdf_gradient = f.template segment<Dim>(1);
        switch (sign_method) {
            case kSignGp: {
                ERL_DEBUG_ASSERT(sign_gp != nullptr, "sign_gp is not initialized.");
                (*std::reinterpret_pointer_cast<typename SignGp::TestResult>(
                     sign_gp->Test(test_position, false)))
                    .GetMean(0, 0, sign);
                break;
            }
            case kNormalGp: {
                auto normal = f.template tail<Dim>();
                if (!edf_result.template GetGradientD<Dim>(0, 0, sdf_gradient.data())) {
                    var[0] = 1e6f;
                    return false;
                }
                for (long i = 1; i <= Dim; ++i) { edf_result.GetMean(0, i, normal[i - 1]); }
                sign = sdf_gradient.dot(normal);
                sdf_gradient_computed = true;
                break;
            }
            case kExternal: {
                sign = external_sign;
                break;
            }
            case kHybrid:
            case kNone:
                break;
        }

        // compute sdf gradient
        if (compute_gradient && !sdf_gradient_computed) {
            if (!edf_result.template GetGradientD<Dim>(0, 0, sdf_gradient.data())) {
                var[0] = 1e6f;
                return false;
            }
        }

        // compute sdf variance (always)
        // compute sdf gradient variance if compute_gradient_variance is true
        // compute covariance if compute_covariance is true
        Dtype &var_sdf = var[0];
        if (use_gp_covariance) {
            edf_result.GetMeanVariance(0, var_sdf);
            if (compute_gradient_variance) { edf_result.GetGradientVariance(0, var.data() + 1); }
            if (compute_covariance) { edf_result.GetCovariance(0, covariance.data()); }
        } else {
            EstimateVariance(
                test_position,
                sdf,
                compute_gradient_variance,
                compute_covariance,
                var.data(),
                covariance.data());
        }

        sdf -= offset_distance;
        if (std::signbit(sdf) != std::signbit(sign)) {
            sdf = std::copysign(sdf, sign);
            if (compute_gradient) {  // flip the gradient if the sign is different
                for (long i = 1; i <= Dim; ++i) { f[i] = -f[i]; }
            }
        }
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::operator==(const SdfGaussianProcess &other) const {
        if (setting == nullptr && other.setting != nullptr) { return false; }
        if (setting != nullptr && (other.setting == nullptr || *setting != *other.setting)) {
            return false;
        }
        if (active != other.active) { return false; }
        if (outdated != other.outdated) { return false; }
        if (use_normal_gp != other.use_normal_gp) { return false; }
        if (offset_distance != other.offset_distance) { return false; }
        if (locked_for_test.load() != other.locked_for_test.load()) { return false; }
        if (position != other.position) { return false; }
        if (half_size != other.half_size) { return false; }
        if (sign_gp == nullptr && other.sign_gp != nullptr) { return false; }
        if (sign_gp != nullptr && (other.sign_gp == nullptr || *sign_gp != *other.sign_gp)) {
            return false;
        }
        if (edf_gp == nullptr && other.edf_gp != nullptr) { return false; }
        if (edf_gp != nullptr && (other.edf_gp == nullptr || *edf_gp != *other.edf_gp)) {
            return false;
        }
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::operator!=(const SdfGaussianProcess &other) const {
        return !(*this == other);
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::Write(std::ostream &s) const {
        // no need to write the setting, as it will be written externally.
        using namespace common;
        static const TokenWriteFunctionPairs<SdfGaussianProcess> token_function_pairs = {
            {
                "active",
                [](const SdfGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << gp->active;
                    return stream.good();
                },
            },
            {
                "outdated",
                [](const SdfGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << gp->outdated;
                    return stream.good();
                },
            },
            {
                "use_normal_gp",
                [](const SdfGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << gp->use_normal_gp;
                    return stream.good();
                },
            },
            {
                "offset_distance",
                [](const SdfGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream.write(
                        reinterpret_cast<const char *>(&gp->offset_distance),
                        sizeof(gp->offset_distance));
                    return stream.good();
                },
            },
            {
                "locked_for_test",
                [](const SdfGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << gp->locked_for_test.load();
                    return stream.good();
                },
            },
            {
                "position",
                [](const SdfGaussianProcess *gp, std::ostream &stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, gp->position) && stream.good();
                },
            },
            {
                "half_size",
                [](const SdfGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream.write(
                        reinterpret_cast<const char *>(&gp->half_size),
                        sizeof(gp->half_size));
                    return stream.good();
                },
            },
            {
                "sign_gp",
                [](const SdfGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << (gp->sign_gp != nullptr) << '\n';
                    if (gp->sign_gp != nullptr && !gp->sign_gp->Write(stream)) { return false; }
                    return stream.good();
                },
            },
            {
                "edf_gp",
                [](const SdfGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << (gp->edf_gp != nullptr) << '\n';
                    if (gp->edf_gp != nullptr && !gp->edf_gp->Write(stream)) { return false; }
                    return stream.good();
                },
            },
        };
        return WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::Read(std::istream &s) {
        using namespace common;
        static const TokenReadFunctionPairs<SdfGaussianProcess> token_function_pairs = {
            {
                "active",
                [](SdfGaussianProcess *gp, std::istream &stream) -> bool {
                    stream >> gp->active;
                    return stream.good();
                },
            },
            {
                "outdated",
                [](SdfGaussianProcess *gp, std::istream &stream) -> bool {
                    stream >> gp->outdated;
                    return stream.good();
                },
            },
            {
                "use_normal_gp",
                [](SdfGaussianProcess *gp, std::istream &stream) -> bool {
                    stream >> gp->use_normal_gp;
                    return stream.good();
                },
            },
            {
                "offset_distance",
                [](SdfGaussianProcess *gp, std::istream &stream) -> bool {
                    stream.read(
                        reinterpret_cast<char *>(&gp->offset_distance),
                        sizeof(gp->offset_distance));
                    return stream.good();
                },
            },
            {
                "locked_for_test",
                [](SdfGaussianProcess *gp, std::istream &stream) -> bool {
                    bool locked;
                    stream >> locked;
                    gp->locked_for_test.store(locked);
                    return stream.good();
                },
            },
            {
                "position",
                [](SdfGaussianProcess *gp, std::istream &stream) -> bool {
                    return common::LoadEigenMatrixFromBinaryStream(stream, gp->position) &&
                           stream.good();
                },
            },
            {
                "half_size",
                [](SdfGaussianProcess *gp, std::istream &stream) -> bool {
                    stream.read(reinterpret_cast<char *>(&gp->half_size), sizeof(gp->half_size));
                    return stream.good();
                },
            },
            {
                "sign_gp",
                [](SdfGaussianProcess *gp, std::istream &stream) -> bool {
                    bool has_gp;
                    stream >> has_gp;
                    SkipLine(stream);
                    if (!has_gp) {  // no sign GP, skip
                        gp->sign_gp = nullptr;
                        return stream.good();
                    }
                    if (gp->sign_gp == nullptr) {
                        gp->sign_gp = std::make_shared<SignGp>(gp->setting->sign_gp);
                    }
                    return gp->sign_gp->Read(stream) && stream.good();
                },
            },
            {
                "edf_gp",
                [](SdfGaussianProcess *gp, std::istream &stream) -> bool {
                    bool has_gp;
                    stream >> has_gp;
                    SkipLine(stream);
                    if (!has_gp) {  // no EDF GP, skip
                        gp->edf_gp = nullptr;
                        return stream.good();
                    }
                    if (gp->edf_gp == nullptr) {
                        gp->edf_gp = std::make_shared<EdfGp>(gp->setting->edf_gp);
                    }
                    return gp->edf_gp->Read(stream) && stream.good();
                },
            },
        };
        return ReadTokens(s, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    void
    SdfGaussianProcess<Dtype, Dim>::EstimateVariance(
        const VectorD &test_position,
        const Dtype edf_pred,
        const bool compute_gradient_variance,
        const bool compute_covariance,
        Dtype *var,
        Dtype *covariance) const {

        const typename LogEdfGaussianProcess<Dtype>::TrainSet &train_set = edf_gp->GetTrainSet();
        const long num_samples = train_set.num_samples;
        const Dtype softmin_temperature = setting->softmin_temperature;
        const bool compute_cov_grad = compute_gradient_variance || compute_covariance;

        VectorX s(num_samples);
        Dtype s_sum = 0;
        VectorX z(num_samples);
        Eigen::Matrix<Dtype, Dim, Eigen::Dynamic> mat_v(Dim, num_samples);
        for (long k = 0; k < num_samples; ++k) {
            const VectorD v = test_position - train_set.x.col(k);
            Dtype &d = z[k];
            d = v.norm();  // distance to the training sample

            s[k] = std::max(
                static_cast<Dtype>(1.0e-6),
                std::exp(-(d - edf_pred) * softmin_temperature));
            s_sum += s[k];

            mat_v.col(k) = v / d;
        }
        const Dtype inv_s_sum = 1.0f / s_sum;
        const Dtype sz = s.dot(z) * inv_s_sum;
        var[0] = 0.0f;  // var_sdf
        VectorX l(num_samples);
        VectorD g = VectorD::Zero();  // sum_i (l_i * v_i)
        VectorD f = VectorD::Zero();  // sum_i (s_i * v_i)
        for (long k = 0; k < num_samples; ++k) {
            Dtype &w = l[k];
            w = inv_s_sum * s[k] * (1.0f + softmin_temperature * (sz - z[k]));
            var[0] += w * w * train_set.var_x[k];
            if (!compute_cov_grad) { continue; }
            g += w * mat_v.col(k);
            f += s[k] * mat_v.col(k);
        }

        using SqMat = Eigen::Matrix<Dtype, Dim, Dim>;
        SqMat cov_grad = SqMat::Zero();
        if (compute_cov_grad) {
            const SqMat identity = SqMat::Identity();
            const double g_norm = g.norm();
            const VectorD g_normalized = g / g_norm;
            const SqMat grad_norm =
                (1.0f / g_norm) * (identity - g_normalized * g_normalized.transpose());
            for (long j = 0; j < num_samples; ++j) {
                const Dtype a = softmin_temperature * l[j];
                const Dtype b = softmin_temperature * s[j];
                const Dtype c = l[j] / z[j];
                const auto vj = mat_v.col(j);
                const VectorD v = (a + b + c) * vj - a * f - b * g;
                SqMat grad_j = vj * v.transpose();
                grad_j.diagonal().array() -= c;
                grad_j = grad_j * grad_norm;
                cov_grad += train_set.var_x[j] * (grad_j.transpose() * grad_j);
            }
        }

        if (compute_gradient_variance) {
            for (long i = 1; i <= Dim; ++i) { var[i] = cov_grad(i - 1, i - 1); }  // var_grad
        }

        if (compute_covariance) {
            // 2D: 0, 0, cov_grad(1, 0)
            // 3D: 0, 0, 0, cov_grad(1, 0), cov_grad(2, 0), cov_grad(2, 1)
            for (long i = 0; i < Dim; ++i) { covariance[i] = 0; }
            covariance[Dim] = cov_grad(1, 0);
            if (Dim == 3) {
                covariance[Dim + 1] = cov_grad(2, 0);
                covariance[Dim + 2] = cov_grad(2, 1);
            }
        }
    }

}  // namespace erl::sdf_mapping
