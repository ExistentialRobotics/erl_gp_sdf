#pragma once

template<>
struct YAML::convert<erl::sdf_mapping::SignPredictionMethod> {
    static Node
    encode(const erl::sdf_mapping::SignPredictionMethod &method) {
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
            default:
                ERL_FATAL("Unknown SignPredictionMethod: {}", static_cast<int>(method));
        }
        return node;
    }

    static bool
    decode(const Node &node, erl::sdf_mapping::SignPredictionMethod &method) {
        const std::string method_str = node.as<std::string>();
        if (method_str == "kNone") {
            method = erl::sdf_mapping::kNone;
        } else if (method_str == "kSignGp") {
            method = erl::sdf_mapping::kSignGp;
        } else if (method_str == "kNormalGp") {
            method = erl::sdf_mapping::kNormalGp;
        } else {
            ERL_FATAL("Unknown SignPredictionMethod: {}", method_str);
        }
        return true;
    }
};

namespace erl::sdf_mapping {

    template<typename Dtype>
    YAML::Node
    SdfGaussianProcessSetting<Dtype>::YamlConvertImpl::encode(const SdfGaussianProcessSetting &setting) {
        YAML::Node node;
        node["sign_prediction_method"] = setting.sign_prediction_method;
        node["sign_gp"] = setting.sign_gp;
        node["edf_gp"] = setting.edf_gp;
        return node;
    }

    template<typename Dtype>
    bool
    SdfGaussianProcessSetting<Dtype>::YamlConvertImpl::decode(const YAML::Node &node, SdfGaussianProcessSetting &setting) {
        if (!node.IsMap()) { return false; }
        setting.sign_prediction_method = node["sign_prediction_method"].as<SignPredictionMethod>();
        setting.sign_gp = node["sign_gp"].as<decltype(setting.sign_gp)>();
        setting.edf_gp = node["edf_gp"].as<decltype(setting.edf_gp)>();
        return true;
    }

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim>::SdfGaussianProcess(std::shared_ptr<Setting> setting_)
        : setting(std::move(setting_)) {
        ERL_ASSERTM(setting != nullptr, "Setting is null.");
    }

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim>::SdfGaussianProcess(const SdfGaussianProcess &other)
        : setting(other.setting),
          active(other.active),
          position(other.position),
          half_size(other.half_size) {
        if (other.edf_gp != nullptr) { edf_gp = std::make_shared<LogEdfGaussianProcess>(*other.edf_gp); }
    }

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim>::SdfGaussianProcess(SdfGaussianProcess &&other) noexcept
        : setting(other.setting),
          active(other.active),
          position(std::move(other.position)),
          half_size(other.half_size),
          edf_gp(std::move(other.edf_gp)) {}

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim> &
    SdfGaussianProcess<Dtype, Dim>::operator=(const SdfGaussianProcess &other) {
        if (this == &other) { return *this; }
        setting = other.setting;
        active = other.active;
        position = other.position;
        half_size = other.half_size;
        if (other.edf_gp != nullptr) { edf_gp = std::make_shared<EdfGp>(*other.edf_gp); }
        return *this;
    }

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim> &
    SdfGaussianProcess<Dtype, Dim>::operator=(SdfGaussianProcess &&other) noexcept {
        if (this == &other) { return *this; }
        setting = other.setting;
        active = other.active;
        position = other.position;
        half_size = other.half_size;
        edf_gp = std::move(other.edf_gp);
        return *this;
    }

    template<typename Dtype, int Dim>
    void
    SdfGaussianProcess<Dtype, Dim>::Activate() {
        if (sign_gp == nullptr && setting->sign_prediction_method == kSignGp) { sign_gp = std::make_shared<SignGp>(setting->sign_gp); }
        if (edf_gp == nullptr) { edf_gp = std::make_shared<EdfGp>(setting->edf_gp); }
        active = true;
    }

    template<typename Dtype, int Dim>
    void
    SdfGaussianProcess<Dtype, Dim>::Deactivate() {
        active = false;
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
    SdfGaussianProcess<Dtype, Dim>::Intersects(const VectorD &other_position, const Dtype other_half_size) const {
        for (int i = 0; i < Dim; ++i) {
            if (std::abs(position[i] - other_position[i]) > half_size + other_half_size) { return false; }
        }
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::Intersects(const VectorD &other_position, const VectorD &other_half_sizes) const {
        for (int i = 0; i < Dim; ++i) {
            if (std::abs(position[i] - other_position[i]) > half_size + other_half_sizes[i]) { return false; }
        }
        return true;
    }

    template<typename Dtype, int Dim>
    void
    SdfGaussianProcess<Dtype, Dim>::LoadSurfaceData(
        std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices,
        const std::vector<SurfaceData<Dtype, Dim>> &surface_data_vec,
        Dtype offset_distance,
        Dtype sensor_noise,
        Dtype max_valid_gradient_var,
        Dtype invalid_position_var) {
        if (sign_gp != nullptr) {
            sign_gp->template LoadSurfaceData<Dim>(
                surface_data_indices,
                surface_data_vec,
                position,
                offset_distance,
                sensor_noise,
                max_valid_gradient_var,
                invalid_position_var);
        }
        if (edf_gp != nullptr) {
            edf_gp->template LoadSurfaceData<Dim>(
                surface_data_indices,
                surface_data_vec,
                position,
                setting->sign_prediction_method == kNormalGp,
                setting->normal_scale,
                offset_distance,
                sensor_noise,
                max_valid_gradient_var,
                invalid_position_var);
        }
    }

    template<typename Dtype, int Dim>
    void
    SdfGaussianProcess<Dtype, Dim>::Train() const {
        ERL_DEBUG_ASSERT(active, "SdfGaussianProcess is not active.");
        if (sign_gp != nullptr) { sign_gp->Train(); }
        if (edf_gp != nullptr) { edf_gp->Train(); }
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::Test(
        const VectorD &test_position,  // single position to test
        Eigen::Ref<Eigen::Vector<Dtype, 2 * Dim + 1>> f,
        Eigen::Ref<Eigen::Vector<Dtype, Dim + 1>> var,
        Eigen::Ref<Eigen::Vector<Dtype, Dim *(Dim + 1) / 2>> covariance,
        const Dtype offset_distance,
        const Dtype softmin_temperature,
        const bool use_gp_covariance,
        const bool compute_covariance) const {

        ERL_DEBUG_ASSERT(active, "SdfGaussianProcess is not active.");

        // TODO: sign prediction

        VectorX no_variance, no_covariance;
        Eigen::Scalar<Dtype> sign;
        std::vector<std::pair<long, bool>> y_index_grad_pairs = {{0, true}};

        bool predict_sign = false;

        auto predict_sign_func = [&]() -> bool {
            if (setting->sign_prediction_method == kSignGp) {
                ERL_DEBUG_ASSERT(sign_gp != nullptr, "Sign GP is not initialized.");
                const bool success = sign_gp->Test(test_position, {{0, false}}, sign, no_variance, no_covariance);
                ERL_DEBUG_ASSERT(success, "Sign GP failed to predict sign.");
                return success;
            }
            if (setting->sign_prediction_method == kNormalGp) {
                sign[0] = f.template segment<Dim>(1).dot(f.template tail<Dim>());
                return true;  // use normal GP for sign prediction
            }
            return false;
        };

        if (setting->sign_prediction_method == kNormalGp) {
            y_index_grad_pairs.reserve(1 + Dim);
            for (long i = 0; i < Dim; ++i) { y_index_grad_pairs.emplace_back(i + 1, false); }
        }

        if (use_gp_covariance) {
            if (compute_covariance) {
                edf_gp->Test(test_position, y_index_grad_pairs, f, var, covariance);
            } else {
                edf_gp->Test(test_position, y_index_grad_pairs, f, var, no_covariance);
            }
            if (f.template segment<Dim>(1).norm() < 1.e-6) { return false; }  // invalid gradient, skip this GP

            predict_sign = predict_sign_func();
        } else {
            edf_gp->Test(test_position, y_index_grad_pairs, f, no_variance, no_covariance);
            if (f.template segment<Dim>(1).norm() < 1.e-6) { return false; }  // invalid gradient, skip this GP

            predict_sign = predict_sign_func();

            const typename LogEdfGaussianProcess<Dtype>::TrainSet &train_set = edf_gp->GetTrainSet();
            VectorX s(static_cast<int>(train_set.num_samples));
            Dtype s_sum = 0;
            VectorX z_sdf(static_cast<int>(train_set.num_samples));
            Eigen::Matrix<Dtype, Dim, Eigen::Dynamic> diff_z_sdf(Dim, train_set.num_samples);
            using SquareMatrix = Eigen::Matrix<Dtype, Dim, Dim>;
            const SquareMatrix identity = SquareMatrix::Identity();
            for (long k = 0; k < train_set.num_samples; ++k) {
                const VectorD v = test_position - train_set.x.col(k);
                const Dtype d = v.norm();  // distance to the training sample

                z_sdf[k] = d;
                s[k] = std::max(static_cast<Dtype>(1.e-6), std::exp(-(d - f[0]) * softmin_temperature));
                s_sum += s[k];

                diff_z_sdf.col(k) = v / d;
            }
            const Dtype inv_s_sum = 1.0f / s_sum;
            const Dtype sz_sdf = s.dot(z_sdf) * inv_s_sum;
            var[0] = 0.0f;  // var_sdf
            SquareMatrix cov_grad = SquareMatrix::Zero();
            for (long k = 0; k < train_set.num_samples; ++k) {
                Dtype w = s[k] * (sz_sdf + 1.0f - z_sdf[k]) * inv_s_sum;
                w = w * w * train_set.var_x[k];
                var[0] += w;
                w = w / (z_sdf[k] * z_sdf[k]);
                const SquareMatrix diff_grad = (identity - diff_z_sdf.col(k) * diff_z_sdf.col(k).transpose()) * w;
                cov_grad += diff_grad;
            }
            var.template segment<Dim>(1) << cov_grad.diagonal();  // var_grad
            if (compute_covariance) {
                if (Dim == 2) {
                    covariance << 0, 0, cov_grad(1, 0);
                } else {
                    covariance << 0, 0, 0, cov_grad(1, 0), cov_grad(2, 0), cov_grad(2, 1);
                }
            }
        }

        f[0] -= offset_distance;
        if (predict_sign && std::signbit(f[0]) != std::signbit(sign[0])) {
            f[0] = std::copysign(f[0], sign[0]);
            for (long i = 1; i <= Dim; ++i) { f[i] = -f[i]; }  // flip the gradient if the sign is different
        }
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::operator==(const SdfGaussianProcess &other) const {
        if (active != other.active) { return false; }
        if (locked_for_test.load() != other.locked_for_test.load()) { return false; }
        if (position != other.position) { return false; }
        if (half_size != other.half_size) { return false; }
        if (edf_gp == nullptr && other.edf_gp != nullptr) { return false; }
        if (edf_gp != nullptr && (other.edf_gp == nullptr || *edf_gp != *other.edf_gp)) { return false; }
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
        s << "# " << type_name(*this) << "\n# (feel free to add / change comments, but leave the first line as it is!)\n";
        // no need to write the setting, as it will be written externally.
        static const std::vector<std::pair<const char *, std::function<bool(const SdfGaussianProcess *, std::ostream &)>>> token_function_pairs = {
            {
                "active",
                [](const SdfGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << gp->active;
                    return true;
                },
            },
            {
                "locked_for_test",
                [](const SdfGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << gp->locked_for_test.load();
                    return true;
                },
            },
            {
                "position",
                [](const SdfGaussianProcess *gp, std::ostream &stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, gp->position)) {
                        ERL_WARN("Failed to write position.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "half_size",
                [](const SdfGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream.write(reinterpret_cast<const char *>(&gp->half_size), sizeof(gp->half_size));
                    return true;
                },
            },
            {
                "sign_gp",
                [](const SdfGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << (gp->sign_gp != nullptr) << '\n';
                    if (gp->sign_gp != nullptr && !gp->sign_gp->Write(stream)) {
                        ERL_WARN("Failed to write sign GP.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "edf_gp",
                [](const SdfGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << (gp->edf_gp != nullptr) << '\n';
                    if (gp->edf_gp != nullptr && !gp->edf_gp->Write(stream)) {
                        ERL_WARN("Failed to write EDF GP.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "end_of_SdfGaussianProcess",
                [](const SdfGaussianProcess *, std::ostream &) -> bool { return true; },
            },
        };
        return common::WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::Read(std::istream &s, const std::shared_ptr<typename EdfGp::Setting> &edf_gp_setting) {
        if (!s.good()) {
            ERL_WARN("Input stream is not ready for reading");
            return false;
        }

        // check if the first line is valid
        std::string line;
        std::getline(s, line);
        if (std::string file_header = fmt::format("# {}", type_name(*this));
            line.compare(0, file_header.length(), file_header) != 0) {  // check if the first line is valid
            ERL_WARN("Header does not start with \"{}\"", file_header);
            return false;
        }

        static const std::vector<std::pair<const char *, std::function<bool(SdfGaussianProcess *, std::istream &)>>> token_function_pairs = {
            {
                "active",
                [](SdfGaussianProcess *gp, std::istream &stream) -> bool {
                    stream >> gp->active;
                    return true;
                },
            },
            {
                "locked_for_test",
                [](SdfGaussianProcess *gp, std::istream &stream) -> bool {
                    bool locked;
                    stream >> locked;
                    gp->locked_for_test.store(locked);
                    return true;
                },
            },
            {
                "position",
                [](SdfGaussianProcess *gp, std::istream &stream) -> bool {
                    common::SkipLine(stream);
                    if (!common::LoadEigenMatrixFromBinaryStream(stream, gp->position)) {
                        ERL_WARN("Failed to read position.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "half_size",
                [](SdfGaussianProcess *gp, std::istream &stream) -> bool {
                    common::SkipLine(stream);
                    stream.read(reinterpret_cast<char *>(&gp->half_size), sizeof(gp->half_size));
                    return true;
                },
            },
            {
                "edf_gp",
                [](SdfGaussianProcess *gp, std::istream &stream) -> bool {
                    bool has_gp;
                    stream >> has_gp;
                    if (!has_gp) {  // no EDF GP, skip
                        gp->edf_gp = nullptr;
                        return true;
                    }
                    common::SkipLine(stream);
                    if (gp->edf_gp == nullptr) { gp->edf_gp = std::make_shared<EdfGp>(gp->setting->edf_gp); }
                    if (!gp->edf_gp->Read(stream)) {
                        ERL_WARN("Failed to read EDF GP.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "end_of_SdfGaussianProcess",
                [](SdfGaussianProcess *, std::istream &stream) -> bool {
                    common::SkipLine(stream);
                    return true;
                },
            },
        };
        return common::ReadTokens(s, this, token_function_pairs);
    }
}  // namespace erl::sdf_mapping
