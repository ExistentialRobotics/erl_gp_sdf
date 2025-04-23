#pragma once

namespace erl::sdf_mapping {

    template<typename Dtype>
    YAML::Node
    SdfGaussianProcessSetting<Dtype>::YamlConvertImpl::encode(const SdfGaussianProcessSetting &setting) {
        YAML::Node node;
        node["enable_sign_gp"] = setting.enable_sign_gp;
        node["sign_gp"] = setting.sign_gp;
        node["edf_gp"] = setting.edf_gp;
        return node;
    }

    template<typename Dtype>
    bool
    SdfGaussianProcessSetting<Dtype>::YamlConvertImpl::decode(const YAML::Node &node, SdfGaussianProcessSetting &setting) {
        if (!node.IsMap()) { return false; }
        setting.enable_sign_gp = node["enable_sign_gp"].as<bool>();
        setting.sign_gp = node["sign_gp"].as<decltype(setting.sign_gp)>();
        setting.edf_gp = node["edf_gp"].as<decltype(setting.edf_gp)>();
        return true;
    }

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim>::SdfGaussianProcess(std::shared_ptr<Setting> setting)
        : setting(std::move(setting)) {}

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim>::SdfGaussianProcess(const SdfGaussianProcess &other)
        : setting(other.setting),
          active(other.active),
          num_edf_samples(other.num_edf_samples),
          position(other.position),
          half_size(other.half_size) {
        if (other.edf_gp != nullptr) { edf_gp = std::make_shared<LogEdfGaussianProcess>(*other.edf_gp); }
    }

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim>::SdfGaussianProcess(SdfGaussianProcess &&other) noexcept
        : setting(other.setting),
          active(other.active),
          num_edf_samples(other.num_edf_samples),
          position(std::move(other.position)),
          half_size(other.half_size),
          edf_gp(std::move(other.edf_gp)) {}

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim> &
    SdfGaussianProcess<Dtype, Dim>::operator=(const SdfGaussianProcess &other) {
        if (this == &other) { return *this; }
        setting = other.setting;
        active = other.active;
        num_edf_samples = other.num_edf_samples;
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
        num_edf_samples = other.num_edf_samples;
        position = other.position;
        half_size = other.half_size;
        edf_gp = std::move(other.edf_gp);
        return *this;
    }

    template<typename Dtype, int Dim>
    void
    SdfGaussianProcess<Dtype, Dim>::Activate() {
        if (sign_gp == nullptr && setting->enable_sign_gp) { sign_gp = std::make_shared<SignGp>(setting->sign_gp); }
        if (edf_gp == nullptr) { edf_gp = std::make_shared<EdfGp>(setting->edf_gp); }
        active = true;
    }

    template<typename Dtype, int Dim>
    void
    SdfGaussianProcess<Dtype, Dim>::Deactivate() {
        active = false;
        num_edf_samples = 0;
    }

    template<typename Dtype, int Dim>
    std::size_t
    SdfGaussianProcess<Dtype, Dim>::GetMemoryUsage() const {
        std::size_t memory_usage = sizeof(SdfGaussianProcess);
        if (edf_gp != nullptr) { memory_usage += edf_gp->GetMemoryUsage(); }
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
            num_sign_samples = sign_gp->template LoadSurfaceData<Dim>(
                surface_data_indices,
                surface_data_vec,
                position,
                offset_distance,
                sensor_noise,
                max_valid_gradient_var,
                invalid_position_var);
        }
        if (edf_gp != nullptr) {
            num_edf_samples = edf_gp->template LoadSurfaceData<Dim>(
                surface_data_indices,
                surface_data_vec,
                position,
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
        if (sign_gp != nullptr) { sign_gp->Train(num_sign_samples); }
        if (edf_gp != nullptr) { edf_gp->Train(num_edf_samples); }
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::Test(
        const VectorD &test_position,
        Eigen::Ref<Eigen::Vector<Dtype, Dim + 1>> f,
        Eigen::Ref<Eigen::Vector<Dtype, Dim + 1>> var,
        Eigen::Ref<Eigen::Vector<Dtype, Dim *(Dim + 1) / 2>> covariance,
        const Dtype offset_distance,
        const Dtype softmin_temperature,
        const bool use_gp_covariance,
        const bool compute_covariance) const {

        ERL_DEBUG_ASSERT(active, "SdfGaussianProcess is not active.");

        VectorX no_variance, no_covariance;
        VectorX sign(f.size());
        const bool predict_sign = sign_gp != nullptr && sign_gp->Test(test_position, sign, no_variance, no_covariance, false);

        if (use_gp_covariance) {
            if (compute_covariance) {
                edf_gp->Test(test_position, f, var, covariance, true);
            } else {
                edf_gp->Test(test_position, f, var, no_covariance, true);
            }
            if (f.template tail<Dim>().norm() < 1.e-15) { return false; }  // invalid gradient, skip this GP
        } else {
            edf_gp->Test(test_position, f, no_variance, no_covariance, true);
            VectorD grad = f.template tail<Dim>();
            if (grad.norm() <= 1.e-15) { return false; }  // invalid gradient, skip this GP

            auto &mat_x = edf_gp->GetTrainInputSamplesBuffer();
            auto &vec_x_var = edf_gp->GetTrainInputSamplesVarianceBuffer();
            const long num_samples = edf_gp->GetNumTrainSamples();

            VectorX s(static_cast<int>(num_samples));
            Dtype s_sum = 0;
            VectorX z_sdf(static_cast<int>(num_samples));
            Eigen::Matrix<Dtype, Dim, Eigen::Dynamic> diff_z_sdf(Dim, num_samples);
            using SquareMatrix = Eigen::Matrix<Dtype, Dim, Dim>;
            const SquareMatrix identity = SquareMatrix::Identity();
            for (long k = 0; k < num_samples; ++k) {
                const VectorD v = test_position - mat_x.col(k);
                const Dtype d = v.norm();  // distance to the training sample

                z_sdf[k] = d;
                s[k] = std::max(static_cast<Dtype>(1.e-6), std::exp(-(d - f[0]) * softmin_temperature));
                s_sum += s[k];

                diff_z_sdf.col(k) = v / d;
            }

            // s /= s_sum;  // this line causes an extra for loop. replace it with the following line
            const Dtype inv_s_sum = 1.0f / s_sum;

            const Dtype sz_sdf = s.dot(z_sdf) * inv_s_sum;
            var[0] = 0.0f;  // var_sdf
            SquareMatrix cov_grad = SquareMatrix::Zero();
            for (long k = 0; k < num_samples; ++k) {
                Dtype w = s[k] * (sz_sdf + 1.0f - z_sdf[k]) * inv_s_sum;
                w = w * w * vec_x_var[k];
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
        if (predict_sign) { f[0] = std::copysign(f[0], sign[0]); }
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::operator==(const SdfGaussianProcess &other) const {
        if (active != other.active) { return false; }
        if (locked_for_test.load() != other.locked_for_test.load()) { return false; }
        if (num_edf_samples != other.num_edf_samples) { return false; }
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
        s << kFileHeader << std::endl  //
          << "# (feel free to add / change comments, but leave the first line as it is!)" << std::endl;
        s << "active " << active << std::endl
          << "locked_for_test " << locked_for_test.load() << std::endl
          << "num_edf_samples " << num_edf_samples << std::endl;
        s << "position" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, position)) { return false; }
        s << "half_size" << std::endl;
        s.write(reinterpret_cast<const char *>(&half_size), sizeof(half_size));
        s << "edf_gp " << (edf_gp != nullptr) << std::endl;
        if (edf_gp != nullptr && !edf_gp->Write(s)) { return false; }
        s << "end_of_SdfGaussianProcess" << std::endl;
        return s.good();
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
        if (line.compare(0, kFileHeader.length(), kFileHeader) != 0) {  // check if the first line is valid
            ERL_WARN("Header does not start with \"{}\"", kFileHeader);
            return false;
        }

        auto skip_line = [&s]() {
            char c;
            do { c = static_cast<char>(s.get()); } while (s.good() && c != '\n');
        };

        static const char *tokens[] = {
            "active",
            "locked_for_test",
            "num_edf_samples",
            "position",
            "half_size",
            "edf_gp",
            "end_of_SdfGaussianProcess",
        };

        // read data
        std::string token;
        int token_idx = 0;
        while (s.good()) {
            s >> token;
            if (token.compare(0, 1, "#") == 0) {
                skip_line();  // comment line, skip forward until end of line
                continue;
            }
            // non-comment line
            if (token != tokens[token_idx]) {
                ERL_WARN("Expected token {}, got {}.", tokens[token_idx], token);  // check token
                return false;
            }
            // reading state machine
            switch (token_idx) {
                case 0: {  // active
                    s >> active;
                    break;
                }
                case 1: {  // locked_for_test
                    bool locked;
                    s >> locked;
                    locked_for_test.store(locked);
                    break;
                }
                case 2: {  // num_edf_samples
                    s >> num_edf_samples;
                    break;
                }
                case 3: {  // position
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, position)) {
                        ERL_WARN("Failed to read position.");
                        return false;
                    }
                    break;
                }
                case 4: {  // half_size
                    skip_line();
                    s.read(reinterpret_cast<char *>(&half_size), sizeof(half_size));
                    break;
                }
                case 5: {  // edf_gp
                    bool has_gp;
                    s >> has_gp;
                    if (has_gp) {
                        skip_line();
                        if (edf_gp == nullptr) { edf_gp = std::make_shared<LogEdfGaussianProcess<Dtype>>(edf_gp_setting); }
                        if (!edf_gp->Read(s)) { return false; }
                    }
                    break;
                }
                case 6: {  // end_of_SdfGaussianProcess
                    skip_line();
                    return true;
                }
                default: {  // should not reach here
                    ERL_FATAL("Internal error, should not reach here.");
                }
            }
            ++token_idx;
        }
        ERL_WARN("Failed to read SdfGaussianProcess. Truncated file?");
        return false;  // should not reach here
    }
}  // namespace erl::sdf_mapping
