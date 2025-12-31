#include "erl_gp_sdf/sdf_gp.hpp"

namespace erl::gp_sdf {

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim>::SdfGaussianProcess(std::shared_ptr<Setting> setting_)
        : setting(std::move(setting_)) {
        ERL_ASSERTM(setting != nullptr, "Setting is null.");
        use_normal_gp = setting->sign_method == SignMethod::kNormalGp ||
                        (setting->sign_method == SignMethod::kHybrid &&
                         (setting->hybrid_sign_methods.first == SignMethod::kNormalGp ||
                          setting->hybrid_sign_methods.second == SignMethod::kNormalGp));
        std::array<Dtype, Dim> mean_pos{};
        mean_pos.fill(0);
        running_mean_position.store(mean_pos);
    }

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim>::SdfGaussianProcess(const SdfGaussianProcess &other)
        : setting(other.setting),
          active(other.active),
          time_stamp(other.time_stamp),
          buf_outdated_count(other.buf_outdated_count),
          gp_outdated_count(other.gp_outdated_count.load()),
          query_count(other.query_count.load()),
          use_normal_gp(other.use_normal_gp),
          position(other.position),
          running_mean_position(other.running_mean_position.load()),
          running_num_samples(other.running_num_samples),
          half_size(other.half_size) {
        if (other.sign_gp != nullptr) { sign_gp = std::make_shared<SignGp>(*other.sign_gp); }
        if (other.edf_gp != nullptr) { edf_gp = std::make_shared<EdfGp>(*other.edf_gp); }
    }

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim>::SdfGaussianProcess(SdfGaussianProcess &&other) noexcept
        : setting(std::move(other.setting)),
          active(other.active),
          time_stamp(other.time_stamp),
          buf_outdated_count(other.buf_outdated_count),
          gp_outdated_count(other.gp_outdated_count.load()),
          query_count(other.query_count.load()),
          use_normal_gp(other.use_normal_gp),
          position(std::move(other.position)),
          running_mean_position(other.running_mean_position.load()),
          running_num_samples(other.running_num_samples),
          half_size(other.half_size),
          sign_gp(std::move(other.sign_gp)),
          edf_gp(std::move(other.edf_gp)) {}

    template<typename Dtype, int Dim>
    SdfGaussianProcess<Dtype, Dim> &
    SdfGaussianProcess<Dtype, Dim>::operator=(const SdfGaussianProcess &other) {
        if (this == &other) { return *this; }
        setting = other.setting;
        active = other.active;
        time_stamp = other.time_stamp;
        buf_outdated_count = other.buf_outdated_count;
        gp_outdated_count.store(other.gp_outdated_count.load());
        query_count.store(other.query_count.load());
        use_normal_gp = other.use_normal_gp;
        position = other.position;
        running_mean_position.store(other.running_mean_position.load());
        running_num_samples = other.running_num_samples;
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
        time_stamp = other.time_stamp;
        buf_outdated_count = other.buf_outdated_count;
        gp_outdated_count.store(other.gp_outdated_count.load());
        query_count.store(other.query_count.load());
        use_normal_gp = other.use_normal_gp;
        running_mean_position.store(other.running_mean_position.load());
        running_num_samples = other.running_num_samples;
        position = other.position;
        half_size = other.half_size;
        sign_gp = std::move(other.sign_gp);
        edf_gp = std::move(other.edf_gp);
        return *this;
    }

    template<typename Dtype, int Dim>
    void
    SdfGaussianProcess<Dtype, Dim>::Activate() {
        if (sign_gp == nullptr &&
            (setting->sign_method == SignMethod::kSignGp ||
             (setting->sign_method == SignMethod::kHybrid &&
              (setting->hybrid_sign_methods.first == SignMethod::kSignGp ||
               setting->hybrid_sign_methods.second == SignMethod::kSignGp)))) {
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
    SdfGaussianProcess<Dtype, Dim>::MarkBufferOutdated() {
        ++buf_outdated_count;
    }

    template<typename Dtype, int Dim>
    void
    SdfGaussianProcess<Dtype, Dim>::MarkGpOutdated() {
        ++gp_outdated_count;
    }

    template<typename Dtype, int Dim>
    void
    SdfGaussianProcess<Dtype, Dim>::MarkQueried() {
        constexpr long max_query_count = 10000;
        if (query_count.load() < max_query_count) { ++query_count; }
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::BufferOutdated() const {
        return buf_outdated_count > 0;
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::GpOutdated() const {
        return gp_outdated_count.load() > 0;
    }

    template<typename Dtype, int Dim>
    Dtype
    SdfGaussianProcess<Dtype, Dim>::GetLoadingPriority(const Dtype query_count_weight) const {
        return static_cast<Dtype>(buf_outdated_count) *
               (1.0f + query_count_weight * static_cast<Dtype>(query_count.load()));
    }

    template<typename Dtype, int Dim>
    Dtype
    SdfGaussianProcess<Dtype, Dim>::GetRetrainPriority(Dtype query_count_weight) const {
        return static_cast<Dtype>(gp_outdated_count.load()) *
               (1.0f + query_count_weight * static_cast<Dtype>(query_count.load()));
    }

    template<typename Dtype, int Dim>
    void
    SdfGaussianProcess<Dtype, Dim>::SetMeanPosition(const VectorD &mean_position) {
        std::array<Dtype, Dim> pos{};
        for (int i = 0; i < Dim; ++i) { CHECKED_AT(pos, i) = mean_position[i]; }
        running_mean_position.store(pos);
    }

    template<typename Dtype, int Dim>
    typename SdfGaussianProcess<Dtype, Dim>::VectorD
    SdfGaussianProcess<Dtype, Dim>::GetMeanPosition() const {
        VectorD pos;
        std::array<Dtype, Dim> mean_pos = running_mean_position.load();
        for (int i = 0; i < Dim; ++i) { pos[i] = CHECKED_AT(mean_pos, i); }
        return pos;
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
    bool
    SdfGaussianProcess<Dtype, Dim>::LoadSurfaceData(
        std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices,
        const std::vector<SurfaceData<Dtype, Dim>> &surface_data_vec,
        const bool data_sorted,
        Dtype sensor_noise,
        Dtype max_valid_gradient_var,
        Dtype invalid_position_var) {
        bool loaded = false;
        if (sign_gp != nullptr) {
            loaded |= sign_gp->template LoadSurfaceData<Dim>(
                surface_data_indices,
                surface_data_vec,
                position,
                data_sorted,
                setting->normal_scale,
                setting->sign_gp_offset_distance,
                sensor_noise,
                max_valid_gradient_var,
                invalid_position_var);
        }
        if (edf_gp != nullptr && edf_gp->template LoadSurfaceData<Dim>(
                                     surface_data_indices,
                                     surface_data_vec,
                                     position,
                                     data_sorted,
                                     use_normal_gp,
                                     setting->normal_scale,
                                     setting->edf_gp_offset_distance,
                                     sensor_noise,
                                     max_valid_gradient_var,
                                     invalid_position_var)) {
            loaded = true;
            auto &buf = edf_gp->GetLoadingBuffer();
            std::array<Dtype, Dim> mean_pos = running_mean_position.load();
            for (int i = 0; i < Dim; ++i) {
                CHECKED_AT(mean_pos, i) *= static_cast<Dtype>(running_num_samples);
            }
            for (long i = 0; i < buf.num_samples; ++i) {
                auto p = buf.x.col(i);
                for (int d = 0; d < Dim; ++d) { CHECKED_AT(mean_pos, d) += p[d]; }
            }
            running_num_samples += buf.num_samples;
            for (int i = 0; i < Dim; ++i) {
                CHECKED_AT(mean_pos, i) /= static_cast<Dtype>(running_num_samples);
            }
            running_mean_position.store(mean_pos);
        }
        gp_outdated_count += buf_outdated_count;
        buf_outdated_count = 0;
        return loaded;
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
        bool trained = false;
        if (sign_gp != nullptr) { trained |= sign_gp->ReTrain(); }
        if (edf_gp != nullptr) { trained |= edf_gp->ReTrain(); }
        if (trained) {
            gp_outdated_count.store(0);
            query_count.store(query_count.load() / 2);  // decay query count
            time_stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        }
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::Test(
        const VectorD &test_position,                     // single position to test
        Eigen::Ref<Eigen::Vector<Dtype, 2 * Dim + 1>> f,  // sdf, sdf_gradient, normal
        Eigen::Ref<Eigen::Vector<Dtype, Dim + 1>> var,
        Eigen::Ref<Eigen::Vector<Dtype, Dim *(Dim + 1) / 2>> covariance,
        Dtype &sign,
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
        Dtype edf = 0.0f;
        Dtype &sdf = f[0];
        using Result = typename EdfGp::TestResult;
        auto result = std::reinterpret_pointer_cast<Result>(
            edf_gp->Test(test_position, compute_gradient || use_normal_gp));
        result->GetMean(0, 0, edf);
        if (!std::isfinite(edf)) {  // invalid sdf
            ERL_DEBUG("edf is not finite at position [{}].", test_position.transpose());
            var[0] = 1e6f;  // set a large variance if sdf is invalid
            return false;
        }
        sdf = edf - setting->edf_gp_offset_distance;

        // compute sign
        SignMethod sign_method = setting->sign_method;
        if (sign_method == SignMethod::kHybrid) {
            if (setting->hybrid_sign_threshold > edf) {
                sign_method = setting->hybrid_sign_methods.first;
            } else {
                sign_method = setting->hybrid_sign_methods.second;
            }
        }
        const Dtype external_sign = sign;
        sign = 1.0f;
        bool sdf_gradient_computed = false;
        auto sdf_gradient = f.template segment<Dim>(1);
        switch (sign_method) {
            case SignMethod::kSignGp: {
                ERL_DEBUG_ASSERT(sign_gp != nullptr, "sign_gp is not initialized.");
                (*std::reinterpret_pointer_cast<typename SignGp::TestResult>(
                     sign_gp->Test(test_position, false)))
                    .GetMean(0, 0, sign);
                break;
            }
            case SignMethod::kNormalGp: {
                auto normal = f.template tail<Dim>();
                if (!result->template GetGradientD<Dim>(0, 0, sdf_gradient.data())) {
                    ERL_DEBUG("Failed to predict gradient.");
                    var[0] = 1e6f;
                    return false;
                }
                for (long i = 1; i <= Dim; ++i) { result->GetMean(0, i, normal[i - 1]); }
                sign = sdf_gradient.dot(normal);
                sdf_gradient_computed = true;
                break;
            }
            case SignMethod::kExternal: {
                sign = external_sign;
                break;
            }
            case SignMethod::kNone: {
                sign = sdf < 0 ? -1.0f : 1.0f;  // default sign based on sdf value
                break;
            }
            case SignMethod::kHybrid:
                break;
        }
        if (std::signbit(sdf) != std::signbit(sign)) { sdf = std::copysign(sdf, sign); }

        // compute sdf gradient
        if (compute_gradient && !sdf_gradient_computed) {
            if (!result->template GetGradientD<Dim>(0, 0, sdf_gradient.data())) {
                ERL_DEBUG("Failed to predict gradient.");
                var[0] = 1e6f;
                return false;
            }
            if (sdf < 0.0f) {
                for (long i = 0; i < Dim; ++i) { sdf_gradient[i] = -sdf_gradient[i]; }
            }
        }

        // compute sdf variance (always)
        // compute sdf gradient variance if compute_gradient_variance is true
        // compute covariance if compute_covariance is true
        Dtype &var_sdf = var[0];
        if (use_gp_covariance) {
            result->GetMeanVariance(0, var_sdf);
            if (compute_gradient_variance) { result->GetGradientVariance(0, var.data() + 1); }
            if (compute_covariance) { result->GetCovariance(0, covariance.data()); }
        } else {
            EstimateVariance(
                test_position,
                edf,
                compute_gradient_variance,
                compute_covariance,
                var.data(),
                covariance.data());
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
        if (time_stamp != other.time_stamp) { return false; }
        if (buf_outdated_count != other.buf_outdated_count) { return false; }
        if (gp_outdated_count.load() != other.gp_outdated_count.load()) { return false; }
        if (query_count != other.query_count) { return false; }
        if (use_normal_gp != other.use_normal_gp) { return false; }
        if (position != other.position) { return false; }
        if (half_size != other.half_size) { return false; }
        if (running_num_samples != other.running_num_samples) { return false; }
        if (running_mean_position.load() != other.running_mean_position.load()) { return false; }
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
    SdfGaussianProcess<Dtype, Dim>::Write(std::ostream &stream) const {
        // no need to write the setting, as it will be written externally.
        using namespace common;
        using namespace common::serialization;
        static const TokenWriteFunctionPairs<SdfGaussianProcess> token_function_pairs = {
            {
                "active",
                [](const SdfGaussianProcess *gp, std::ostream &s) -> bool {
                    s << gp->active;
                    return s.good();
                },
            },
            {
                "time_stamp",
                [](const SdfGaussianProcess *gp, std::ostream &s) -> bool {
                    s.write(
                        reinterpret_cast<const char *>(&gp->time_stamp),
                        sizeof(gp->time_stamp));
                    return s.good();
                },
            },
            {
                "buf_outdated_count",
                [](const SdfGaussianProcess *gp, std::ostream &s) -> bool {
                    s.write(
                        reinterpret_cast<const char *>(&gp->buf_outdated_count),
                        sizeof(gp->buf_outdated_count));
                    return s.good();
                },
            },
            {
                "gp_outdated_count",
                [](const SdfGaussianProcess *gp, std::ostream &s) -> bool {
                    const long count = gp->gp_outdated_count.load();
                    s.write(reinterpret_cast<const char *>(&count), sizeof(count));
                    return s.good();
                },
            },
            {
                "query_count",
                [](const SdfGaussianProcess *gp, std::ostream &s) -> bool {
                    s.write(
                        reinterpret_cast<const char *>(&gp->query_count),
                        sizeof(gp->query_count));
                    return s.good();
                },
            },
            {
                "use_normal_gp",
                [](const SdfGaussianProcess *gp, std::ostream &s) -> bool {
                    s << gp->use_normal_gp;
                    return s.good();
                },
            },
            {
                "position",
                [](const SdfGaussianProcess *gp, std::ostream &s) -> bool {
                    return SaveEigenMatrixToBinaryStream(s, gp->position) && s.good();
                },
            },
            {
                "half_size",
                [](const SdfGaussianProcess *gp, std::ostream &s) -> bool {
                    s.write(reinterpret_cast<const char *>(&gp->half_size), sizeof(gp->half_size));
                    return s.good();
                },
            },
            {
                "running_mean_position",
                [](const SdfGaussianProcess *gp, std::ostream &s) -> bool {
                    std::array<Dtype, Dim> mean_pos = gp->running_mean_position.load();
                    for (std::size_t i = 0; i < Dim; ++i) {
                        s.write(
                            reinterpret_cast<const char *>(&CHECKED_AT(mean_pos, i)),
                            sizeof(Dtype));
                    }
                    return s.good();
                },
            },
            {
                "running_num_samples",
                [](const SdfGaussianProcess *gp, std::ostream &s) -> bool {
                    s.write(
                        reinterpret_cast<const char *>(&gp->running_num_samples),
                        sizeof(gp->running_num_samples));
                    return s.good();
                },
            },
            {
                "sign_gp",
                [](const SdfGaussianProcess *gp, std::ostream &s) -> bool {
                    s << (gp->sign_gp != nullptr) << '\n';
                    if (gp->sign_gp != nullptr && !gp->sign_gp->Write(s)) { return false; }
                    return s.good();
                },
            },
            {
                "edf_gp",
                [](const SdfGaussianProcess *gp, std::ostream &s) -> bool {
                    s << (gp->edf_gp != nullptr) << '\n';
                    if (gp->edf_gp != nullptr && !gp->edf_gp->Write(s)) { return false; }
                    return s.good();
                },
            },
        };
        return WriteTokens(stream, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    bool
    SdfGaussianProcess<Dtype, Dim>::Read(std::istream &stream) {
        using namespace common;
        using namespace common::serialization;
        static const TokenReadFunctionPairs<SdfGaussianProcess> token_function_pairs = {
            {
                "active",
                [](SdfGaussianProcess *gp, std::istream &s) -> bool {
                    s >> gp->active;
                    return s.good();
                },
            },
            {
                "time_stamp",
                [](SdfGaussianProcess *gp, std::istream &s) -> bool {
                    s.read(reinterpret_cast<char *>(&gp->time_stamp), sizeof(gp->time_stamp));
                    return s.good();
                },
            },
            {
                "buf_outdated_count",
                [](SdfGaussianProcess *gp, std::istream &s) -> bool {
                    s.read(
                        reinterpret_cast<char *>(&gp->buf_outdated_count),
                        sizeof(gp->buf_outdated_count));
                    return s.good();
                },
            },
            {
                "gp_outdated_count",
                [](SdfGaussianProcess *gp, std::istream &s) -> bool {
                    long count = 0;
                    s.read(reinterpret_cast<char *>(&count), sizeof(count));
                    gp->gp_outdated_count.store(count);
                    return s.good();
                },
            },
            {
                "query_count",
                [](SdfGaussianProcess *gp, std::istream &s) -> bool {
                    s.read(reinterpret_cast<char *>(&gp->query_count), sizeof(gp->query_count));
                    return s.good();
                },
            },
            {
                "use_normal_gp",
                [](SdfGaussianProcess *gp, std::istream &s) -> bool {
                    s >> gp->use_normal_gp;
                    return s.good();
                },
            },
            {
                "position",
                [](SdfGaussianProcess *gp, std::istream &s) -> bool {
                    return LoadEigenMatrixFromBinaryStream(s, gp->position) && s.good();
                },
            },
            {
                "half_size",
                [](SdfGaussianProcess *gp, std::istream &s) -> bool {
                    s.read(reinterpret_cast<char *>(&gp->half_size), sizeof(gp->half_size));
                    return s.good();
                },
            },
            {
                "running_mean_position",
                [](SdfGaussianProcess *gp, std::istream &s) -> bool {
                    std::array<Dtype, Dim> mean_pos{};
                    for (std::size_t i = 0; i < Dim; ++i) {
                        s.read(reinterpret_cast<char *>(&CHECKED_AT(mean_pos, i)), sizeof(Dtype));
                    }
                    gp->running_mean_position.store(mean_pos);
                    return s.good();
                },
            },
            {
                "running_num_samples",
                [](SdfGaussianProcess *gp, std::istream &s) -> bool {
                    s.read(
                        reinterpret_cast<char *>(&gp->running_num_samples),
                        sizeof(gp->running_num_samples));
                    return s.good();
                },
            },
            {
                "sign_gp",
                [](SdfGaussianProcess *gp, std::istream &s) -> bool {
                    bool has_gp = false;
                    s >> has_gp;
                    SkipLine(s);
                    if (!has_gp) {  // no sign GP, skip
                        gp->sign_gp = nullptr;
                        return s.good();
                    }
                    if (gp->sign_gp == nullptr) {
                        gp->sign_gp = std::make_shared<SignGp>(gp->setting->sign_gp);
                    }
                    return gp->sign_gp->Read(s) && s.good();
                },
            },
            {
                "edf_gp",
                [](SdfGaussianProcess *gp, std::istream &s) -> bool {
                    bool has_gp = false;
                    s >> has_gp;
                    SkipLine(s);
                    if (!has_gp) {  // no EDF GP, skip
                        gp->edf_gp = nullptr;
                        return s.good();
                    }
                    if (gp->edf_gp == nullptr) {
                        gp->edf_gp = std::make_shared<EdfGp>(gp->setting->edf_gp);
                    }
                    return gp->edf_gp->Read(s) && s.good();
                },
            },
        };
        return ReadTokens(stream, this, token_function_pairs);
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

        const typename LogEdfGaussianProcess<Dtype>::TrainBuf &train_buf = edf_gp->GetTrainBuffer();
        const long num_samples = train_buf.num_samples;
        const Dtype softmin_temperature = setting->softmin_temperature;
        const bool compute_cov_grad = compute_gradient_variance || compute_covariance;

        VectorX s(num_samples);
        Dtype s_sum = 0;
        VectorX z(num_samples);
        Eigen::Matrix<Dtype, Dim, Eigen::Dynamic> mat_v(Dim, num_samples);
        for (long k = 0; k < num_samples; ++k) {
            const VectorD v = test_position - train_buf.x.col(k);
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
            var[0] += w * w * train_buf.var_x[k];
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
                cov_grad += train_buf.var_x[j] * (grad_j.transpose() * grad_j);
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

    template struct SdfGaussianProcessSetting<double>;
    template struct SdfGaussianProcessSetting<float>;
    template struct SdfGaussianProcess<double, 3>;
    template struct SdfGaussianProcess<float, 3>;
    template struct SdfGaussianProcess<double, 2>;
    template struct SdfGaussianProcess<float, 2>;
}  // namespace erl::gp_sdf
