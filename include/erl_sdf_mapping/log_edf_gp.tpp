#pragma once

#include "erl_covariance/reduced_rank_covariance.hpp"

namespace erl::sdf_mapping {

    template<typename Dtype>
    YAML::Node
    LogEdfGaussianProcess<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node = YAML::convert<typename Super::Setting>::encode(setting);
        node["log_lambda"] = setting.log_lambda;
        return node;
    }

    template<typename Dtype>
    bool
    LogEdfGaussianProcess<Dtype>::Setting::YamlConvertImpl::decode(const YAML::Node &node, Setting &setting) {
        if (!node.IsMap()) { return false; }
        YAML::convert<typename Super::Setting>::decode(node, setting);
        setting.log_lambda = node["log_lambda"].as<Dtype>();
        return true;
    }

    template<typename Dtype>
    LogEdfGaussianProcess<Dtype>::LogEdfGaussianProcess(std::shared_ptr<Setting> setting)
        : Super([setting]() -> std::shared_ptr<Setting> {
              setting->kernel->scale = std::sqrt(3.) / setting->log_lambda;
              setting->no_gradient_observation = true;
              return setting;
          }()),
          m_setting_(std::move(setting)) {}

    template<typename Dtype>
    std::size_t
    LogEdfGaussianProcess<Dtype>::GetMemoryUsage() const {
        std::size_t memory_usage = Super::GetMemoryUsage();
        memory_usage += sizeof(*this) - sizeof(Super);
        return memory_usage;
    }

    template<typename Dtype>
    template<int Dim>
    long
    LogEdfGaussianProcess<Dtype>::LoadSurfaceData(
        std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices,
        const std::vector<SurfaceData<Dtype, Dim>> &surface_data_vec,
        const Eigen::Vector<Dtype, Dim> &coord_origin,
        const Dtype offset_distance,
        const Dtype sensor_noise,
        const Dtype max_valid_gradient_var,
        const Dtype invalid_position_var) {

        this->SetKernelCoordOrigin(coord_origin);

        const long max_num_samples = std::min(m_setting_->max_num_samples, static_cast<long>(surface_data_vec.size()));
        this->Reset(max_num_samples, Dim);

        std::sort(surface_data_indices.begin(), surface_data_indices.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

        long count = 0;
        for (auto &[distance, surface_data_index]: surface_data_indices) {
            auto &surface_data = surface_data_vec[surface_data_index];
            this->m_mat_x_train_.col(count) = surface_data.position - offset_distance * surface_data.normal;
            this->m_vec_var_h_[count] = sensor_noise;
            this->m_vec_var_x_[count] = surface_data.var_position;
            // this->m_vec_grad_flag_[count] = false;  // m_setting_->no_gradient_observation is true, so no need to set this
            if ((surface_data.var_normal > max_valid_gradient_var) ||                                   // invalid gradient
                (surface_data.normal.norm() < 0.9)) {                                                   // invalid normal
                this->m_vec_var_x_[count] = std::max(this->m_vec_var_x_[count], invalid_position_var);  // position is unreliable
            }
            if (++count >= this->m_mat_x_train_.cols()) { break; }  // reached max_num_samples
        }
        this->m_num_train_samples_ = count;
        this->m_vec_y_train_.setOnes(this->m_num_train_samples_);
        if (this->m_reduced_rank_kernel_) { this->UpdateKtrain(this->m_num_train_samples_); }
        return count;
    }

    template<typename Dtype>
    bool
    LogEdfGaussianProcess<Dtype>::Test(
        const Eigen::Ref<const MatrixX> &mat_x_test,
        Eigen::Ref<MatrixX> mat_f_out,
        Eigen::Ref<MatrixX> mat_var_out,
        Eigen::Ref<MatrixX> mat_cov_out,
        const bool predict_gradient) const {

        if (!Super::Test(mat_x_test, mat_f_out, mat_var_out, mat_cov_out, predict_gradient)) { return false; }
        const long dim = mat_x_test.rows();
        const long n = mat_x_test.cols();
        const Dtype log_lambda = m_setting_->log_lambda;
        for (long i = 0; i < n; ++i) {
            // edf
            Dtype *f = mat_f_out.col(i).data();
            const Dtype f_log_gpis = f[0];
            f[0] = std::log(std::abs(f_log_gpis)) / -log_lambda;
            // gradient
            Dtype norm = 0;
            const Dtype d = -1.0 / (log_lambda * std::abs(f_log_gpis));  // d = -ln(f)/lambda, grad_d = -1/(lambda*f)*grad_f
            for (long j = 1; j <= dim; ++j) {                            // gradient
                Dtype &grad = f[j];                                      // grad_f
                grad *= d;                                               // grad_d
                norm += grad * grad;
            }
            norm = std::sqrt(norm);
            if (norm > 1.e-15) {                                   // avoid zero division
                for (long j = 1; j <= dim; ++j) { f[j] /= norm; }  // gradient norm is always 1. https://en.wikipedia.org/wiki/Eikonal_equation
            }
        }
        return true;
    }

    template<typename Dtype>
    bool
    LogEdfGaussianProcess<Dtype>::operator==(const LogEdfGaussianProcess &other) const {
        if (!Super::operator==(other)) { return false; }
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) { return false; }
        return true;
    }

    template<typename Dtype>
    bool
    LogEdfGaussianProcess<Dtype>::Write(std::ostream &s) const {
        if (!Super::Write(s)) {
            ERL_WARN("Failed to write parent class NoisyInputGaussianProcess.");
            return false;
        }
        s << kFileHeader << std::endl  //
          << "# (feel free to add / change comments, but leave the first line as it is!)" << std::endl
          << "setting" << std::endl;
        // write setting
        if (!m_setting_->Write(s)) {
            ERL_WARN("Failed to write setting.");
            return false;
        }
        s << "end_of_LogEdfGaussianProcess" << std::endl;
        return s.good();
    }

    template<typename Dtype>
    bool
    LogEdfGaussianProcess<Dtype>::Read(std::istream &s) {
        if (!Super::Read(s)) {
            ERL_WARN("Failed to read parent class NoisyInputGaussianProcess.");
            return false;
        }

        if (!s.good()) {
            ERL_WARN("Input stream is not ready for reading");
            return false;
        }

        // check if the first line is valid
        std::string line;
        std::getline(s, line);
        if (line.compare(0, kFileHeader.length(), kFileHeader) != 0) {  // check if the first line is valid
            ERL_WARN("Header does not start with \"{}\"", kFileHeader.c_str());
            return false;
        }

        auto skip_line = [&s]() {
            char c;
            do { c = static_cast<char>(s.get()); } while (s.good() && c != '\n');
        };

        static const char *tokens[] = {
            "setting",
            "end_of_LogEdfGaussianProcess",
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
                case 0: {  // setting
                    skip_line();
                    if (!m_setting_->Read(s)) {
                        ERL_WARN("Failed to read setting.");
                        return false;
                    }
                    break;
                }
                case 1: {  // end_of_LogEdfGaussianProcess
                    skip_line();
                    return true;
                }
                default: {
                    ERL_WARN("Unknown token: {}", token);
                    return false;
                }
            }
            ++token_idx;
        }
        ERL_WARN("Failed to read NoisyInputGaussianProcess. Truncated file?");
        return false;  // should not reach here
    }
}  // namespace erl::sdf_mapping
