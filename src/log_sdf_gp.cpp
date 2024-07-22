#include "erl_sdf_mapping/log_sdf_gp.hpp"

namespace erl::sdf_mapping {

    void
    LogSdfGaussianProcess::Reset(const long max_num_samples, const long x_dim) {
        if (m_setting_->kernel->x_dim > 0) {
            ERL_ASSERTM(x_dim == m_setting_->kernel->x_dim, "x_dim {} does not match kernel x_dim {}", x_dim, m_setting_->kernel->x_dim);
        }
        NoisyInputGaussianProcess::Reset(max_num_samples, x_dim);
        if (m_setting_->max_num_samples <= 0 || m_setting_->kernel->x_dim <= 0) { AllocateMemory2(max_num_samples, x_dim); }
        if (m_setting_->unify_scale) {
            m_setting_->kernel->scale = std::sqrt(3.) / m_setting_->log_lambda;
            m_kernel_ = covariance::Covariance::CreateCovariance(m_setting_->kernel_type, m_setting_->kernel);
        } else {
            const auto kernel_setting = std::make_shared<covariance::Covariance::Setting>(*m_setting_->kernel);  // copy kernel setting
            kernel_setting->scale = std::sqrt(3.) / m_setting_->log_lambda;
            m_kernel_ = covariance::Covariance::CreateCovariance(m_setting_->kernel_type, kernel_setting);
        }
    }

    std::size_t
    LogSdfGaussianProcess::GetMemoryUsage() const {
        std::size_t memory_usage = NoisyInputGaussianProcess::GetMemoryUsage();
        memory_usage += sizeof(*this) - sizeof(NoisyInputGaussianProcess);
        if (m_kernel_ != nullptr) { memory_usage += m_kernel_->GetMemoryUsage(); }
        memory_usage += m_mat_log_k_train_.size() * sizeof(double);
        memory_usage += m_mat_log_l_.size() * sizeof(double);
        memory_usage += m_vec_log_alpha_.size() * sizeof(double);
        return memory_usage;
    }

    void
    LogSdfGaussianProcess::Train(const long num_train_samples) {

        if (m_trained_) {
            ERL_WARN("The model has been trained. Please reset the model before training.");
            return;
        }

        m_num_train_samples_ = num_train_samples;
        if (m_num_train_samples_ <= 0) {
            ERL_WARN("num_train_samples = {}, it should be > 0.", m_num_train_samples_);
            return;
        }

        InitializeVectorAlpha();  // initialize m_vec_alpha_

        // Compute kernel matrix
        const auto [ktrain_rows, ktrain_cols] = NoisyInputGaussianProcess::m_kernel_->ComputeKtrainWithGradient(  // gpis
            m_mat_x_train_,
            m_num_train_samples_,
            m_vec_grad_flag_,
            m_vec_var_x_,
            m_vec_var_h_,
            m_vec_var_grad_,
            m_mat_k_train_);
        ERL_DEBUG_ASSERT(!m_mat_k_train_.topLeftCorner(ktrain_rows, ktrain_cols).hasNaN(), "NaN in m_mat_k_train_!");
        // log-gpis
        const auto [log_ktrain_rows, log_ktrain_cols] = covariance::Covariance::GetMinimumKtrainSize(m_num_train_samples_, 0, m_mat_x_train_.rows());
        if (m_setting_->unify_scale) {
            m_mat_log_k_train_.topLeftCorner(log_ktrain_rows, log_ktrain_cols) = m_mat_k_train_.topLeftCorner(ktrain_rows, ktrain_cols);
            m_mat_log_k_train_.diagonal().head(num_train_samples).array() -= m_vec_var_x_.head(num_train_samples).array();
        } else {  // scale is different
            (void) m_kernel_->ComputeKtrain(m_mat_x_train_, m_vec_var_h_, m_num_train_samples_, m_mat_log_k_train_);
            ERL_DEBUG_ASSERT(!m_mat_log_k_train_.topLeftCorner(log_ktrain_rows, log_ktrain_cols).hasNaN(), "NaN in m_mat_log_k_train_!");
        }

        // Compute log-sdf mapping
        const auto vec_alpha = m_vec_alpha_.head(ktrain_rows);        // h and gradient of h
        auto vec_log_alpha = m_vec_log_alpha_.head(log_ktrain_cols);  // log mapping of h
        vec_log_alpha.head(log_ktrain_cols).setOnes();

        // Compute cholesky decomposition and alpha
        auto &&mat_l = m_mat_l_.topLeftCorner(ktrain_rows, ktrain_cols);                                 // square matrix, lower triangular
        mat_l = m_mat_k_train_.topLeftCorner(ktrain_rows, ktrain_cols).llt().matrixL();                  // gpis, Ktrain = L * L^T
        mat_l.triangularView<Eigen::Lower>().solveInPlace(vec_alpha);                                    // Ktrain^-1 = L^-T * L^-1
        mat_l.transpose().triangularView<Eigen::Upper>().solveInPlace(vec_alpha);                        // alpha = Ktrain^-1 * [h, dh/dx_1, ..., dh/dx_dim]
        auto &&mat_log_l = m_mat_log_l_.topLeftCorner(log_ktrain_rows, log_ktrain_cols);                 // square matrix, lower triangular
        mat_log_l = m_mat_log_k_train_.topLeftCorner(log_ktrain_rows, log_ktrain_cols).llt().matrixL();  // log-gpis, logKtrain = logL * logL^T
        mat_log_l.triangularView<Eigen::Lower>().solveInPlace(vec_log_alpha);                            // logKtrain^-1 = logL^-T * logL^-1
        mat_log_l.transpose().triangularView<Eigen::Upper>().solveInPlace(vec_log_alpha);                // log_alpha = logKtrain^-1 * log(-lambda * h)

        m_trained_ = true;
    }

    void
    LogSdfGaussianProcess::Test(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test,
        Eigen::Ref<Eigen::MatrixXd> mat_f_out,
        Eigen::Ref<Eigen::MatrixXd> mat_var_out,
        Eigen::Ref<Eigen::MatrixXd> mat_cov_out) const {

        if (!m_trained_) {
            ERL_WARN("The model has not been trained.");
            return;
        }

        long dim = mat_x_test.rows();
        long n = mat_x_test.cols();
        if (n == 0) { return; }

        // compute mean and gradient of the test queries
        ERL_ASSERTM(mat_f_out.rows() >= dim + 1, "mat_f_out.rows() = {}, it should be >= Dim + 1 = {}.", mat_f_out.rows(), dim + 1);
        ERL_ASSERTM(mat_f_out.cols() >= n, "mat_f_out.cols() = {}, it should be >= n = {}.", mat_f_out.cols(), n);
        double f_threshold = std::exp(-m_setting_->log_lambda * std::abs(m_setting_->edf_threshold));

        const auto [ktest_min_rows, ktest_min_cols] = covariance::Covariance::GetMinimumKtestSize(m_num_train_samples_, m_num_train_samples_with_grad_, dim, n);
        Eigen::MatrixXd ktest(ktest_min_rows, ktest_min_cols);                                                 // (dim of train samples, dim of test queries)
        const auto [ktest_rows, ktest_cols] = NoisyInputGaussianProcess::m_kernel_->ComputeKtestWithGradient(  //
            m_mat_x_train_,
            m_num_train_samples_,
            m_vec_grad_flag_,
            mat_x_test,
            n,
            ktest);
        ERL_DEBUG_ASSERT(
            (ktest_rows == ktest_min_rows) && (ktest_cols == ktest_min_cols),
            "output_size = ({}, {}), it should be ({}, {}).",
            ktest_rows,
            ktest_cols,
            ktest_min_rows,
            ktest_min_cols);

        Eigen::MatrixXd log_ktest;
        if (m_setting_->unify_scale) {
            log_ktest = ktest.topRows(m_num_train_samples_);  // copy is not necessary
        } else {
            const auto [log_ktest_min_rows, log_ktest_min_cols] = covariance::Covariance::GetMinimumKtestSize(m_num_train_samples_, 0, dim, n);
            log_ktest.resize(log_ktest_min_rows, log_ktest_min_cols);  // (dim of train samples, dim of test queries)
            Eigen::VectorXl vec_grad_flag2 = Eigen::VectorXl::Constant(m_num_train_samples_, false);
            (void) m_kernel_->ComputeKtestWithGradient(m_mat_x_train_, m_num_train_samples_, vec_grad_flag2, mat_x_test, n, log_ktest);
        }

        // output sdf and gradient
        auto vec_alpha = m_vec_alpha_.head(ktest_rows);
        auto vec_log_alpha = m_vec_log_alpha_.head(m_num_train_samples_);
        for (long i = 0; i < n; ++i) {
            double f_gpis = ktest.col(i).dot(vec_alpha);
            double sign = f_gpis >= 0. ? 1. : -1.;  // sdf sign
            double f_log_gpis = log_ktest.col(i).dot(vec_log_alpha);

            mat_f_out(0, i) = std::log(std::abs(f_log_gpis)) * sign / -m_setting_->log_lambda;  // sdf magnitude
            ERL_DEBUG_ASSERT(!std::isinf(mat_f_out(0, i)), "inf. sdf!");
            double norm = 0;                                            // gradient norm
            if (f_log_gpis > f_threshold) {                             // close to the surface: use gpis
                for (long j = 1, jj = i + n; j <= dim; ++j, jj += n) {  // gradient
                    double &grad = mat_f_out(j, i);
                    grad = ktest.col(jj).dot(vec_alpha);
                    norm += grad * grad;
                }
            } else {                                                       // use log-gpis
                double d = -sign / (m_setting_->log_lambda * f_log_gpis);  // d = -ln(f)/lambda, grad_d = -1/(lambda*f)*grad_f
                for (long j = 1, jj = i + n; j <= dim; ++j, jj += n) {     // gradient
                    double &grad = mat_f_out(j, i);
                    grad = log_ktest.col(jj).dot(vec_log_alpha) * d;
                    norm += grad * grad;
                }
            }
            norm = std::sqrt(norm);
            if (norm > 1.e-15) {                                              // avoid zero division
                for (long j = 1; j <= dim; ++j) { mat_f_out(j, i) /= norm; }  // gradient norm is always 1. https://en.wikipedia.org/wiki/Eikonal_equation
            }
        }
        if (mat_var_out.size() == 0 && mat_cov_out.size() == 0) { return; }

        // compute (co)variance of the test queries: use gpis, log-gpis has numerical issue!!!
        // solve Lx = ktest -> x = m_mat_l_.m_inv_() * ktest
        m_mat_l_.topLeftCorner(ktest_rows, ktest_rows).triangularView<Eigen::Lower>().solveInPlace(ktest);
        bool compute_var = mat_var_out.size() > 0;
        if (compute_var) {
            ERL_ASSERTM(mat_var_out.rows() >= dim + 1, "mat_var_out.rows() = {}, it should be >= Dim + 1 = {} for variance.", mat_var_out.rows(), dim + 1);
            ERL_ASSERTM(mat_var_out.cols() >= n, "mat_var_out.cols() = {}, not enough for {} test queries.", mat_var_out.cols(), n);
        }
        if (mat_cov_out.size() == 0 && compute_var) {  // compute variance
            // column-wise square sum of ktest = var([h(x1),...,h(xn),dh(x1)/dx_1,...,dh(xn)/dx_1,...,dh(x1)/dx_dim,...,dh(xn)/dx_dim])
            for (long i = 0; i < n; ++i) {
                mat_var_out(0, i) = m_setting_->kernel->alpha - ktest.col(i).squaredNorm();  // variance of h(x)
                for (long j = 1, jj = i + n; j <= dim; ++j, jj += n) {                       // variance of dh(x)/dx_j
                    mat_var_out(j, i) = m_three_over_scale_square_ - ktest.col(jj).squaredNorm();
                }
            }
        } else {
            long min_n_rows = (dim + 1) * dim / 2;
            ERL_ASSERTM(mat_cov_out.rows() >= min_n_rows, "mat_cov_out.rows() = {}, it should be >= {} for covariance.", mat_cov_out.rows(), min_n_rows);
            ERL_ASSERTM(mat_cov_out.cols() >= n, "mat_cov_out.cols() = {}, not enough for {} test queries.", mat_cov_out.cols(), n);
            // each column of mat_cov_out is the lower triangular part of the covariance matrix of the corresponding test query
            for (long i = 0; i < n; ++i) {
                if (compute_var) { mat_var_out(0, i) = m_setting_->kernel->alpha - ktest.col(i).squaredNorm(); }  // var(h(x))
                long index = 0;
                for (long j = 1, jj = i + n; j <= dim; ++j, jj += n) {
                    const auto &col_jj = ktest.col(jj);
                    mat_cov_out(index++, i) = -col_jj.dot(ktest.col(i));                                                         // cov(dh(x)/dx_j, h(x))
                    for (long k = 1, kk = i + n; k < j; ++k, kk += n) { mat_cov_out(index++, i) = -col_jj.dot(ktest.col(kk)); }  // cov(dh(x)/dx_j, dh(x)/dx_k)
                    if (compute_var) { mat_var_out(j, i) = m_three_over_scale_square_ - col_jj.squaredNorm(); }                  // var(dh(x)/dx_j)
                }
            }
        }
    }

    bool
    LogSdfGaussianProcess::operator==(const LogSdfGaussianProcess &other) const {
        if (!NoisyInputGaussianProcess::operator==(other)) { return false; }
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) { return false; }
        if (m_num_train_samples_ == 0) { return true; }  // no training samples, no need to compare the following
        const auto [rows, cols] = covariance::Covariance::GetMinimumKtrainSize(m_num_train_samples_, m_num_train_samples_with_grad_, m_x_dim_);
        if (m_mat_log_k_train_.topLeftCorner(rows, cols) != other.m_mat_log_k_train_.topLeftCorner(rows, cols)) { return false; }
        if (m_mat_log_l_.topLeftCorner(rows, cols) != other.m_mat_log_l_.topLeftCorner(rows, cols)) { return false; }
        if (m_vec_log_alpha_.head(cols) != other.m_vec_log_alpha_.head(cols)) { return false; }
        return true;
    }

    bool
    LogSdfGaussianProcess::Write(const std::string &filename) const {
        ERL_INFO("Writing LogSdfGaussianProcess to file: {}", filename);
        std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
        std::ofstream file(filename, std::ios_base::out | std::ios_base::binary);
        if (!file.is_open()) {
            ERL_WARN("Failed to open file: {}", filename);
            return false;
        }

        const bool success = Write(file);
        file.close();
        return success;
    }

    static const std::string kFileHeader = "# erl::sdf_mapping::LogSdfGaussianProcess";

    bool
    LogSdfGaussianProcess::Write(std::ostream &s) const {
        if (!NoisyInputGaussianProcess::Write(s)) {
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
        // write data
        s << "mat_log_k_train" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, m_mat_log_k_train_)) {
            ERL_WARN("Failed to write mat_log_k_train.");
            return false;
        }
        s << "mat_log_l" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, m_mat_log_l_)) {
            ERL_WARN("Failed to write mat_log_l.");
            return false;
        }
        s << "vec_log_alpha" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, m_vec_log_alpha_)) {
            ERL_WARN("Failed to write vec_log_alpha.");
            return false;
        }
        s << "end_of_LogSdfGaussianProcess" << std::endl;
        return s.good();
    }

    bool
    LogSdfGaussianProcess::Read(const std::string &filename) {
        ERL_INFO("Reading LogSdfGaussianProcess from file: {}", std::filesystem::absolute(filename));
        std::ifstream file(filename.c_str(), std::ios_base::in | std::ios_base::binary);
        if (!file.is_open()) {
            ERL_WARN("Failed to open file: {}", filename.c_str());
            return false;
        }

        const bool success = Read(file);
        file.close();
        return success;
    }

    bool
    LogSdfGaussianProcess::Read(std::istream &s) {
        if (!NoisyInputGaussianProcess::Read(s)) {
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
            "mat_log_k_train",
            "mat_log_l",
            "vec_log_alpha",
            "end_of_LogSdfGaussianProcess",
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
                case 1: {  // mat_log_k_train
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_mat_log_k_train_)) {
                        ERL_WARN("Failed to read mat_log_k_train.");
                        return false;
                    }
                    break;
                }
                case 2: {  // mat_log_l
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_mat_log_l_)) {
                        ERL_WARN("Failed to read mat_log_l.");
                        return false;
                    }
                    break;
                }
                case 3: {  // vec_log_alpha
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_vec_log_alpha_)) {
                        ERL_WARN("Failed to read vec_log_alpha.");
                        return false;
                    }
                    break;
                }
                case 4: {  // end_of_LogSdfGaussianProcess
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

    bool
    LogSdfGaussianProcess::AllocateMemory2(const long max_num_samples, const long x_dim) {
        const auto [rows, cols] = covariance::Covariance::GetMinimumKtrainSize(max_num_samples, 0, x_dim);
        if (m_mat_log_k_train_.rows() < rows || m_mat_log_k_train_.cols() < cols) { m_mat_log_k_train_.resize(rows, cols); }
        if (m_mat_log_l_.rows() < rows || m_mat_log_l_.cols() < cols) { m_mat_log_l_.resize(rows, cols); }
        if (m_vec_log_alpha_.size() < max_num_samples) { m_vec_log_alpha_.resize(max_num_samples); }
        return true;
    }

}  // namespace erl::sdf_mapping
