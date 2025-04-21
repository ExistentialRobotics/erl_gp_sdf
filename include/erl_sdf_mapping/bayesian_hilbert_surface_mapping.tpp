#pragma once

namespace erl::sdf_mapping {

    template<typename Dtype>
    YAML::Node
    LocalBayesianHilbertMapSetting<Dtype>::YamlConvertImpl::encode(const LocalBayesianHilbertMapSetting &setting) {
        YAML::Node node;
        node["bhm"] = setting.bhm;
        node["kernel_type"] = setting.kernel_type;
        node["kernel_setting_type"] = setting.kernel_setting_type;
        node["kernel"] = setting.kernel;
        node["max_dataset_size"] = setting.max_dataset_size;
        node["hit_buffer_size"] = setting.hit_buffer_size;
        node["track_surface"] = setting.track_surface;
        node["surface_resolution"] = setting.surface_resolution;
        node["surface_occ_prob_threshold"] = setting.surface_occ_prob_threshold;
        node["surface_occ_prob_target"] = setting.surface_occ_prob_target;
        node["surface_adjust_step"] = setting.surface_adjust_step;
        node["var_scale"] = setting.var_scale;
        return node;
    }

    template<typename Dtype>
    bool
    LocalBayesianHilbertMapSetting<Dtype>::YamlConvertImpl::decode(const YAML::Node &node, LocalBayesianHilbertMapSetting &setting) {
        if (!node.IsMap()) { return false; }
        setting.bhm = node["bhm"].as<decltype(setting.bhm)>();
        setting.kernel_type = node["kernel_type"].as<std::string>();
        setting.kernel_setting_type = node["kernel_setting_type"].as<std::string>();
        setting.kernel = common::YamlableBase::Create<KernelSetting>(setting.kernel_setting_type);
        if (!setting.kernel->FromYamlNode(node["kernel"])) { return false; }
        setting.max_dataset_size = node["max_dataset_size"].as<long>();
        setting.hit_buffer_size = node["hit_buffer_size"].as<long>();
        setting.track_surface = node["track_surface"].as<bool>();
        setting.surface_resolution = node["surface_resolution"].as<Dtype>();
        setting.surface_occ_prob_threshold = node["surface_occ_prob_threshold"].as<Dtype>();
        setting.surface_occ_prob_target = node["surface_occ_prob_target"].as<Dtype>();
        setting.surface_adjust_step = node["surface_adjust_step"].as<Dtype>();
        setting.var_scale = node["var_scale"].as<Dtype>();
        return true;
    }

    template<typename Dtype, int Dim>
    YAML::Node
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node;
        node["local_bhm"] = setting.local_bhm;
        node["tree"] = setting.tree;
        node["hinged_grid_size"] = setting.hinged_grid_size;
        node["bhm_depth"] = setting.bhm_depth;
        node["bhm_overlap"] = setting.bhm_overlap;
        node["bhm_test_margin"] = setting.bhm_test_margin;
        node["test_knn"] = setting.test_knn;
        node["test_batch_size"] = setting.test_batch_size;
        node["build_bhm_on_hit"] = setting.build_bhm_on_hit;
        return node;
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Setting::YamlConvertImpl::decode(const YAML::Node &node, Setting &setting) {
        if (!node.IsMap()) { return false; }
        setting.local_bhm = node["local_bhm"].as<decltype(setting.local_bhm)>();
        setting.tree = node["tree"].as<decltype(setting.tree)>();
        setting.hinged_grid_size = node["hinged_grid_size"].as<int>();
        setting.bhm_depth = node["bhm_depth"].as<uint32_t>();
        setting.bhm_overlap = node["bhm_overlap"].as<Dtype>();
        setting.bhm_test_margin = node["bhm_test_margin"].as<Dtype>();
        setting.test_knn = node["test_knn"].as<int>();
        setting.test_batch_size = node["test_batch_size"].as<int>();
        setting.build_bhm_on_hit = node["build_bhm_on_hit"].as<bool>();
        return true;
    }

    template<typename Dtype, int Dim>
    BayesianHilbertSurfaceMapping<Dtype, Dim>::LocalBayesianHilbertMap::LocalBayesianHilbertMap(
        std::shared_ptr<LocalBayesianHilbertMapSetting<Dtype>> setting_,
        Positions hinged_points,
        Aabb map_boundary,
        uint64_t seed,
        std::optional<Aabb> track_surface_boundary_)
        : setting(std::move(setting_)),
          tracked_surface_boundary(track_surface_boundary_.has_value() ? track_surface_boundary_.value() : map_boundary),
          bhm(setting->bhm, Covariance::CreateCovariance(setting->kernel_type, setting->kernel), std::move(hinged_points), std::move(map_boundary), seed) {

        const Position map_size = tracked_surface_boundary.sizes();
        Eigen::Vector<int, Dim> map_shape;
        for (long dim = 0; dim < Dim; ++dim) { map_shape[dim] = static_cast<int>(std::ceil(map_size[dim] / setting->surface_resolution)); }
        strides = common::ComputeCStrides<int>(map_shape, 1);

        if (setting->hit_buffer_size > 0) { hit_buffer.reserve(setting->hit_buffer_size); }  // reserve the hit buffer size
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::LocalBayesianHilbertMap::Update(
        const Eigen::Ref<const Position> &sensor_origin,
        const Eigen::Ref<const Positions> &points) {

        if (!UpdateBhm(sensor_origin, points)) { return false; }
        if (!setting->track_surface) { return true; }
        TrackSurface(points);
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::LocalBayesianHilbertMap::UpdateBhm(
        const Eigen::Ref<const Position> &sensor_origin,
        const Eigen::Ref<const Positions> &points) {
        // ERL_BLOCK_TIMER_MSG("local bhm update");

        const long max_dataset_size = setting->max_dataset_size;
        {
            // ERL_BLOCK_TIMER_MSG("bhm generate dataset");
            bhm.GenerateDataset(sensor_origin, points, max_dataset_size, num_dataset_points, dataset_points, dataset_labels, hit_indices);
        }
        if (num_dataset_points == 0) { return false; }
        if (!hit_buffer.empty() && (max_dataset_size < 0 || num_dataset_points < max_dataset_size)) {
            // ERL_BLOCK_TIMER_MSG("bhm add hit buffer");

            // there is data in the hit buffer, and
            // 1. no dataset size limit
            // 2. dataset size limit but not reached
            long n = num_dataset_points + static_cast<long>(hit_buffer.size());
            if (max_dataset_size > 0) { n = std::min(max_dataset_size, n); }
            if (n > dataset_points.cols()) {
                dataset_points.conservativeResize(Dim, n);
                dataset_labels.conservativeResize(n);
            }
            for (const Position &point: hit_buffer) {
                dataset_points.col(num_dataset_points) = point;
                dataset_labels[num_dataset_points] = 1;
                ++num_dataset_points;
                if (num_dataset_points >= n) { break; }  // stop if the dataset size limit is reached
            }
        }
        {
            // ERL_BLOCK_TIMER_MSG("bhm run expectation maximization");
            bhm.RunExpectationMaximization(dataset_points, dataset_labels, num_dataset_points);
        }
        if (setting->hit_buffer_size > 0 && !hit_indices.empty()) {
            // ERL_BLOCK_TIMER_MSG("bhm update hit buffer");

            // hit buffer has space and there are hit points
            for (const long &hit_index: hit_indices) {
                hit_buffer[hit_buffer_head] = points.col(hit_index);
                hit_buffer_head = (hit_buffer_head + 1) % hit_buffer.capacity();
            }
        }
        return true;
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::LocalBayesianHilbertMap::TrackSurface(const Eigen::Ref<const Positions> &points) {
        // ERL_BLOCK_TIMER_MSG("track surface");

        // add new surface points, update the existing surface points
        const Position map_min = tracked_surface_boundary.min();
        const Dtype surface_resolution = setting->surface_resolution;
        for (const auto &hit_index: hit_indices) {                            // for each hit point
            const Position hit_point = points.col(hit_index);                 // get the hit point
            if (!tracked_surface_boundary.contains(hit_point)) { continue; }  // skip if the hit point is out of the surface boundary
            Eigen::Vector<int, Dim> grid_coords;                              // grid coordinates of the hit point
            for (long dim = 0; dim < Dim; ++dim) { grid_coords[dim] = common::MeterToGrid<Dtype>(hit_point[dim], map_min[dim], surface_resolution); }
            const int surface_index = common::CoordsToIndex<Dim>(strides, grid_coords);  // compute the index of the surface point
            auto it = surface.find(surface_index);                                       // find the surface point in the map
            if (it != surface.end()) { continue; }                                       // already exist
            surface_indices.push_back(surface_index);                                    // add the index to the list
            surface.insert({surface_index, Surface(hit_point)});                         // add the new surface point
        }
        // collect the points
        Positions surface_points(Dim, static_cast<long>(surface_indices.size()));
        for (long i = 0; i < static_cast<long>(surface.size()); ++i) { surface_points.col(i) = surface.at(surface_indices[i]).position; }
        // compute the normals
        VectorX prob_occupied;
        Gradients gradients;
        // by default, OMP_NESTED is false, omp_set_nested(1) is needed to enable nested parallelism
        // parallel is set to true, but the parallelism will be disabled if this function is called in a parallel region
        bhm.Predict(
            surface_points,
            false /*logodd*/,
            true /*faster*/,
            true /*compute_gradient*/,
            false /*gradient_with_sigmoid*/,
            true /*parallel*/,
            prob_occupied,
            gradients);
        const Dtype prob_threshold = setting->surface_occ_prob_threshold;
        const Dtype prob_target = setting->surface_occ_prob_target;
        const Dtype step_alpha = setting->surface_adjust_step;
        for (long i = surface_points.cols() - 1; i >= 0; --i) {
            const int surface_point_index = surface_indices[i];
            if (prob_occupied[i] < prob_threshold) {
                surface.erase(surface_point_index);                  // remove the surface point if the occupancy probability is too low
                surface_indices.erase(surface_indices.begin() + i);  // remove the index from the list
                continue;
            }
            Gradient normal = -gradients.col(i);
            const Dtype norm = normal.norm();
            if (norm < std::numeric_limits<Dtype>::epsilon()) { continue; }  // skip if the normal is too small
            normal /= norm;

            auto &surf_pt = surface.at(surface_point_index);
            surf_pt.normal = normal;
            surf_pt.prob_occupied = prob_occupied[i];

            if (step_alpha == 0.0f) { continue; }  // no need to adjust the surface point
            surf_pt.position -= (step_alpha * (prob_target - prob_occupied[i])) * normal;
        }
    }

    template<typename Dtype, int Dim>
    BayesianHilbertSurfaceMapping<Dtype, Dim>::BayesianHilbertSurfaceMapping(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)) {
        ERL_ASSERTM(m_setting_ != nullptr, "setting is nullptr.");
        m_tree_ = std::make_shared<Tree>(m_setting_->tree);
        GenerateHingedPoints();
    }

    template<typename Dtype, int Dim>
    std::shared_ptr<const typename BayesianHilbertSurfaceMapping<Dtype, Dim>::Setting>
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetSetting() const {
        return m_setting_;
    }

    template<typename Dtype, int Dim>
    std::shared_ptr<const typename BayesianHilbertSurfaceMapping<Dtype, Dim>::Tree>
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetTree() const {
        return m_tree_;
    }

    template<typename Dtype, int Dim>
    const absl::flat_hash_map<
        typename BayesianHilbertSurfaceMapping<Dtype, Dim>::Key,
        std::shared_ptr<typename BayesianHilbertSurfaceMapping<Dtype, Dim>::LocalBayesianHilbertMap>> &
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetLocalBayesianHilbertMaps() const {
        return m_key_bhm_dict_;
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Update(
        const Eigen::Ref<const Position> &sensor_origin,
        const Eigen::Ref<const Positions> &points,
        const bool parallel) {

        // to update the occupancy tree first, the resolution of the tree should not be too high so that we will not spend too much time on the tree update.
        // the tree helps us find where to place local Bayesian Hilbert maps.
        constexpr bool lazy_eval = false;
        constexpr bool discrete = true;
        {
            ERL_BLOCK_TIMER_MSG("tree update");
            m_tree_->InsertPointCloud(points, sensor_origin, m_setting_->local_bhm->bhm->max_distance, false /*parallel*/, lazy_eval, discrete);
        }

        // find the local Bayesian Hilbert maps to build or update
        KeySet bhm_keys_set;
        const uint32_t bhm_depth = m_setting_->bhm_depth;
        {
            ERL_BLOCK_TIMER_MSG("bhm find");
            if (m_setting_->build_bhm_on_hit) {
                // any hit point will trigger building the corresponding local Bayesian Hilbert map
                const auto &end_point_maps = discrete ? m_tree_->GetDiscreteEndPointMaps() : m_tree_->GetEndPointMaps();
                for (const auto &[key, hit_indices]: end_point_maps) { bhm_keys_set.insert(m_tree_->AdjustKeyToDepth(key, bhm_depth)); }
            } else {
                // only the occupied node will trigger building the corresponding local Bayesian Hilbert map
                const auto &end_point_maps = discrete ? m_tree_->GetDiscreteEndPointMaps() : m_tree_->GetEndPointMaps();
                for (const auto &[key, hit_indices]: end_point_maps) {
                    if (const TreeNode *node = m_tree_->Search(key); node != nullptr && m_tree_->IsNodeOccupied(node)) {
                        bhm_keys_set.insert(m_tree_->AdjustKeyToDepth(key, bhm_depth));
                    }
                }
            }
        }

        // create bhm for new keys
        const KeyVector bhm_keys(bhm_keys_set.begin(), bhm_keys_set.end());
        {
            ERL_BLOCK_TIMER_MSG("bhm create");
            const Dtype half_surface_size = m_tree_->GetNodeSize(bhm_depth) * 0.5f;
            const Dtype half_bhm_size = half_surface_size + m_setting_->bhm_overlap;
            for (const Key &key: bhm_keys) {
                auto it = m_key_bhm_dict_.find(key);
                if (it != m_key_bhm_dict_.end()) { continue; }  // already exist

                Position map_center = m_tree_->KeyToCoord(key, bhm_depth);
                Positions hinged_points = m_hinged_points_.colwise() + map_center;
                Aabb map_boundary(map_center, half_bhm_size);
                uint64_t seed = typename Key::KeyHash()(key);
                m_key_bhm_positions_.emplace_back(key, map_center);
                m_key_bhm_dict_.insert(
                    {key,
                     std::make_shared<LocalBayesianHilbertMap>(m_setting_->local_bhm, hinged_points, map_boundary, seed, Aabb(map_center, half_surface_size))});
                m_bhm_kdtree_needs_update_ = true;  // need to update the kdtree after adding new bhm
            }
        }

        // update the local Bayesian Hilbert maps
        std::vector<int> updated(bhm_keys.size(), 0);
        {
            ERL_BLOCK_TIMER_MSG("bhm update");
            (void) parallel;
            ERL_INFO("{} local bhm(s) to update", bhm_keys.size());
#pragma omp parallel for if (parallel) default(none) shared(bhm_keys, updated, sensor_origin, points, std::cout)
            for (std::size_t i = 0; i < bhm_keys.size(); ++i) { updated[i] = m_key_bhm_dict_[bhm_keys[i]]->Update(sensor_origin, points); }
        }

        return std::any_of(updated.begin(), updated.end(), [](const int i) { return i > 0; });  // return true if any local bhm is updated
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Predict(
        const Eigen::Ref<const Positions> &points,
        const bool logodd,
        const bool faster,
        const bool compute_gradient,
        const bool gradient_with_sigmoid,
        const bool parallel,
        VectorX &prob_occupied,
        Gradients &gradient) const {

        const long num_points = points.cols();
        if (prob_occupied.size() < num_points) { prob_occupied.resize(num_points); }
        if (compute_gradient && gradient.cols() < num_points) { gradient.resize(Dim, num_points); }
        prob_occupied.fill(logodd ? 0.0f : 0.5f);  // initialize to unknown
        gradient.fill(0.0f);                       // initialize to zero
        BuildBhmKdtree();

        const int batch_size = m_setting_->test_batch_size;
        if (batch_size > num_points) {  // no need to run in parallel
            PredictThread(
                points.data(),
                0,
                num_points,
                logodd,
                faster,
                compute_gradient,
                gradient_with_sigmoid,
                parallel,  // let the thread decide
                prob_occupied.data(),
                gradient.data());
            return;
        }

        const uint32_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (long i = 0; i < num_points; i += batch_size) {
            long start = i;
            long end = std::min(i + batch_size, num_points);

            bool no_free = threads.size() >= num_threads;
            while (no_free) {
                for (auto it = threads.begin(); it != threads.end(); ++it) {
                    if (!it->joinable()) { continue; }
                    it->join();
                    threads.erase(it);
                    no_free = false;
                    break;
                }
                if (no_free) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
            }
            threads.emplace_back(
                &BayesianHilbertSurfaceMapping::PredictThread,
                this,
                points.col(0).data(),
                start,
                end,
                logodd,
                faster,
                compute_gradient,
                gradient_with_sigmoid,
                false /*parallel*/,  // no need to run in parallel within the thread
                prob_occupied.data(),
                gradient.data());
        }

        for (auto &thread: threads) { thread.join(); }
        threads.clear();

        // absl::flat_hash_map<Key, std::vector<long>> key_to_point_indices;
        // key_to_point_indices.reserve(points.size());
        // std::vector<Key> bhm_keys_set;
        // bhm_keys_set.reserve(points.size());
        // const uint32_t bhm_depth = m_setting_->bhm_depth;
        // const Dtype half_size = 0.5f * m_tree_->GetNodeSize(bhm_depth) + m_setting_->bhm_test_margin;
        // long bhm_index = -1;
        // Dtype bhm_distance = 0;
        // for (long i = 0; i < points.cols(); ++i) {
        //     m_bhm_kdtree_->Nearest(points.col(i), bhm_index, bhm_distance);             // find the nearest bhm
        //     bhm_distance = std::sqrt(bhm_distance);                                     // distance is squared
        //     if (bhm_distance > half_size || bhm_index < 0) {                            // too far from the bhm
        //         Key key;                                                                // use the tree to predict the occupancy
        //         if (!m_tree_->CoordToKeyChecked(points.col(i), key)) { continue; }      // outside the map
        //         const TreeNode *node = m_tree_->Search(key);                            // use the tree to predict the occupancy
        //         if (node == nullptr) { continue; }                                      // unknown
        //         prob_occupied[i] = logodd ? node->GetLogOdds() : node->GetOccupancy();  // get the occupancy from the tree
        //         continue;
        //     }
        //
        //     const Key bhm_key = m_key_bhm_positions_[bhm_index].first;
        //     auto it = key_to_point_indices.insert({bhm_key, std::vector<long>()});  // insert the key if not exist
        //     if (it.second) { bhm_keys_set.push_back(bhm_key); }                     // if the key is new, add it to the set
        //     it.first->second.push_back(i);
        // }
        //
        // pragma omp parallel for if (parallel) default(none) \
//     shared(bhm_keys_set, key_to_point_indices, points, logodd, faster, compute_gradient, gradient_with_sigmoid, parallel, prob_occupied, gradient)
        // for (const Key &bhm_key: bhm_keys_set) {
        //     const auto &indices = key_to_point_indices[bhm_key];
        //
        //     // copy the points of this key to a new matrix
        //     Positions points_of_key(Dim, static_cast<long>(indices.size()));
        //     for (long i = 0; i < points_of_key.cols(); ++i) { points_of_key.col(i) = points.col(indices[i]); }
        //
        //     // predict
        //     VectorX prob_occupied_of_key;
        //     Gradients gradients_of_key;
        //     std::shared_ptr<LocalBayesianHilbertMap> bhm = m_key_bhm_dict_.at(bhm_key);
        //     bhm->bhm.Predict(points_of_key, logodd, faster, compute_gradient, gradient_with_sigmoid, !parallel, prob_occupied_of_key, gradients_of_key);
        //
        //     // copy the results back to the original matrix
        //     for (long i = 0; i < points_of_key.cols(); ++i) {
        //         prob_occupied[indices[i]] = prob_occupied_of_key[i];
        //         if (compute_gradient) { gradient.col(indices[i]) = gradients_of_key.col(i); }
        //     }
        // }
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::PredictGradient(
        const Eigen::Ref<const Positions> &points,
        const bool faster,
        const bool with_sigmoid,
        const bool parallel,
        Gradients &gradient) const {

        // we can only predict the gradient when bhm is available for the point

        if (gradient.cols() < points.cols()) { gradient.resize(Dim, points.cols()); }
        gradient.fill(0.0f);
        BuildBhmKdtree();

        absl::flat_hash_map<Key, std::vector<long>> key_to_point_indices;
        key_to_point_indices.reserve(points.size());
        std::vector<Key> bhm_keys_set;
        bhm_keys_set.reserve(points.size());
        const uint32_t bhm_depth = m_setting_->bhm_depth;
        const Dtype half_size = 0.5f * m_tree_->GetNodeSize(bhm_depth) + m_setting_->bhm_test_margin;
        long bhm_index = -1;
        Dtype bhm_distance = 0;
        for (long i = 0; i < points.size(); ++i) {
            m_bhm_kdtree_->Nearest(points.col(i), bhm_index, bhm_distance);  // find the nearest bhm
            bhm_distance = std::sqrt(bhm_distance);                          // distance is squared
            if (bhm_distance > half_size) { continue; }                      // too far from the bhm

            const Key bhm_key = m_key_bhm_positions_[bhm_index].first;
            auto it = key_to_point_indices.insert({bhm_key, std::vector<long>()});  // insert the key if not exist
            if (it.second) { bhm_keys_set.push_back(bhm_key); }                     // if the key is new, add it to the set
            it.first->second.push_back(i);
        }

#pragma omp parallel for if (parallel) default(none) shared(bhm_keys_set, key_to_point_indices, points, faster, with_sigmoid, parallel, gradient)
        for (const Key &bhm_key: bhm_keys_set) {
            const auto &indices = key_to_point_indices[bhm_key];

            // copy the points of this key to a new matrix
            Positions points_of_key(Dim, static_cast<long>(indices.size()));
            for (long i = 0; i < points_of_key.cols(); ++i) { points_of_key.col(i) = points.col(indices[i]); }

            // predict
            Gradients gradients_of_key;
            std::shared_ptr<LocalBayesianHilbertMap> bhm = m_key_bhm_dict_[bhm_key];
            bhm->bhm.PredictGradient(points_of_key, faster, with_sigmoid, !parallel, gradients_of_key);

            // copy the results back to the original matrix
            for (long i = 0; i < points_of_key.cols(); ++i) { gradient.col(indices[i]) = gradients_of_key.col(i); }
        }
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::operator==(const AbstractSurfaceMapping &other) const {
        const auto *other_ptr = dynamic_cast<const BayesianHilbertSurfaceMapping *>(&other);
        if (other_ptr == nullptr) { return false; }
        if (m_setting_ == nullptr && other_ptr->m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other_ptr->m_setting_ == nullptr || *m_setting_ != *other_ptr->m_setting_)) { return false; }
        if (m_tree_ == nullptr && other_ptr->m_tree_ != nullptr) { return false; }
        if (m_tree_ != nullptr && (other_ptr->m_tree_ == nullptr || *m_tree_ != *other_ptr->m_tree_)) { return false; }
        // if (m_key_bhm_dict_ != other_ptr->m_key_bhm_dict_) { return false; }  // TODO: compare the content of the map
        if (m_hinged_points_ != other_ptr->m_hinged_points_) { return false; }
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Write(std::ostream &s) const {
        s << kFileHeader << std::endl  //
          << "# (feel free to add / change comments, but leave the first line as it is!)" << std::endl
          << "setting" << std::endl;
        // write setting
        if (!m_setting_->Write(s)) {
            ERL_WARN("Failed to write setting.");
            return false;
        }
        // write data
        // TODO: write data
        return s.good();
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Read(std::istream &s) {
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

        auto skip_line = [&s] {
            char c;
            do { c = static_cast<char>(s.get()); } while (s.good() && c != '\n');
        };

        // read data
        // TODO: read data

        ERL_WARN("Failed to read {}. Truncated file?", kClassName);
        return false;  // should not reach here
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GenerateHingedPoints() {
        const Dtype map_size = m_setting_->tree->resolution + 2 * m_setting_->bhm_overlap;
        const Dtype half_map_size = map_size / 2;
        const Eigen::Vector<int, Dim> grid_size = Eigen::Vector<int, Dim>::Constant(m_setting_->hinged_grid_size);
        const Eigen::Vector<Dtype, Dim> grid_half_size = Eigen::Vector<Dtype, Dim>::Constant(half_map_size);
        common::GridMapInfo<Dtype, Dim> grid_map_info(grid_size, -grid_half_size, grid_half_size);
        m_hinged_points_ = grid_map_info.GenerateMeterCoordinates(true);
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::BuildBhmKdtree() const {
        if (!m_bhm_kdtree_needs_update_ && m_bhm_kdtree_ != nullptr) { return; }
        Positions bhm_positions(Dim, m_key_bhm_positions_.size());
        long i = 0;
        for (const auto &[key, center]: m_key_bhm_positions_) { bhm_positions.col(i++) = center; }
        const_cast<BayesianHilbertSurfaceMapping *>(this)->m_bhm_kdtree_ = std::make_shared<Kdtree>(bhm_positions);
        const_cast<BayesianHilbertSurfaceMapping *>(this)->m_bhm_kdtree_needs_update_ = true;
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::PredictThread(
        const Dtype *points_ptr,
        const long start,
        const long end,
        const bool logodd,
        const bool faster,
        const bool compute_gradient,
        const bool gradient_with_sigmoid,
        const bool parallel,
        Dtype *prob_occupied_ptr,
        Dtype *gradient_ptr) const {

        ERL_DEBUG_ASSERT(points_ptr != nullptr, "points_ptr is nullptr.");
        ERL_DEBUG_ASSERT(prob_occupied_ptr != nullptr, "prob_occupied_ptr is nullptr.");
        ERL_DEBUG_ASSERT(!compute_gradient || gradient_ptr != nullptr, "gradient_ptr is nullptr.");
        ERL_DEBUG_ASSERT(start >= 0, "start is negative.");
        ERL_DEBUG_ASSERT(end > start, "end is not greater than start.");

        points_ptr += start * Dim;
        prob_occupied_ptr += start;
        if (compute_gradient) { gradient_ptr += start * Dim; }

        const long num_points = end - start;
        const long knn = m_setting_->test_knn;
        const uint32_t bhm_depth = m_setting_->bhm_depth;
        const Dtype half_size = 0.5f * m_tree_->GetNodeSize(bhm_depth) + m_setting_->bhm_test_margin;

        (void) parallel;
#pragma omp parallel for if (parallel) default(none) \
    shared(num_points, knn, half_size, logodd, faster, compute_gradient, gradient_with_sigmoid, points_ptr, prob_occupied_ptr, gradient_ptr)
        for (long i = 0; i < num_points; ++i) {

            // auto point = points.col(i);
            Eigen::Map<const Position> point(points_ptr + i * Dim, Dim);

            if (knn == 1) {
                long bhm_index = -1;
                Dtype bhm_distance = 0.0f;
                m_bhm_kdtree_->Nearest(point, bhm_index, bhm_distance);                         // find the nearest bhm
                bhm_distance = std::sqrt(bhm_distance);                                         // distance is squared
                if (bhm_distance > half_size || bhm_index < 0) {                                // too far from the bhm or no bhm
                    Key key;                                                                    // use the tree to predict the occupancy
                    if (!m_tree_->CoordToKeyChecked(point, key)) { continue; }                  // outside the map
                    const TreeNode *node = m_tree_->Search(key);                                // use the tree to predict the occupancy
                    if (node == nullptr) { continue; }                                          // unknown
                    prob_occupied_ptr[i] = logodd ? node->GetLogOdds() : node->GetOccupancy();  // get the occupancy from the tree
                    continue;
                }
                const Key &bhm_key = m_key_bhm_positions_[bhm_index].first;        // obtain the key of the bhm
                const BayesianHilbertMap &bhm = m_key_bhm_dict_.at(bhm_key)->bhm;  // obtain the bhm
                VectorX prob(1);
                Gradients grad(Dim, 1);
                bhm.Predict(point, logodd, faster, compute_gradient, gradient_with_sigmoid, false, prob, grad);  // predict
                prob_occupied_ptr[i] = prob[0];
                if (compute_gradient) {
                    Dtype *grad_ptr = gradient_ptr + i * Dim;
                    for (int dim = 0; dim < Dim; ++dim) { grad_ptr[dim] = grad.data()[dim]; }
                }
                continue;
            }

            Eigen::VectorXl bhm_indices(knn);
            VectorX bhm_distances(knn);
            bhm_indices.fill(-1);
            bhm_distances.fill(0);
            m_bhm_kdtree_->Knn(knn, point, bhm_indices, bhm_distances);  // find the neighboring bhms
            Dtype weight_sum = 0.0f;
            Dtype prob_sum = 0.0f;
            Gradient gradient_sum = Gradient::Zero();
            for (long j = 0; j < knn; ++j) {  // iterate over the neighbors
                const long &bhm_index = bhm_indices[j];
                if (bhm_index < 0) { break; }                                      // no more neighbors
                const Dtype bhm_distance = std::sqrt(bhm_distances[j]);            // distance is squared
                if (bhm_distance > half_size) { break; }                           // too far from the bhm
                const Key &bhm_key = m_key_bhm_positions_[bhm_index].first;        // obtain the key of the bhm
                const BayesianHilbertMap &bhm = m_key_bhm_dict_.at(bhm_key)->bhm;  // obtain the bhm
                VectorX prob(1);
                Gradients grad(Dim, 1);
                bhm.Predict(point, logodd, faster, compute_gradient, gradient_with_sigmoid, false, prob, grad);  // predict
                const Dtype weight = 1.0f / (bhm.GetMapBoundary().center - point).cwiseAbs().prod();
                weight_sum += weight;
                prob_sum += prob[0] * weight;
                if (compute_gradient) {
                    for (int dim = 0; dim < Dim; ++dim) { gradient_sum[dim] += grad.data()[dim] * weight; }
                }
            }
            if (weight_sum == 0.0f) {                                                       // no neighboring bhm
                Key key;                                                                    // use the tree to predict the occupancy
                if (!m_tree_->CoordToKeyChecked(point, key)) { continue; }                  // outside the map
                const TreeNode *node = m_tree_->Search(key);                                // use the tree to predict the occupancy
                if (node == nullptr) { continue; }                                          // unknown
                prob_occupied_ptr[i] = logodd ? node->GetLogOdds() : node->GetOccupancy();  // get the occupancy from the tree, gradient is not available
            } else {
                prob_occupied_ptr[i] = prob_sum / weight_sum;
                if (compute_gradient) {
                    Dtype *grad_ptr = gradient_ptr + i * Dim;
                    for (int dim = 0; dim < Dim; ++dim) { grad_ptr[dim] = gradient_sum[dim] / weight_sum; }
                }
            }
        }
    }

}  // namespace erl::sdf_mapping
