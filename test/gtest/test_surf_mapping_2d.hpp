#pragma once
#include "test_mapping_2d.hpp"

template<typename Dtype>
struct Options : public erl::common::Yamlable<Options<Dtype>, OptionsForTestMapping2D<Dtype>> {

    using Super = OptionsForTestMapping2D<Dtype>;

    std::string surf_map_config_file;

    ERL_REFLECT_SCHEMA(Options, ERL_REFLECT_MEMBER(Options, surf_map_config_file));

    bool
    PostDeserialization() override {
        if (!Super::PostDeserialization()) { return false; }
        ERL_ASSERTM(
            !surf_map_config_file.empty(),
            "Please provide the surface mapping config file via --surf_map_config_file");
        return true;
    }
};

template<typename Dtype, typename SurfMapType>
struct TestSurfMapping2D : public TestMapping2D<Dtype, SurfMapType> {
    using Super = TestMapping2D<Dtype, SurfMapType>;
    using SurfMap = SurfMapType;
    using SurfMapSetting = typename SurfMap::Setting;
    using OptionType = Options<Dtype>;

    using Super::fps_data;
    using Super::mapping;
    using Super::max_wp_idx;
    using Super::quadtree;
    using Super::rotation;
    using Super::scaling;
    using Super::translation;

    std::shared_ptr<SurfMapSetting> surf_map_setting = nullptr;
    std::shared_ptr<SurfMap> surf_map = nullptr;

protected:
    std::shared_ptr<OptionType> options = nullptr;

public:
    TestSurfMapping2D(
        const int argc,
        char *argv[],
        std::shared_ptr<OptionType> options = std::make_shared<OptionType>())
        : Super(argc, argv, options), options(options) {}

protected:
    void
    Init() override {
        surf_map_setting = std::make_shared<SurfMapSetting>();
        ERL_ASSERTM(
            surf_map_setting->FromYamlFile(options->surf_map_config_file),
            "Failed to load surf_map_config_file: {}",
            options->surf_map_config_file);
        surf_map_setting->AsYamlFile(options->output_dir / "surf_map.yaml");

        // create mappings
        surf_map = std::make_shared<SurfMap>(surf_map_setting);
        quadtree = surf_map->GetTree();
        mapping = surf_map;
        scaling = surf_map_setting->scaling;

        // base init
        Super::Init();

        // other
        fps_data.setConstant(3, Super::GetNumOfFrames(), 0.0);
    }

    bool
    UpdateMap() override {
        if (Super::mapping_uses_points) {
            // are_points: true, are_local: false
            return surf_map->Update(rotation, translation, Super::train_world_points, true, false);
        }

        // are_points: false, are_local: true
        return surf_map->Update(rotation, translation, Super::train_ranges, false, true);
    }

    void
    ShowFinalResults() override {}

    void
    Interactive() override {}

    std::string
    GetBinFileName() override {
        std::string bin_file = fmt::format("surf_map_2d_{}.bin", type_name<Dtype>());
        bin_file = options->output_dir / bin_file;
        return bin_file;
    }

    void
    TestIo() override {
        SurfMap surf_map_read(std::make_shared<SurfMapSetting>());
        Super::TestIo(surf_map_read);
    }
};
