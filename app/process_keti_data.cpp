

#include "erl_sdf_mapping/gp_occ_surface_mapping.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping.hpp"

#include <boost/program_options.hpp>

struct Options {
    std::string data_dir;
    std::string surface_mapping_config_file = "surface_mapping.yaml";
    std::string sdf_mapping_config_file = "sdf_mapping.yaml";
    std::string sdf_mapping_bin_file;
};

using Dtype = float;
using SurfaceMapping = erl::sdf_mapping::GpOccSurfaceMapping<Dtype, 3>;
using SdfMapping = erl::sdf_mapping::GpSdfMapping<Dtype, 3, SurfaceMapping>;

std::shared_ptr<Options> parse_options(
    int argc,
    char** argv
){
    auto options = std::make_shared<Options>();

    namespace po = boost::program_options;
    po::options_description desc;
    // clang-format off
    desc.add_options()
        ("help", "produce help message")
        (
            "data-dir",
            po::value<std::string>(&options->data_dir),
            "directory containing the dataset"
        )
        (
            "surface-mapping-config-file",
            po::value<std::string>(&options->surface_mapping_config_file)->default_value(options->surface_mapping_config_file)->value_name("file"),
            "surface mapping config file"
        )
        (
            "sdf-mapping-config-file",
            po::value<std::string>(&options->sdf_mapping_config_file)->default_value(options->sdf_mapping_config_file)->value_name("file"),
            "SDF mapping config file"
        )
        (
            "sdf-mapping-bin-file",
            po::value<std::string>(&options->sdf_mapping_bin_file)->default_value(options->sdf_mapping_bin_file)->value_name("file"),
            "SDF mapping bin file"
        );

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        if (vm.count("help")) {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl << desc << std::endl;
            return nullptr;
        }
        po::notify(vm);

        return options;
}

std::shared_ptr<SdfMapping> construct_sdf_mapping(
    std::shared_ptr<Options> options
){
    // load setting
    const auto surface_mapping_setting = std::make_shared<typename SurfaceMapping::Setting>();
    ERL_ASSERTM(
        surface_mapping_setting->FromYamlFile(options->surface_mapping_config_file),
        "Failed to load surface_mapping_config_file: {}",
        options->surface_mapping_config_file);
    const auto sdf_mapping_setting = std::make_shared<typename SdfMapping::Setting>();
    ERL_ASSERTM(
        sdf_mapping_setting->FromYamlFile(options->sdf_mapping_config_file),
        "Failed to load sdf_mapping_config_file: {}",
        options->sdf_mapping_config_file);

    // prepare the mapping
    const auto surface_mapping = std::make_shared<SurfaceMapping>(surface_mapping_setting);  // create surface mapping
    // SdfMapping sdf_mapping(sdf_mapping_setting, surface_mapping);
    auto sdf_mapping = std::make_shared<SdfMapping>(
        sdf_mapping_setting,
        surface_mapping
    );

    return sdf_mapping;
}

int main(int argc, char** argv) {
    const auto options = parse_options(argc, argv);
    if (options == nullptr) {
        return -1;
    }
    auto sdf_mapping = construct_sdf_mapping(options);
}