#include "erl_common/test_helper.hpp"
#include "erl_geometry/marching_cubes.hpp"
#include "erl_sdf_mapping/bayesian_hilbert_surface_mapping.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping.hpp"

#include <boost/program_options.hpp>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/visualization/utility/DrawGeometry.h>

using namespace erl::common;
using namespace erl::geometry;
using namespace erl::sdf_mapping;
using GpOccSurfaceSdfMapping3Dd = GpSdfMapping<double, 3, GpOccSurfaceMapping3Dd>;
using GpOccSurfaceSdfMapping3Df = GpSdfMapping<float, 3, GpOccSurfaceMapping3Df>;
using BayesianHilbertSdfMapping3Df = GpSdfMapping<float, 3, BayesianHilbertSurfaceMapping3Df>;
using BayesianHilbertSdfMapping3Dd = GpSdfMapping<double, 3, BayesianHilbertSurfaceMapping3Dd>;

const std::filesystem::path kProjectRootDir = ERL_SDF_MAPPING_ROOT_DIR;
const std::filesystem::path kConfigDir = kProjectRootDir / "config";
int g_argc = 0;
char **g_argv = nullptr;

struct Options {
    std::string surface_mapping_type = type_name<GpOccSurfaceMapping3Df>();
    std::string surface_mapping_setting_type = type_name<GpOccSurfaceMapping3Df::Setting>();
    std::string sdf_mapping_type = type_name<GpOccSurfaceSdfMapping3Df>();
    std::string sdf_mapping_setting_type = type_name<GpSdfMappingSetting3Df>();
    std::string sdf_mapping_bin = "sdf_mapping.bin";
    std::string output_mesh_file = "output_mesh.ply";
    double grid_resolution = 0.05;
};

TEST(GpSdfMapping, ToMesh) {

    ERL_INFO(
        "Surface mapping types: {}, {}, {}, {}",
        type_name<GpOccSurfaceMapping3Dd>(),
        type_name<GpOccSurfaceMapping3Df>(),
        type_name<BayesianHilbertSurfaceMapping3Dd>(),
        type_name<BayesianHilbertSurfaceMapping3Df>());
    ERL_INFO(
        "SDF mapping types: {}, {}, {}, {}",
        type_name<GpOccSurfaceSdfMapping3Dd>(),
        type_name<GpOccSurfaceSdfMapping3Df>(),
        type_name<BayesianHilbertSdfMapping3Dd>(),
        type_name<BayesianHilbertSdfMapping3Df>());
    ERL_INFO(
        "Surface mapping setting types: {}, {}, {}, {}",
        type_name<GpOccSurfaceMapping3Dd::Setting>(),
        type_name<GpOccSurfaceMapping3Df::Setting>(),
        type_name<BayesianHilbertSurfaceMapping3Dd::Setting>(),
        type_name<BayesianHilbertSurfaceMapping3Df::Setting>());
    ERL_INFO(
        "SDF mapping setting types: {}, {}",
        type_name<GpSdfMappingSetting3Dd>(),
        type_name<GpSdfMappingSetting3Df>());

    Options options;
    bool options_parsed = false;
    namespace po = boost::program_options;
    po::options_description desc;
    try {
        // clang-format off
        desc.add_options()
            ("help", "produce help message")
            (
                "surface-mapping-type",
                po::value<std::string>(&options.surface_mapping_type)->default_value(options.surface_mapping_type),
                "Surface mapping type"
            )
            (
                "surface-mapping-setting-type",
                po::value<std::string>(&options.surface_mapping_setting_type)->default_value(options.surface_mapping_setting_type),
                fmt::format("Surface mapping setting type: {}, {}, {}, {}",
                            type_name<GpOccSurfaceMapping3Dd::Setting>(),
                            type_name<GpOccSurfaceMapping3Df::Setting>(),
                            type_name<BayesianHilbertSurfaceMapping3Dd::Setting>(),
                            type_name<BayesianHilbertSurfaceMapping3Df::Setting>()).c_str()
            )
            (
                "sdf-mapping-type",
                po::value<std::string>(&options.sdf_mapping_type)->default_value(options.sdf_mapping_type),
                fmt::format("SDF mapping type: {}, {}, {}, {}",
                        type_name<GpOccSurfaceSdfMapping3Dd>(),
                        type_name<GpOccSurfaceSdfMapping3Df>(),
                        type_name<BayesianHilbertSdfMapping3Dd>(),
                        type_name<BayesianHilbertSdfMapping3Df>()).c_str()
            )
            (
                "sdf-mapping-setting-type",
                po::value<std::string>(&options.sdf_mapping_setting_type)->default_value(options.sdf_mapping_setting_type),
                fmt::format("SDF mapping setting type: {}, {}",
                            type_name<GpSdfMappingSetting3Dd>(),
                            type_name<GpSdfMappingSetting3Df>()).c_str()
            )
            (
                "sdf-mapping-bin",
                po::value<std::string>(&options.sdf_mapping_bin)->default_value(options.sdf_mapping_bin),
                "SDF mapping binary file"
            )
            (
                "output-mesh-file",
                po::value<std::string>(&options.output_mesh_file)->default_value(options.output_mesh_file),
                "Output mesh file"
            )
            (
                "grid-resolution",
                po::value<double>(&options.grid_resolution)->default_value(options.grid_resolution),
                "Grid resolution for mesh extraction (in meters)"
            );
        // clang-format on

        po::variables_map vm;
        po::store(po::command_line_parser(g_argc, g_argv).options(desc).run(), vm);
        if (vm.count("help")) {
            std::cout << "Usage: " << g_argv[0] << " [options]" << std::endl << desc << std::endl;
            return;
        }
        po::notify(vm);
        options_parsed = true;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cout << "Usage: " << g_argv[0] << " [options]" << std::endl << desc << std::endl;
    }
    ASSERT_TRUE(options_parsed);

    auto &setting_factory = YamlableBase::Factory::GetInstance();

    auto surface_mapping_setting = setting_factory.Create(options.surface_mapping_setting_type);
    auto sdf_mapping_setting = setting_factory.Create(options.sdf_mapping_setting_type);

    auto sdf_mapping = AbstractGpSdfMapping::Create(
        options.sdf_mapping_type,
        surface_mapping_setting,
        sdf_mapping_setting);
    using Serializer = Serialization<AbstractGpSdfMapping>;
    ASSERT_TRUE(Serializer::Read(options.sdf_mapping_bin, sdf_mapping));
    auto abstract_surface_mapping = sdf_mapping->GetAbstractSurfaceMapping();

    Eigen::AlignedBox3d map_boundary;
    if (options.surface_mapping_type == type_name<BayesianHilbertSurfaceMapping3Dd>()) {
        auto surface_mapping =
            std::dynamic_pointer_cast<BayesianHilbertSurfaceMapping3Dd>(abstract_surface_mapping);
        ASSERT_TRUE(surface_mapping != nullptr);
        map_boundary = static_cast<Eigen::AlignedBox<double, 3>>(surface_mapping->GetMapBoundary());
    } else if (options.surface_mapping_type == type_name<BayesianHilbertSurfaceMapping3Df>()) {
        auto surface_mapping =
            std::dynamic_pointer_cast<BayesianHilbertSurfaceMapping3Df>(abstract_surface_mapping);
        ASSERT_TRUE(surface_mapping != nullptr);
        map_boundary = surface_mapping->GetMapBoundary().cast<double>();
    } else if (options.surface_mapping_type == type_name<GpOccSurfaceMapping3Dd>()) {
        auto surface_mapping =
            std::dynamic_pointer_cast<GpOccSurfaceMapping3Dd>(abstract_surface_mapping);
        ASSERT_TRUE(surface_mapping != nullptr);
        map_boundary = static_cast<Eigen::AlignedBox<double, 3>>(surface_mapping->GetMapBoundary());
    } else if (options.surface_mapping_type == type_name<GpOccSurfaceMapping3Df>()) {
        auto surface_mapping =
            std::dynamic_pointer_cast<GpOccSurfaceMapping3Df>(abstract_surface_mapping);
        ASSERT_TRUE(surface_mapping != nullptr);
        map_boundary = surface_mapping->GetMapBoundary().cast<double>();
    } else {
        ERL_FATAL("Unsupported surface mapping type: {}", options.surface_mapping_type);
    }

    GridMapInfo3Dd grid_map_info(
        map_boundary.min(),
        map_boundary.max(),
        Eigen::Vector3d(options.grid_resolution, options.grid_resolution, options.grid_resolution),
        Eigen::Vector3i::Zero());
    constexpr bool row_major = false;
    Eigen::Matrix3Xd positions = grid_map_info.GenerateMeterCoordinates(row_major);

    Eigen::VectorXd distances(positions.cols());
    Eigen::MatrixXd gradients(3, positions.cols());
    Eigen::MatrixXd variances(4, positions.cols());
    Eigen::MatrixXd covariances(6, positions.cols());
    ASSERT_TRUE(sdf_mapping->Predict(positions, distances, gradients, variances, covariances));
    ERL_INFO("SDF mapping prediction done, {} points.", positions.cols());

    auto extracted_mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    constexpr bool parallel = true;
    MarchingCubes::Run(
        grid_map_info.Min(),
        grid_map_info.Resolution(),
        grid_map_info.Shape(),
        distances,
        row_major,
        extracted_mesh->vertices_,
        extracted_mesh->triangles_,
        extracted_mesh->triangle_normals_,
        parallel);
    ASSERT_TRUE(!extracted_mesh->vertices_.empty());
    ASSERT_TRUE(!extracted_mesh->triangles_.empty());
    open3d::io::WriteTriangleMesh(options.output_mesh_file, *extracted_mesh, true);
    open3d::visualization::DrawGeometries({extracted_mesh});
}

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
