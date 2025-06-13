#include "erl_common/test_helper.hpp"
#include "erl_geometry/marching_cubes.hpp"
#include "erl_gp_sdf/bayesian_hilbert_surface_mapping.hpp"
#include "erl_gp_sdf/gp_occ_surface_mapping.hpp"
#include "erl_gp_sdf/gp_sdf_mapping.hpp"

#include <boost/program_options.hpp>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/visualization/utility/DrawGeometry.h>

using namespace erl::common;

const std::filesystem::path kProjectRootDir = ERL_GP_SDF_ROOT_DIR;
const std::filesystem::path kConfigDir = kProjectRootDir / "config";
int g_argc = 0;
char **g_argv = nullptr;

template<typename Dtype>
void
ToMeshImpl() {

    using AbstractSurfaceMapping = erl::sdf_mapping::AbstractSurfaceMapping<Dtype, 3>;
    using GpOccSurfaceMapping = erl::sdf_mapping::GpOccSurfaceMapping<Dtype, 3>;
    using BayesianHilbertSurfaceMapping = erl::sdf_mapping::BayesianHilbertSurfaceMapping<Dtype, 3>;
    using GpSdfMapping = erl::sdf_mapping::GpSdfMapping<Dtype, 3>;

    ERL_INFO(
        "Surface mapping setting types: {}, {}",
        type_name<typename GpOccSurfaceMapping::Setting>(),
        type_name<typename BayesianHilbertSurfaceMapping::Setting>());
    ERL_INFO("SDF mapping setting types: {}", type_name<typename GpSdfMapping::Setting>());

    struct Options {
        std::string surface_mapping_type = type_name<GpOccSurfaceMapping>();
        std::string surface_mapping_setting_type =
            type_name<typename GpOccSurfaceMapping::Setting>();
        std::string sdf_mapping_bin = "sdf_mapping.bin";
        std::string output_mesh_file = "output_mesh.ply";
        double grid_resolution = 0.05;
    };

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
                fmt::format("Surface mapping setting type: {}, {}",
                            type_name<typename GpOccSurfaceMapping::Setting>(),
                            type_name<typename BayesianHilbertSurfaceMapping::Setting>()).c_str()
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
    auto surface_mapping =
        AbstractSurfaceMapping::Create(options.surface_mapping_type, surface_mapping_setting);
    auto sdf_mapping_setting = std::make_shared<typename GpSdfMapping::Setting>();
    GpSdfMapping sdf_mapping(sdf_mapping_setting, surface_mapping);
    using Serializer = Serialization<GpSdfMapping>;
    ASSERT_TRUE(Serializer::Read(options.sdf_mapping_bin, &sdf_mapping));

    using Aabb = erl::geometry::Aabb<Dtype, 3>;
    using Vector3 = Eigen::Vector3<Dtype>;
    using Matrix3X = Eigen::Matrix3X<Dtype>;

    Aabb map_boundary = surface_mapping->GetMapBoundary();

    GridMapInfo3D<Dtype> grid_map_info(
        map_boundary.min(),
        map_boundary.max(),
        Vector3(options.grid_resolution, options.grid_resolution, options.grid_resolution),
        Eigen::Vector3i::Zero());
    constexpr bool row_major = false;
    Matrix3X positions = grid_map_info.GenerateMeterCoordinates(row_major);

    typename GpSdfMapping::Distances distances(positions.cols());
    typename GpSdfMapping::Gradients gradients(3, positions.cols());
    typename GpSdfMapping::Variances variances(4, positions.cols());
    typename GpSdfMapping::Covariances covariances(6, positions.cols());
    ASSERT_TRUE(sdf_mapping.Test(positions, distances, gradients, variances, covariances));
    ERL_INFO("SDF mapping prediction done, {} points.", positions.cols());

    auto extracted_mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    constexpr bool parallel = true;
    erl::geometry::MarchingCubes::Run(
        grid_map_info.Min().template cast<double>(),
        grid_map_info.Resolution().template cast<double>(),
        grid_map_info.Shape(),
        distances.template cast<double>(),
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

TEST(GpSdfMapping, ToMeshD) { ToMeshImpl<double>(); }

TEST(GpSdfMapping, ToMeshF) { ToMeshImpl<float>(); }

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
