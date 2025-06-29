#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "OceanSim::CppCore" for configuration ""
set_property(TARGET OceanSim::CppCore APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(OceanSim::CppCore PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/liboceansim_core.a"
  )

list(APPEND _cmake_import_check_targets OceanSim::CppCore )
list(APPEND _cmake_import_check_files_for_OceanSim::CppCore "${_IMPORT_PREFIX}/lib/liboceansim_core.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
