add_executable(c4we
  problem.cc
  statistics.cc
  inversion.cc)

target_link_libraries(c4we
  ${ARMADILLO_LIBRARIES}
  ${YAML_CPP_LIBRARIES})

add_library(python SHARED
  statistics.cc
  pybind.cc)

target_link_libraries(python
  ${ARMADILLO_LIBRARIES})
