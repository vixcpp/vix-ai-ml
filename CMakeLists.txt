# ML library minimale
add_library(vix_ai_ml
  src/Dataset.cpp
  src/Model.cpp
  src/Regression.cpp
  src/Clustering.cpp
)

add_library(Vix::AI::ml ALIAS vix_ai_ml)

option(VIX_AI_ML_BUILD_TESTS "Build vix_ai_ml tests" ON)
if (VIX_AI_ML_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# dépend de vix_ai_core
target_link_libraries(vix_ai_ml
  PUBLIC Vix::AI::core
)

target_include_directories(vix_ai_ml
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(vix_ai_ml PUBLIC cxx_std_20)

# Install (optionnel)
include(GNUInstallDirs)
install(TARGETS vix_ai_ml
  EXPORT vix_ai_mlTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT vix_ai_mlTargets
  FILE vix_ai_mlTargets.cmake
  NAMESPACE Vix::AI::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vix_ai_ml
)