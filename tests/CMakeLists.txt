# Petit test sans framework externe (juste CTest)
add_executable(vix_ai_ml_smoke_test ml_smoke_test.cpp)
target_link_libraries(vix_ai_ml_smoke_test PRIVATE Vix::AI::ml)
target_compile_features(vix_ai_ml_smoke_test PRIVATE cxx_std_20)

add_test(NAME vix_ai_ml_smoke_test COMMAND vix_ai_ml_smoke_test)
