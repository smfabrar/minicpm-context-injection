cmake_minimum_required(VERSION 3.20)
project(minicpm_context_injection_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT RUNTIME OR NOT RUNTIME_BUILD)
  message(FATAL_ERROR "Pass -DRUNTIME=<source> and -DRUNTIME_BUILD=<CUDA build>")
endif()

add_executable(native-session native_session.cpp)
target_include_directories(native-session PRIVATE
  "${RUNTIME}/tools/omni"
  "${RUNTIME}/common"
  "${RUNTIME}/include"
  "${RUNTIME}/ggml/include"
  "${RUNTIME}/vendor"
)

foreach(lib omni llama-common llama ggml ggml-base)
  find_library(LIB_${lib}
    NAMES ${lib}
    PATHS "${RUNTIME_BUILD}/bin"
    NO_DEFAULT_PATH
    REQUIRED
  )
  target_link_libraries(native-session PRIVATE ${LIB_${lib}})
endforeach()

# Let the bridge find the exact libraries built from the pinned runtime.
set_target_properties(native-session PROPERTIES BUILD_RPATH "${RUNTIME_BUILD}/bin")
