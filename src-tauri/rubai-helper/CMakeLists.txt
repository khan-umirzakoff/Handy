cmake_minimum_required(VERSION 3.14)

project(rubai_helper LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)

if(APPLE)
  set(GGML_METAL ON CACHE BOOL "" FORCE)
  set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
elseif(UNIX)
  # Linux Handy already supports Vulkan. Keep the Rubai worker in a separate
  # process, but give it the same GPU-capable backend instead of silently
  # falling back to CPU when the macOS-only Metal option is unavailable.
  set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

if(NOT DEFINED WHISPER_CPP_DIR)
  message(FATAL_ERROR "Pass -DWHISPER_CPP_DIR=/path/to/whisper.cpp")
endif()

add_subdirectory("${WHISPER_CPP_DIR}" whisper.cpp-build EXCLUDE_FROM_ALL)

add_executable(rubai-helper main.cpp)
target_link_libraries(rubai-helper PRIVATE whisper)
