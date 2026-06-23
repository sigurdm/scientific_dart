# Install script for directory: /usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy_build/libhwy.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/aligned_allocator.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/base.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/cache_control.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/detect_compiler_arch.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/detect_targets.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/foreach_target.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/highway.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/highway_export.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/nanobenchmark.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/profiler.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/ops" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/ops/arm_neon-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/ops" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/ops/arm_sve-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/ops" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/ops/emu128-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/ops" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/ops/generic_ops-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/ops" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/ops/ppc_vsx-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/ops" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/ops/rvv-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/ops" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/ops/scalar-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/ops" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/ops/set_macros-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/ops" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/ops/shared-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/ops" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/ops/wasm_128-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/ops" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/ops/tuple-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/ops" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/ops/x86_128-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/ops" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/ops/x86_256-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/ops" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/ops/x86_512-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/per_target.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/print-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/print.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/robust_statistics.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/targets.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/timer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/timer-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy_build/libhwy_contrib.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/dot" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/dot/dot-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/image" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/image/image.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/math" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/math/math-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/matvec" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/matvec/matvec-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/sort" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/sort/order.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/sort" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/sort/shared-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/sort" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/sort/sorting_networks-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/sort" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/sort/traits-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/sort" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/sort/traits128-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/sort" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/sort/vqsort-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/sort" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/sort/vqsort.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/thread_pool" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/thread_pool/futex.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/thread_pool" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/thread_pool/thread_pool.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/algo" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/algo/copy-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/algo" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/algo/find-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/algo" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/algo/transform-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hwy/contrib/unroller" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy/contrib/unroller/unroller-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy_build/libhwy.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy_build/libhwy-contrib.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy_build/hwy-config-version.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/hwy/hwy-config.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/hwy/hwy-config.cmake"
         "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy_build/CMakeFiles/Export/748e3176398835d20b4ae2e3610f0270/hwy-config.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/hwy/hwy-config-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/hwy/hwy-config.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy_build/CMakeFiles/Export/748e3176398835d20b4ae2e3610f0270/hwy-config.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/hwy" TYPE FILE FILES "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy_build/CMakeFiles/Export/748e3176398835d20b4ae2e3610f0270/hwy-config-release.cmake")
  endif()
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy_build/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/third_party/highway/hwy_build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
