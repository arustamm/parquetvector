cmake_minimum_required(VERSION 3.10)
project(arrow_pybind11_example)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find Arrow
find_package(Arrow REQUIRED)

# Find Pyarrow includes
execute_process(COMMAND python3 -c "import pyarrow; print(pyarrow.get_include())" 
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYARROW_INCLUDE_DIR)
execute_process(COMMAND python3 -c "import pyarrow; print(pyarrow.get_library_dirs())" 
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYARROW_LIBRARY_DIR)

# Add include directories
include_directories(
  ${pybind11_INCLUDE_DIRS}
  ${ARROW_INCLUDE_DIRS}
  ${PYARROW_INCLUDE_DIR} 
)


# Add the source file
pybind11_add_module(arrow_test pybind_test.cpp)

# Link against pybind11 and Arrow
target_link_libraries(arrow_test PRIVATE
  pybind11::pybind11
  Arrow::arrow_static
  /usr/local/lib/python3.10/dist-packages/pyarrow/libarrow_python.so
)
