cmake_minimum_required(VERSION 3.10)
project(arrow_pybind11_example)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find Arrow
find_package(Arrow REQUIRED)

# Find Pyarrow includes
execute_process(COMMAND python3 -c "import pyarrow; print(pyarrow.get_include())" 
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYARROW_INCLUDE_DIR)
execute_process(COMMAND python3 -c "import pyarrow; print(pyarrow.get_library_dirs()[0])" 
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYARROW_LIBRARY_DIR)

# Add include directories
include_directories(
  ${pybind11_INCLUDE_DIRS}
  ${ARROW_INCLUDE_DIRS}
  ${PYARROW_INCLUDE_DIR} 
)


# Add the source file
pybind11_add_module(pybind11_test pybind_test.cpp)

# Link against pybind11 and Arrow
target_link_libraries(pybind11_test PRIVATE
  pybind11::pybind11
  Arrow::arrow_shared
  ${PYARROW_LIBRARY_DIR}/libarrow_python.so
)
