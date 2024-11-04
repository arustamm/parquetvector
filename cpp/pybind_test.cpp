#include <arrow/python/pyarrow.h>
#include <arrow/table.h>
#include <arrow/api.h>
#include <arrow/result.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Function to create an Arrow Table in C++
std::shared_ptr<arrow::Table> create_arrow_table() {
  // Create some sample data
  arrow::Int64Builder int_builder;
  int_builder.AppendValues({1, 2, 3, 4, 5});
  std::shared_ptr<arrow::Array> int_array;
  int_builder.Finish(&int_array);

  arrow::StringBuilder str_builder;
  str_builder.AppendValues({"a", "b", "c", "d", "e"});
  std::shared_ptr<arrow::Array> str_array;
  str_builder.Finish(&str_array);

  // Create a schema
  auto schema = arrow::schema({
      arrow::field("int_column", arrow::int64()),
      arrow::field("str_column", arrow::utf8())
  });

  // Create the table
  return arrow::Table::Make(schema, {int_array, str_array});
}

py::object table_to_pyarrow(const std::shared_ptr<arrow::Table>& table) {
  // Acquire the GIL (Global Interpreter Lock) 
  py::gil_scoped_acquire acquire; 

  // Convert the Arrow C++ Table to a PyArrow Table object
  PyObject* table_obj = arrow::py::wrap_table(table);

  // Create a pybind11::object from the PyObject*
  return py::reinterpret_steal<py::object>(table_obj);
}

PYBIND11_MODULE(pybind11_test, m) {
  m.doc() = "Example module for converting Arrow Table to PyArrow Table";
  m.def("table_to_pyarrow", &table_to_pyarrow, "Convert an Arrow Table to a PyArrow Table");
}