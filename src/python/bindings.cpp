#include <chrono>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stonks/core/types.h"
#include "stonks/python/icontext.h"

namespace py = pybind11;

namespace {

namespace core = stonks::core;
namespace stonks_py = stonks::python;

} // namespace

PYBIND11_MODULE(_core, m)
{
    m.doc() = "stonks core bindings — value types and the Context interface for Python strategies.";

    py::enum_<core::OrderSide>(m, "OrderSide")
        .value("Buy", core::OrderSide::Buy)
        .value("Sell", core::OrderSide::Sell);

    py::enum_<core::OrderType>(m, "OrderType")
        .value("Market", core::OrderType::Market)
        .value("Limit", core::OrderType::Limit);

    py::enum_<core::TimeInForce>(m, "TimeInForce")
        .value("GTC", core::TimeInForce::GTC);

    py::class_<core::Timestamp>(m, "Timestamp")
        .def(py::init<>())
        .def_static("from_millis", &core::Timestamp::from_millis, py::arg("millis"))
        .def("to_millis", [](const core::Timestamp& t) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                t.value.time_since_epoch()).count();
        })
        .def("__repr__", [](const core::Timestamp& t) {
            std::ostringstream os;
            os << t;
            return os.str();
        })
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
        .def(py::self == py::self)
        .def(py::self != py::self);

    py::class_<core::KLine>(m, "KLine")
        .def_readonly("timestamp", &core::KLine::timestamp)
        .def_readonly("symbol", &core::KLine::symbol)
        .def_readonly("open", &core::KLine::open)
        .def_readonly("high", &core::KLine::high)
        .def_readonly("low", &core::KLine::low)
        .def_readonly("close", &core::KLine::close)
        .def_readonly("volume", &core::KLine::volume)
        .def("__repr__", [](const core::KLine& k) {
            std::ostringstream os;
            os << "KLine(" << k.timestamp << ", " << k.symbol
               << ", o=" << k.open << ", h=" << k.high
               << ", l=" << k.low << ", c=" << k.close
               << ", v=" << k.volume << ")";
            return os.str();
        });

    // Bind IContext as `Context` so user-facing names match the C++ Context API.
    // No constructor exposed: instances only originate from the engine via
    // PythonStrategy's adapter (cast as a non-owning reference).
    py::class_<stonks_py::IContext>(m, "Context")
        .def("now", &stonks_py::IContext::now)
        .def("cash", &stonks_py::IContext::cash)
        .def("equity", &stonks_py::IContext::equity)
        .def("klines",
             [](stonks_py::IContext& self, py::args args, py::kwargs kwargs) -> py::object {
                 if (args.size() == 1 && kwargs.empty() && py::isinstance<py::int_>(args[0])) {
                     return py::cast(self.klines_count(args[0].cast<int>()));
                 }
                 if (args.size() >= 1 && py::isinstance<core::Timestamp>(args[0])) {
                     const auto start = args[0].cast<core::Timestamp>();
                     std::optional<core::Timestamp> end = std::nullopt;
                     if (args.size() == 2) {
                         end = args[1].cast<core::Timestamp>();
                     } else if (kwargs.contains("end")) {
                         const auto end_obj = kwargs["end"];
                         if (!end_obj.is_none()) { end = end_obj.cast<core::Timestamp>(); }
                     }
                     return py::cast(self.klines_range(start, end));
                 }
                 throw py::type_error(
                     "klines() expects (count: int) or (start: Timestamp, end: Optional[Timestamp])");
             },
             "klines(count: int) or klines(start: Timestamp, end: Optional[Timestamp] = None)")
        .def("place_market_order",
             [](stonks_py::IContext& self,
                core::Symbol symbol,
                core::OrderSide side,
                core::Quantity quantity,
                core::TimeInForce time_in_force) {
                 return self.place_market_order(core::MarketOrderParams{
                     std::move(symbol),
                     side,
                     quantity,
                     time_in_force,
                 });
             },
             py::arg("symbol"),
             py::arg("side"),
             py::arg("quantity"),
             py::arg("time_in_force") = core::TimeInForce::GTC)
        .def("place_limit_order",
             [](stonks_py::IContext& self,
                core::Symbol symbol,
                core::OrderSide side,
                core::Quantity quantity,
                core::Price price,
                core::TimeInForce time_in_force) {
                 return self.place_limit_order(core::LimitOrderParams{
                     std::move(symbol),
                     side,
                     quantity,
                     price,
                     time_in_force,
                 });
             },
             py::arg("symbol"),
             py::arg("side"),
             py::arg("quantity"),
             py::arg("price"),
             py::arg("time_in_force") = core::TimeInForce::GTC);
}
