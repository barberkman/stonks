#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

#include "stonks/core/log.h"
#include "stonks/core/types.h"
#include "stonks/python/icontext.h"

namespace py = pybind11;

namespace {

namespace core = stonks::core;
namespace stonks_py = stonks::python;

// Python-facing result of Context.history(n): all printing symbols' bars gathered
// into one set of equal-length columns (a long frame). The numpy arrays are
// freshly-allocated copies owned by Python, so there's no lifetime coupling to
// the feed. Build a DataFrame with pd.DataFrame({...}).groupby("symbol").
struct PyMarketWindow
{
    py::list symbol;                                     // one entry per row
    py::array timestamp, open, high, low, close, volume; // int64 / float64
    py::ssize_t length{ 0 };
};

PyMarketWindow gather(const core::MarketWindow& win)
{
    py::ssize_t total = 0;
    for (const auto& s : win.series) { total += static_cast<py::ssize_t>(s.bars.size()); }

    // GIL is held here (entered in PythonStrategy::invoke). Aggregate counts only
    // — never log per element, and never touch py::object values.
    py::array_t<std::int64_t> timestamp(total);
    py::array_t<double> open(total), high(total), low(total), close(total), volume(total);
    auto* ts = timestamp.mutable_data();
    auto* op = open.mutable_data();
    auto* hi = high.mutable_data();
    auto* lo = low.mutable_data();
    auto* cl = close.mutable_data();
    auto* vo = volume.mutable_data();

    py::list symbol;
    py::ssize_t k = 0;
    for (const auto& s : win.series) {
        const py::str ticker{ std::string{ s.symbol } };
        const std::size_t m = s.bars.size();
        for (std::size_t j = 0; j < m; ++j) {
            symbol.append(ticker);
            ts[k] = s.bars.timestamp[j];
            op[k] = s.bars.open[j];
            hi[k] = s.bars.high[j];
            lo[k] = s.bars.low[j];
            cl[k] = s.bars.close[j];
            vo[k] = s.bars.volume[j];
            ++k;
        }
    }
    return PyMarketWindow{ std::move(symbol), timestamp, open, high, low, close, volume, total };
}

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

    // Returned by Context.history(): a long frame over every symbol that printed
    // this tick. `symbol` is a per-row column; build with
    // pd.DataFrame({ "symbol": w.symbol, "close": w.close, ... }).groupby("symbol").
    py::class_<PyMarketWindow>(m, "MarketWindow")
        .def_readonly("symbol", &PyMarketWindow::symbol)
        .def_readonly("timestamp", &PyMarketWindow::timestamp)
        .def_readonly("open", &PyMarketWindow::open)
        .def_readonly("high", &PyMarketWindow::high)
        .def_readonly("low", &PyMarketWindow::low)
        .def_readonly("close", &PyMarketWindow::close)
        .def_readonly("volume", &PyMarketWindow::volume)
        .def("__len__", [](const PyMarketWindow& w) { return w.length; });

    // Bind IContext as `Context` so user-facing names match the C++ Context API.
    // No constructor exposed: instances only originate from the engine via
    // PythonStrategy's adapter (cast as a non-owning reference).
    py::class_<stonks_py::IContext>(m, "Context")
        .def("now", &stonks_py::IContext::now)
        .def("cash", &stonks_py::IContext::cash)
        .def("equity", &stonks_py::IContext::equity)
        .def("history",
             [](const stonks_py::IContext& self, int count) {
                 return gather(self.history(count));
             },
             py::arg("count"),
             "This tick's window: every symbol that printed, each with its last "
             "`count` bars, as one combined frame.")
        .def("place_market_order",
             [](stonks_py::IContext& self,
                core::Symbol symbol,
                core::OrderSide side,
                core::Quantity quantity,
                core::TimeInForce time_in_force) {
                 const bool ok = self.place_market_order(core::MarketOrderParams{
                     std::move(symbol),
                     side,
                     quantity,
                     time_in_force,
                 });
                 return ok;
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
                 const bool ok = self.place_limit_order(core::LimitOrderParams{
                     std::move(symbol),
                     side,
                     quantity,
                     price,
                     time_in_force,
                 });
                 return ok;
             },
             py::arg("symbol"),
             py::arg("side"),
             py::arg("quantity"),
             py::arg("price"),
             py::arg("time_in_force") = core::TimeInForce::GTC);
}
