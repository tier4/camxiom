// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "camxiom/camera_info_yaml.hpp"

#include "camxiom/model.hpp"
#include "compat/internal.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace camxiom
{
namespace
{

// Layout mirrors the intrinsic camera calibrator's hand-rolled emitter (which
// itself mirrors the reference calibrator's output): fixed-point numbers and
// column-aligned matrix blocks, one source row per output line.

constexpr int kMatrixPrecision = 6;
constexpr int kDistortionPrecision = 12;

std::string formatDouble(
  const double value, const int fixed_digits, CameraInfoYamlPrecision precision
)
{
  std::ostringstream stream;
  // A machine-readable serializer must not follow the ambient global locale:
  // a comma decimal point (e.g. de_DE) would split every number into two
  // flow-sequence entries for any real YAML parser reading the file back.
  stream.imbue(std::locale::classic());
  if (precision == CameraInfoYamlPrecision::ROUND_TRIP)
  {
    if (std::isnan(value)) return ".nan";
    if (std::isinf(value)) return std::signbit(value) ? "-.inf" : ".inf";
    // Scientific notation gives every value an explicit decimal point, so
    // YAML readers retain floating-point type and the sign of negative zero.
    stream << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10 - 1)
           << value;
  }
  else
  {
    stream << std::fixed << std::setprecision(fixed_digits) << value;
  }
  return stream.str();
}

// Emit `text` for a YAML scalar position. Identifier-like strings pass
// through plain (matching the reference emitter byte-for-byte); anything
// else — empty, YAML-structural characters like `: ` or `#`, quotes,
// control characters — is double-quoted with `\`, `"` and newlines escaped
// so a hostile camera_name cannot corrupt the document structure.
std::string yamlScalar(const std::string &text)
{
  bool plain_safe = !text.empty();
  for (const char c : text)
  {
    const bool identifier_like = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                                 (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.' ||
                                 c == '/';
    if (!identifier_like)
    {
      plain_safe = false;
      break;
    }
  }
  if (plain_safe)
  {
    return text;
  }

  std::string quoted;
  quoted.reserve(text.size() + 2U);
  quoted += '"';
  for (const char c : text)
  {
    switch (c)
    {
      case '\\':
        quoted += "\\\\";
        break;
      case '"':
        quoted += "\\\"";
        break;
      case '\n':
        quoted += "\\n";
        break;
      case '\t':
        quoted += "\\t";
        break;
      case '\r':
        quoted += "\\r";
        break;
      default: {
        // Remaining C0 control bytes (and DEL) are not representable raw in
        // a YAML double-quoted scalar: strict parsers reject them and
        // yaml-cpp silently corrupts the string. Emit the \xHH escape.
        const unsigned char byte = static_cast<unsigned char>(c);
        if (byte < 0x20U || byte == 0x7FU)
        {
          constexpr char kHex[] = "0123456789ABCDEF";
          quoted += "\\x";
          quoted += kHex[(byte >> 4U) & 0x0FU];
          quoted += kHex[byte & 0x0FU];
        }
        else
        {
          quoted += c;
        }
        break;
      }
    }
  }
  quoted += '"';
  return quoted;
}

std::string rightPad(const std::string &text, const std::size_t width)
{
  if (text.size() >= width)
  {
    return text;
  }
  return text + std::string(width - text.size(), ' ');
}

// Emit `name:` with rows/cols and a `data: [ ... ]` block, one source row per
// output line, continuation lines indented under the opening bracket and the
// cells right-padded so columns line up vertically.
void appendAlignedMatrix(
  std::ostringstream &out, const char *name, const int rows, const int cols, const double *data,
  CameraInfoYamlPrecision precision
)
{
  out << name << ":\n"
      << "  rows: " << rows << "\n"
      << "  cols: " << cols << "\n";

  const std::size_t cell_count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  std::vector<std::string> cells(cell_count);
  for (std::size_t i = 0; i < cell_count; ++i)
  {
    cells[i] = formatDouble(data[i], kMatrixPrecision, precision);
  }

  std::vector<std::size_t> column_widths(static_cast<std::size_t>(cols), 0U);
  for (int row = 0; row < rows; ++row)
  {
    for (int col = 0; col < cols; ++col)
    {
      const std::size_t index = static_cast<std::size_t>(row * cols + col);
      column_widths[static_cast<std::size_t>(col)] =
        (std::max)(column_widths[static_cast<std::size_t>(col)], cells[index].size());
    }
  }

  out << "  data: [";
  for (int row = 0; row < rows; ++row)
  {
    if (row > 0)
    {
      out << "         ";
    }
    for (int col = 0; col < cols; ++col)
    {
      const std::size_t index = static_cast<std::size_t>(row * cols + col);
      out << rightPad(cells[index], column_widths[static_cast<std::size_t>(col)]);
      if (row == rows - 1 && col == cols - 1)
      {
        out << "]";
      }
      else
      {
        out << ", ";
      }
    }
    out << "\n";
  }
}

std::string formatWithPrecision(const double value, const int digits, const bool fixed)
{
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  if (fixed)
  {
    stream << std::fixed;
  }
  stream.precision(digits);
  stream << value;
  return stream.str();
}

// A decimal as strtod reads it, independent of the global locale: strtod
// takes the C locale's decimal point, so it is handed that character.
bool parseDecimal(const std::string &text, double &value_out)
{
  if (text.empty())
  {
    return false;
  }
  std::string local = text;
  const char point = *std::localeconv()->decimal_point;
  if (point != '.')
  {
    for (char &c : local)
    {
      if (c == '.')
      {
        c = point;
      }
    }
  }
  errno = 0;
  char *end = nullptr;
  const double value = std::strtod(local.c_str(), &end);
  if (end != local.c_str() + local.size())
  {
    return false;
  }
  // ERANGE on underflow still gives the (subnormal or zero) value; on
  // overflow the text was finite, so infinity would not be what it says.
  if (errno == ERANGE && std::isinf(value))
  {
    return false;
  }
  value_out = value;
  return true;
}

bool readsBackAs(const std::string &text, const double value)
{
  double parsed = 0.0;
  return parseDecimal(text, parsed) && parsed == value &&
         std::signbit(parsed) == std::signbit(value);
}

// The fewest significant digits that read back as the same double, written
// without an exponent for ordinary magnitudes, and with a decimal point or an
// exponent so that YAML readers keep the value a float.
std::string shortestRoundTrip(const double value)
{
  if (std::isnan(value)) return ".nan";
  if (std::isinf(value)) return std::signbit(value) ? "-.inf" : ".inf";
  int digits = 1;
  std::string text;
  for (; digits <= std::numeric_limits<double>::max_digits10; ++digits)
  {
    text = formatWithPrecision(value, digits, false);
    if (readsBackAs(text, value))
    {
      break;
    }
  }
  const double magnitude = std::fabs(value);
  if (text.find_first_of("eE") != std::string::npos && magnitude >= 1e-5 && magnitude < 1e16)
  {
    // The same digits as a plain decimal.
    const int exponent = static_cast<int>(std::floor(std::log10(magnitude)));
    const std::string plain = formatWithPrecision(value, std::max(0, digits - 1 - exponent), true);
    if (readsBackAs(plain, value))
    {
      text = plain;
    }
  }
  if (text.find_first_of(".eE") == std::string::npos)
  {
    text += ".0";
  }
  return text;
}

template <typename Values>
std::string flowSequence(const Values &values)
{
  std::string text = "[";
  bool first = true;
  for (const double value : values)
  {
    if (!first)
    {
      text += ", ";
    }
    text += shortestRoundTrip(value);
    first = false;
  }
  return text + "]";
}

// Limits on the documents the readers accept. Camera files are a few hundred
// bytes; these only bound the work a hostile file can cause.
constexpr std::size_t kYamlMaxBytes = std::size_t{1} << 20U;   // 1 MiB
constexpr int kYamlMaxDepth = 16;                              // nested collections
constexpr std::size_t kYamlMaxEntries = 256;                   // keys in one mapping
constexpr std::size_t kYamlMaxItems = 4096;                    // items in one sequence
constexpr std::size_t kYamlMaxVisits = std::size_t{1} << 20U;  // nodes walked, aliases included

// yaml-cpp accepts repeated keys, any number of keys or items, and aliases,
// which can turn a small text into a large or even cyclic tree. One walk over
// the parsed tree refuses repeated or non-scalar keys and anything past the
// limits, and stops after a fixed number of nodes however the aliases expand.
bool withinLimits(const YAML::Node &node, const int depth, std::size_t &visits)
{
  if (++visits > kYamlMaxVisits)
  {
    return false;
  }
  if (node.IsSequence())
  {
    if (depth >= kYamlMaxDepth || node.size() > kYamlMaxItems)
    {
      return false;
    }
    for (const YAML::Node &item : node)
    {
      if (!withinLimits(item, depth + 1, visits))
      {
        return false;
      }
    }
  }
  else if (node.IsMap())
  {
    if (depth >= kYamlMaxDepth || node.size() > kYamlMaxEntries)
    {
      return false;
    }
    std::unordered_set<std::string> keys;
    for (const auto &entry : node)
    {
      if (!entry.first.IsScalar() || !keys.insert(entry.first.Scalar()).second ||
          !withinLimits(entry.second, depth + 1, visits))
      {
        return false;
      }
    }
  }
  return true;
}

// The one document in `yaml` when it is a mapping within the limits, else an
// undefined node. yaml-cpp throws on malformed YAML (tab indentation, an
// unclosed collection, nesting past its own depth guard); every exception is
// a refusal.
YAML::Node loadDocument(const std::string &yaml)
{
  if (yaml.size() <= kYamlMaxBytes)
  {
    try
    {
      const std::vector<YAML::Node> documents = YAML::LoadAll(yaml);
      std::size_t visits = 0U;
      if (documents.size() == 1U && documents[0].IsMap() && withinLimits(documents[0], 0, visits))
      {
        return documents[0];
      }
    }
    catch (const std::exception &)
    {
    }
  }
  return YAML::Node(YAML::NodeType::Undefined);
}

// The value under `key` in a mapping (keys are unique scalars after
// loadDocument), or an undefined node.
YAML::Node find(const YAML::Node &map, const char *key)
{
  if (map.IsMap())
  {
    for (const auto &entry : map)
    {
      if (entry.first.Scalar() == key)
      {
        return entry.second;
      }
    }
  }
  return YAML::Node(YAML::NodeType::Undefined);
}

// A number is a plain scalar ("?" is yaml-cpp's tag for one); a quoted
// scalar is a string even when it looks like a number.
bool plainScalar(const YAML::Node &node)
{
  return node.IsScalar() && node.Tag() == "?" && !node.Scalar().empty();
}

bool readNumber(const YAML::Node &node, double &value_out)
{
  if (!plainScalar(node))
  {
    return false;
  }
  const std::string &s = node.Scalar();
  if (s == ".nan" || s == ".NaN" || s == ".NAN")
  {
    value_out = std::numeric_limits<double>::quiet_NaN();
    return true;
  }
  for (const char *inf : {".inf", ".Inf", ".INF", "+.inf", "+.Inf", "+.INF"})
  {
    if (s == inf)
    {
      value_out = std::numeric_limits<double>::infinity();
      return true;
    }
  }
  for (const char *inf : {"-.inf", "-.Inf", "-.INF"})
  {
    if (s == inf)
    {
      value_out = -std::numeric_limits<double>::infinity();
      return true;
    }
  }
  // Decimal digits, sign, point and exponent only: strtod would also take
  // other spellings (hexadecimal, "nan", "infinity") that YAML reads as strings.
  for (const char c : s)
  {
    if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '+' || c == '-' || c == '.' ||
          c == 'e' || c == 'E'))
    {
      return false;
    }
  }
  return parseDecimal(s, value_out);
}

bool readUnsigned(const YAML::Node &node, std::uint32_t &value_out)
{
  if (!plainScalar(node) || node.Scalar().size() > 10U)
  {
    return false;
  }
  std::uint64_t value = 0U;
  for (const char c : node.Scalar())
  {
    if (!std::isdigit(static_cast<unsigned char>(c)))
    {
      return false;
    }
    value = value * 10U + static_cast<std::uint64_t>(c - '0');
  }
  if (value > std::numeric_limits<std::uint32_t>::max())
  {
    return false;
  }
  value_out = static_cast<std::uint32_t>(value);
  return true;
}

bool readNumbers(const YAML::Node &node, std::vector<double> &values_out)
{
  if (!node.IsSequence())
  {
    return false;
  }
  std::vector<double> values;
  values.reserve(node.size());
  for (const YAML::Node &item : node)
  {
    double value = 0.0;
    if (!readNumber(item, value))
    {
      return false;
    }
    values.push_back(value);
  }
  values_out = std::move(values);
  return true;
}

bool readString(const YAML::Node &node, std::string &value_out)
{
  if (!node.IsScalar())
  {
    return false;
  }
  value_out = node.Scalar();
  return true;
}

// A ROS matrix block: {rows, cols, data: [rows * cols numbers]}.
bool readMatrix(
  const YAML::Node &node, std::uint32_t &rows_out, std::uint32_t &cols_out,
  std::vector<double> &data_out
)
{
  if (!node.IsMap() || !readUnsigned(find(node, "rows"), rows_out) ||
      !readUnsigned(find(node, "cols"), cols_out) || !readNumbers(find(node, "data"), data_out))
  {
    return false;
  }
  return static_cast<std::uint64_t>(rows_out) * cols_out == data_out.size();
}

template <std::size_t N>
bool readFixedMatrix(
  const YAML::Node &node, const std::uint32_t rows, const std::uint32_t cols,
  std::array<double, N> &out
)
{
  std::uint32_t r = 0U;
  std::uint32_t c = 0U;
  std::vector<double> data;
  if (!readMatrix(node, r, c, data) || r != rows || c != cols)
  {
    return false;
  }
  std::copy(data.begin(), data.end(), out.begin());
  return true;
}

// The ROS layout (see parseCameraInfoYaml).
bool readRosLayout(const YAML::Node &root, CameraInfo &camera_info_out)
{
  if (find(root, "projection_model").IsDefined())
  {
    return false;
  }
  CameraInfo info;
  std::uint32_t rows = 0U;
  std::uint32_t cols = 0U;
  if (!readUnsigned(find(root, "image_width"), info.width) ||
      !readUnsigned(find(root, "image_height"), info.height) ||
      !readFixedMatrix(find(root, "camera_matrix"), 3U, 3U, info.k) ||
      !readString(find(root, "distortion_model"), info.distortion_model) ||
      !readMatrix(find(root, "distortion_coefficients"), rows, cols, info.d) ||
      (rows != 1U && cols != 1U && !info.d.empty()))
  {
    return false;
  }
  const YAML::Node rectification = find(root, "rectification_matrix");
  if (rectification.IsDefined() && !readFixedMatrix(rectification, 3U, 3U, info.r))
  {
    return false;
  }
  const YAML::Node projection = find(root, "projection_matrix");
  if (projection.IsDefined() && !readFixedMatrix(projection, 3U, 4U, info.p))
  {
    return false;
  }
  // Binning and ROI, as the ROS writer stores them: folded into K / P and
  // the image size exactly as the ROS layer does for a message, so both
  // paths give the same model.
  BinningAndRoi crop;
  const YAML::Node binning_x = find(root, "binning_x");
  const YAML::Node binning_y = find(root, "binning_y");
  const YAML::Node roi = find(root, "roi");
  if ((binning_x.IsDefined() && !readUnsigned(binning_x, crop.binning_x)) ||
      (binning_y.IsDefined() && !readUnsigned(binning_y, crop.binning_y)))
  {
    return false;
  }
  if (roi.IsDefined())
  {
    if (!roi.IsMap() || !readUnsigned(find(roi, "x_offset"), crop.x_offset) ||
        !readUnsigned(find(roi, "y_offset"), crop.y_offset) ||
        !readUnsigned(find(roi, "width"), crop.width) ||
        !readUnsigned(find(roi, "height"), crop.height))
    {
      return false;
    }
  }
  foldBinningAndRoi(crop, info);
  camera_info_out = info;
  return true;
}

constexpr const char *kCamxiomKeys[] = {
  "image_width",      "image_height",          "projection_model",    "projection_parameters",
  "distortion_model", "distortion_parameters", "intrinsic_parameters"};

// camxiom's layout: exactly the seven keys toCamxiomCameraInfoYaml writes.
bool readCamxiomLayout(const YAML::Node &root, CamxiomCameraInfo &camera_info_out)
{
  for (const auto &entry : root)
  {
    bool known = false;
    for (const char *key : kCamxiomKeys)
    {
      known = known || entry.first.Scalar() == key;
    }
    if (!known)
    {
      return false;
    }
  }
  CamxiomCameraInfo info;
  std::vector<double> intrinsics;
  if (!readUnsigned(find(root, "image_width"), info.image_width) ||
      !readUnsigned(find(root, "image_height"), info.image_height) ||
      !readString(find(root, "projection_model"), info.projection_model) ||
      !readNumbers(find(root, "projection_parameters"), info.projection_parameters) ||
      !readString(find(root, "distortion_model"), info.distortion_model) ||
      !readNumbers(find(root, "distortion_parameters"), info.distortion_parameters) ||
      !readNumbers(find(root, "intrinsic_parameters"), intrinsics) || intrinsics.size() != 5U)
  {
    return false;
  }
  std::copy(intrinsics.begin(), intrinsics.end(), info.intrinsic_parameters.begin());
  camera_info_out = info;
  return true;
}

// Read at most the YAML size limit, refusing a larger file before reading it
// all (a stream that cannot report its size is read up to the limit).
bool readFile(const std::string &path, std::string &text_out)
{
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.is_open())
  {
    return false;
  }
  file.seekg(0, std::ios::end);
  const std::streamoff size = file.tellg();
  if (size > static_cast<std::streamoff>(kYamlMaxBytes))
  {
    return false;
  }
  file.clear();
  if (size >= 0)
  {
    file.seekg(0, std::ios::beg);
  }
  // A stream that cannot seek (a pipe) reported no size and consumed
  // nothing; read it from where it is, up to the limit.
  file.clear();
  std::string text(kYamlMaxBytes + 1U, '\0');
  file.read(&text[0], static_cast<std::streamsize>(text.size()));
  if (file.bad())
  {
    return false;
  }
  const std::size_t read = static_cast<std::size_t>(file.gcount());
  if (read > kYamlMaxBytes)
  {
    return false;
  }
  text.resize(read);
  text_out = std::move(text);
  return true;
}

template <typename T>
StatusCode parseCameraModelYamlImpl(
  const std::string &yaml, CameraModelT<T> &model_out, ImageSize &image_size_out
)
{
  const YAML::Node root = loadDocument(yaml);
  if (!root.IsDefined())
  {
    return StatusCode::INVALID_INPUT;
  }
  CameraModelT<T> model;
  std::uint32_t width = 0U;
  std::uint32_t height = 0U;
  if (find(root, "projection_model").IsDefined())
  {
    CamxiomCameraInfo info;
    if (!readCamxiomLayout(root, info))
    {
      return StatusCode::INVALID_INPUT;
    }
    if constexpr (std::is_same_v<T, float>)
      model = makeCameraModel(info);
    else
      model = makeCameraModel64(info);
    width = info.image_width;
    height = info.image_height;
  }
  else
  {
    CameraInfo info;
    if (!readRosLayout(root, info))
    {
      return StatusCode::INVALID_INPUT;
    }
    if constexpr (std::is_same_v<T, float>)
      model = makeCameraModel(info);
    else
      model = makeCameraModel64(info);
    width = info.width;
    height = info.height;
  }
  const StatusCode validation = [&]() {
    if constexpr (std::is_same_v<T, float>)
      return validateCameraModel(model);
    else
      return validateCameraModel64(model);
  }();
  if (validation != StatusCode::OK)
  {
    return validation;
  }
  constexpr std::uint32_t kMaxInt = static_cast<std::uint32_t>(std::numeric_limits<int>::max());
  if (width > kMaxInt || height > kMaxInt)
  {
    return StatusCode::INVALID_INPUT;
  }
  model_out = model;
  image_size_out = ImageSize{static_cast<int>(width), static_cast<int>(height)};
  return StatusCode::OK;
}

template <typename T>
StatusCode loadCameraModelYamlImpl(
  const std::string &path, CameraModelT<T> &model_out, ImageSize &image_size_out
)
{
  std::string text;
  if (!readFile(path, text))
  {
    return StatusCode::INVALID_INPUT;
  }
  return parseCameraModelYamlImpl(text, model_out, image_size_out);
}

}  // namespace

std::string toCameraInfoYaml(
  const CameraInfo &camera_info, const std::string &camera_name, CameraInfoYamlPrecision precision
)
{
  std::ostringstream out;
  // Same locale pinning as formatDouble: integer fields (width/height,
  // rows/cols) would otherwise pick up thousands grouping from the ambient
  // global locale ("1.920" for 1920 under de_DE).
  out.imbue(std::locale::classic());

  out << "image_width: " << camera_info.width << "\n";
  out << "image_height: " << camera_info.height << "\n";
  out << "camera_name: " << yamlScalar(camera_name) << "\n";

  appendAlignedMatrix(out, "camera_matrix", 3, 3, camera_info.k.data(), precision);

  out << "distortion_model: " << yamlScalar(camera_info.distortion_model) << "\n";
  out << "distortion_coefficients:\n";
  out << "  rows: 1\n";
  out << "  cols: " << camera_info.d.size() << "\n";
  out << "  data: [";
  for (std::size_t i = 0; i < camera_info.d.size(); ++i)
  {
    if (i > 0)
    {
      out << ", ";
    }
    out << formatDouble(camera_info.d[i], kDistortionPrecision, precision);
  }
  out << "]\n";

  appendAlignedMatrix(out, "rectification_matrix", 3, 3, camera_info.r.data(), precision);
  appendAlignedMatrix(out, "projection_matrix", 3, 4, camera_info.p.data(), precision);
  return out.str();
}

StatusCode saveCameraInfoYaml(
  const std::string &path, const CameraInfo &camera_info, const std::string &camera_name,
  CameraInfoYamlPrecision precision
)
{
  std::ofstream file(path, std::ios::out | std::ios::trunc);
  if (!file.is_open())
  {
    return StatusCode::INVALID_INPUT;
  }
  file << toCameraInfoYaml(camera_info, camera_name, precision);
  file.flush();
  return file.good() ? StatusCode::OK : StatusCode::INVALID_INPUT;
}

std::string toCameraInfoYaml(const CameraInfo &camera_info, const std::string &camera_name)
{
  return toCameraInfoYaml(camera_info, camera_name, CameraInfoYamlPrecision::FIXED);
}

StatusCode saveCameraInfoYaml(
  const std::string &path, const CameraInfo &camera_info, const std::string &camera_name
)
{
  return saveCameraInfoYaml(path, camera_info, camera_name, CameraInfoYamlPrecision::FIXED);
}

}  // namespace camxiom

namespace camxiom
{

std::string toCamxiomCameraInfoYaml(const CamxiomCameraInfo &camera_info)
{
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "image_width: " << camera_info.image_width << "\n";
  out << "image_height: " << camera_info.image_height << "\n";
  out << "projection_model: " << yamlScalar(camera_info.projection_model) << "\n";
  out << "projection_parameters: " << flowSequence(camera_info.projection_parameters) << "\n";
  out << "distortion_model: " << yamlScalar(camera_info.distortion_model) << "\n";
  out << "distortion_parameters: " << flowSequence(camera_info.distortion_parameters) << "\n";
  out << "intrinsic_parameters: " << flowSequence(camera_info.intrinsic_parameters) << "\n";
  return out.str();
}

StatusCode saveCamxiomCameraInfoYaml(const std::string &path, const CamxiomCameraInfo &camera_info)
{
  std::ofstream file(path, std::ios::out | std::ios::trunc);
  if (!file.is_open())
  {
    return StatusCode::INVALID_INPUT;
  }
  file << toCamxiomCameraInfoYaml(camera_info);
  file.flush();
  return file.good() ? StatusCode::OK : StatusCode::INVALID_INPUT;
}

StatusCode parseCameraInfoYaml(const std::string &yaml, CameraInfo &camera_info_out)
{
  const YAML::Node root = loadDocument(yaml);
  return root.IsDefined() && readRosLayout(root, camera_info_out) ? StatusCode::OK
                                                                  : StatusCode::INVALID_INPUT;
}

StatusCode parseCamxiomCameraInfoYaml(const std::string &yaml, CamxiomCameraInfo &camera_info_out)
{
  const YAML::Node root = loadDocument(yaml);
  return root.IsDefined() && readCamxiomLayout(root, camera_info_out) ? StatusCode::OK
                                                                      : StatusCode::INVALID_INPUT;
}

StatusCode parseCameraModelYaml(
  const std::string &yaml, CameraModel &model_out, ImageSize &image_size_out
)
{
  return parseCameraModelYamlImpl(yaml, model_out, image_size_out);
}

StatusCode parseCameraModelYaml64(
  const std::string &yaml, CameraModel64 &model_out, ImageSize &image_size_out
)
{
  return parseCameraModelYamlImpl(yaml, model_out, image_size_out);
}

StatusCode loadCameraModelYaml(
  const std::string &path, CameraModel &model_out, ImageSize &image_size_out
)
{
  return loadCameraModelYamlImpl(path, model_out, image_size_out);
}

StatusCode loadCameraModelYaml64(
  const std::string &path, CameraModel64 &model_out, ImageSize &image_size_out
)
{
  return loadCameraModelYamlImpl(path, model_out, image_size_out);
}

}  // namespace camxiom
