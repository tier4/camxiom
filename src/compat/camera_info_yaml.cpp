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

#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
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

std::string formatFixedDouble(const double value, const int precision)
{
  std::ostringstream stream;
  // A machine-readable serializer must not follow the ambient global locale:
  // a comma decimal point (e.g. de_DE) would split every number into two
  // flow-sequence entries for any real YAML parser reading the file back.
  stream.imbue(std::locale::classic());
  stream << std::fixed << std::setprecision(precision) << value;
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
  std::ostringstream &out, const char *name, const int rows, const int cols, const double *data
)
{
  out << name << ":\n"
      << "  rows: " << rows << "\n"
      << "  cols: " << cols << "\n";

  const std::size_t cell_count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  std::vector<std::string> cells(cell_count);
  for (std::size_t i = 0; i < cell_count; ++i)
  {
    cells[i] = formatFixedDouble(data[i], kMatrixPrecision);
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

}  // namespace

std::string toCameraInfoYaml(const CameraInfo &camera_info, const std::string &camera_name)
{
  std::ostringstream out;
  // Same locale pinning as formatFixedDouble: integer fields (width/height,
  // rows/cols) would otherwise pick up thousands grouping from the ambient
  // global locale ("1.920" for 1920 under de_DE).
  out.imbue(std::locale::classic());

  out << "image_width: " << camera_info.width << "\n";
  out << "image_height: " << camera_info.height << "\n";
  out << "camera_name: " << yamlScalar(camera_name) << "\n";

  appendAlignedMatrix(out, "camera_matrix", 3, 3, camera_info.k.data());

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
    out << formatFixedDouble(camera_info.d[i], kDistortionPrecision);
  }
  out << "]\n";

  appendAlignedMatrix(out, "rectification_matrix", 3, 3, camera_info.r.data());
  appendAlignedMatrix(out, "projection_matrix", 3, 4, camera_info.p.data());
  return out.str();
}

StatusCode saveCameraInfoYaml(
  const std::string &path, const CameraInfo &camera_info, const std::string &camera_name
)
{
  std::ofstream file(path, std::ios::out | std::ios::trunc);
  if (!file.is_open())
  {
    return StatusCode::INVALID_INPUT;
  }
  file << toCameraInfoYaml(camera_info, camera_name);
  file.flush();
  return file.good() ? StatusCode::OK : StatusCode::INVALID_INPUT;
}

}  // namespace camxiom
