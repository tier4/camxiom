# Camera files in YAML (example)

camxiom converts a camera model to and from two plain structs —
`camxiom::CameraInfo` (the fields of a ROS CameraInfo) and
`camxiom::CamxiomCameraInfo` (camxiom's own description, which holds every
model) — and leaves files to the application. This page shows one way an
application can save and load them as YAML with
[yaml-cpp](https://github.com/jbeder/yaml-cpp), in two layouts:

- the **ROS layout** of `camera_calibration_parsers`, which ROS camera
  drivers and calibration tools read and write;
- **camxiom's layout**, one key per field of `CamxiomCameraInfo`.

The code is an example to copy and adapt, not part of camxiom. It needs
C++17 and yaml-cpp 0.7 or newer (`find_package(yaml-cpp REQUIRED)`; the
target is `yaml-cpp::yaml-cpp` in 0.8 and `yaml-cpp` in 0.7). See
[CameraInfo interchange](../api/camera-info.md) for the two structs and the
models each can hold.

## The two layouts

The ROS layout:

```yaml
image_width: 1280
image_height: 720
camera_name: front
camera_matrix:
  rows: 3
  cols: 3
  data: [700.0, 0.0, 639.5, 0.0, 700.0, 359.5, 0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [-0.1, 0.01, 0.0005, -0.0003, -0.002]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [700.0, 0.0, 639.5, 0.0, 0.0, 700.0, 359.5, 0.0, 0.0, 0.0, 1.0, 0.0]
```

camxiom's layout:

```yaml
image_width: 1280
image_height: 960
projection_model: equidistant
projection_parameters: []
distortion_model: kb4
distortion_parameters: [0.021, -0.005, 0.001, -0.0002]
intrinsic_parameters: [320.5, 320.8205, 639.5, 479.5, 0.0]
```

`projection_model` and `distortion_model` are the `toString` names;
`projection_parameters` holds `xi` (omnidirectional), `xi, alpha` (double
sphere), `alpha, beta` (EUCM) or nothing; `distortion_parameters` holds
exactly the correction's coefficients; `intrinsic_parameters` is
`fx, fy, cx, cy, skew`. A file that has a `projection_model` key is in
camxiom's layout, any other camera file in the ROS layout.

## Writing

The writers format the text themselves, so the layout and the numbers are
under the application's control. Every number is written with the fewest
digits that read back as the same double; a float model's values are
widened to double first (`exportCameraInfo` does that), so they read back as
the same floats.

```cpp
#include <camxiom/compat.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <exception>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace camera_yaml
{

// The fewest digits that read back as the same double: a plain decimal for
// ordinary magnitudes, an exponent only for tiny or huge ones, and ".0" on
// integral values so that YAML readers keep them floating point.
inline std::string number(const double value)
{
  const double magnitude = std::fabs(value);
  const bool plain = magnitude == 0.0 || (magnitude >= 1e-5 && magnitude < 1e16);
  char buffer[64];
  const auto result =
    plain ? std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::fixed)
          : std::to_chars(buffer, buffer + sizeof(buffer), value);
  std::string text(buffer, result.ptr);
  if (text.find_first_of(".e") == std::string::npos)
  {
    text += ".0";
  }
  return text;
}

template <typename Values>
std::string sequence(const Values &values)
{
  std::string text = "[";
  for (const double value : values)
  {
    text += (text.size() > 1 ? ", " : "") + number(value);
  }
  return text + "]";
}

// A string as one YAML scalar, quoted only when it has to be.
inline std::string scalar(const std::string &text)
{
  YAML::Emitter emitter;
  emitter << text;
  return emitter.c_str();
}

// camxiom's layout: one key per field of CamxiomCameraInfo.
inline std::string toYaml(const camxiom::CamxiomCameraInfo &info)
{
  std::ostringstream out;
  out.imbue(std::locale::classic());  // no digit grouping in the integers
  out << "image_width: " << info.image_width << "\n"
      << "image_height: " << info.image_height << "\n"
      << "projection_model: " << scalar(info.projection_model) << "\n"
      << "projection_parameters: " << sequence(info.projection_parameters) << "\n"
      << "distortion_model: " << scalar(info.distortion_model) << "\n"
      << "distortion_parameters: " << sequence(info.distortion_parameters) << "\n"
      << "intrinsic_parameters: " << sequence(info.intrinsic_parameters) << "\n";
  return out.str();
}

inline void appendMatrix(
  std::ostringstream &out, const char *name, const int rows, const int cols, const double *data
)
{
  out << name << ":\n  rows: " << rows << "\n  cols: " << cols << "\n  data: [";
  for (int i = 0; i < rows * cols; ++i)
  {
    out << (i > 0 ? ", " : "") << number(data[i]);
  }
  out << "]\n";
}

// The ROS camera_calibration_parsers layout.
inline std::string toRosYaml(const camxiom::CameraInfo &info, const std::string &camera_name)
{
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "image_width: " << info.width << "\n"
      << "image_height: " << info.height << "\n"
      << "camera_name: " << scalar(camera_name) << "\n";
  appendMatrix(out, "camera_matrix", 3, 3, info.k.data());
  out << "distortion_model: " << scalar(info.distortion_model) << "\n";
  appendMatrix(out, "distortion_coefficients", 1, static_cast<int>(info.d.size()), info.d.data());
  appendMatrix(out, "rectification_matrix", 3, 3, info.r.data());
  appendMatrix(out, "projection_matrix", 3, 4, info.p.data());
  return out.str();
}
```

`camera_calibration_parsers` itself writes the matrices with fixed decimals;
both read the same.

## Reading

The readers parse with yaml-cpp and then check the layout themselves: the
required keys, the number of values, and the matrix shapes. The helpers
throw on anything unexpected; `fromYaml` and `fromRosYaml` catch it and
return `false`.

```cpp
// Exactly one document holding a mapping. The size limit bounds the work an
// untrusted file can cause; camera files are a few hundred bytes.
inline YAML::Node parseDocument(const std::string &text)
{
  constexpr std::size_t kMaxBytes = std::size_t{1} << 20U;
  if (text.size() > kMaxBytes)
  {
    throw std::runtime_error("camera file too large");
  }
  const std::vector<YAML::Node> documents = YAML::LoadAll(text);
  if (documents.size() != 1 || !documents[0].IsMap())
  {
    throw std::runtime_error("expected one mapping");
  }
  return documents[0];
}

inline YAML::Node required(const YAML::Node &map, const char *key)
{
  const YAML::Node node = map[key];
  if (!node)
  {
    throw std::runtime_error(std::string("missing key ") + key);
  }
  return node;
}

// A number, read independently of the program's locale.
inline double readNumber(const YAML::Node &node)
{
  if (!node.IsScalar())
  {
    throw std::runtime_error("expected a number");
  }
  const std::string &text = node.Scalar();
  double value = 0.0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc() || result.ptr != text.data() + text.size())
  {
    throw std::runtime_error("expected a number");
  }
  return value;
}

inline std::vector<double> readNumbers(const YAML::Node &node)
{
  if (!node.IsSequence())
  {
    throw std::runtime_error("expected a sequence of numbers");
  }
  std::vector<double> values;
  for (const YAML::Node &item : node)
  {
    values.push_back(readNumber(item));
  }
  return values;
}

inline bool fromYaml(const std::string &text, camxiom::CamxiomCameraInfo &info_out)
{
  try
  {
    const YAML::Node root = parseDocument(text);
    camxiom::CamxiomCameraInfo info;
    info.image_width = required(root, "image_width").as<std::uint32_t>();
    info.image_height = required(root, "image_height").as<std::uint32_t>();
    info.projection_model = required(root, "projection_model").as<std::string>();
    info.projection_parameters = readNumbers(required(root, "projection_parameters"));
    info.distortion_model = required(root, "distortion_model").as<std::string>();
    info.distortion_parameters = readNumbers(required(root, "distortion_parameters"));
    const std::vector<double> intrinsics = readNumbers(required(root, "intrinsic_parameters"));
    if (intrinsics.size() != info.intrinsic_parameters.size())
    {
      return false;
    }
    std::copy(intrinsics.begin(), intrinsics.end(), info.intrinsic_parameters.begin());
    info_out = info;
    return true;
  }
  catch (const std::exception &)  // YAML::Exception included
  {
    return false;
  }
}
```

The ROS layout keeps K and P in full-resolution pixels; a file from a
binned or cropped camera also has `binning_x`, `binning_y` and `roi`, and
the image it describes is the ROI crop decimated by the binning. The reader
folds both into K, P and the image size the same way camxiom's
`sensor_msgs` conversion does (see [ROS / OpenCV interop](../api/interop.md)),
so a file and the same CameraInfo as a message give the same model. An
application that does not expect binned or cropped cameras can refuse such
files instead.

```cpp
// A ROS matrix block {rows, cols, data}.
inline std::vector<double> readMatrix(
  const YAML::Node &node, std::uint32_t &rows_out, std::uint32_t &cols_out
)
{
  rows_out = required(node, "rows").as<std::uint32_t>();
  cols_out = required(node, "cols").as<std::uint32_t>();
  std::vector<double> data = readNumbers(required(node, "data"));
  if (data.size() != std::size_t{rows_out} * cols_out)
  {
    throw std::runtime_error("matrix size does not match rows x cols");
  }
  return data;
}

template <std::size_t N>
void readMatrix(
  const YAML::Node &node, const std::uint32_t rows, const std::uint32_t cols,
  std::array<double, N> &out
)
{
  std::uint32_t r = 0;
  std::uint32_t c = 0;
  const std::vector<double> data = readMatrix(node, r, c);
  if (r != rows || c != cols)
  {
    throw std::runtime_error("unexpected matrix shape");
  }
  std::copy(data.begin(), data.end(), out.begin());
}

// Binning 0 means 1, and an ROI of size 0 means the whole image.
inline void foldBinningAndRoi(const YAML::Node &root, camxiom::CameraInfo &info)
{
  // An absent key (or an absent roi block) reads as 0.
  const auto count = [](const YAML::Node &map, const char *key) -> std::uint32_t {
    if (!map || !map[key])
    {
      return 0;
    }
    return map[key].as<std::uint32_t>();
  };
  const std::uint32_t bx = std::max<std::uint32_t>(1, count(root, "binning_x"));
  const std::uint32_t by = std::max<std::uint32_t>(1, count(root, "binning_y"));
  const YAML::Node roi = root["roi"];
  const double ox = count(roi, "x_offset");
  const double oy = count(roi, "y_offset");
  const std::uint32_t roi_width = count(roi, "width") != 0 ? count(roi, "width") : info.width;
  const std::uint32_t roi_height = count(roi, "height") != 0 ? count(roi, "height") : info.height;

  info.k[0] /= bx;                    // fx
  info.k[1] /= bx;                    // skew
  info.k[2] = (info.k[2] - ox) / bx;  // cx
  info.k[4] /= by;                    // fy
  info.k[5] = (info.k[5] - oy) / by;  // cy
  info.p[0] /= bx;
  info.p[1] /= bx;
  info.p[2] = (info.p[2] - ox) / bx;
  info.p[3] /= bx;
  info.p[5] /= by;
  info.p[6] = (info.p[6] - oy) / by;
  info.p[7] /= by;
  info.width = roi_width / bx;
  info.height = roi_height / by;
}

inline bool fromRosYaml(const std::string &text, camxiom::CameraInfo &info_out)
{
  try
  {
    const YAML::Node root = parseDocument(text);
    if (root["projection_model"])
    {
      return false;  // camxiom's layout
    }
    camxiom::CameraInfo info;
    info.width = required(root, "image_width").as<std::uint32_t>();
    info.height = required(root, "image_height").as<std::uint32_t>();
    readMatrix(required(root, "camera_matrix"), 3, 3, info.k);
    info.distortion_model = required(root, "distortion_model").as<std::string>();
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    info.d = readMatrix(required(root, "distortion_coefficients"), rows, cols);
    if (rows != 1 && cols != 1 && !info.d.empty())
    {
      return false;  // D is one row (or one column)
    }
    if (root["rectification_matrix"])
    {
      readMatrix(root["rectification_matrix"], 3, 3, info.r);
    }
    if (root["projection_matrix"])
    {
      readMatrix(root["projection_matrix"], 3, 4, info.p);
    }
    foldBinningAndRoi(root, info);
    info_out = info;
    return true;
  }
  catch (const std::exception &)
  {
    return false;
  }
}

}  // namespace camera_yaml
```

## Saving and loading a model

`exportCameraInfo` fills a struct from a valid model (the ROS struct only
when ROS can hold the same lens); the image size is the application's to
set. On reading, `makeCameraModel` builds the model and
`validateCameraModel` tells whether the file described a valid one.

```cpp
#include <camxiom/model.hpp>

#include <fstream>

// Save in camxiom's layout, which holds every model.
bool saveModel(const std::string &path, const camxiom::CameraModel &model, int width, int height)
{
  camxiom::CamxiomCameraInfo info;
  if (camxiom::exportCameraInfo(model, info) != camxiom::StatusCode::OK)
  {
    return false;  // not a valid model
  }
  info.image_width = static_cast<std::uint32_t>(width);
  info.image_height = static_cast<std::uint32_t>(height);
  std::ofstream file(path);
  file << camera_yaml::toYaml(info);
  return static_cast<bool>(file);
}

// Load either layout. Reads at most one byte past the size limit, so an
// oversized file is refused without reading all of it.
bool loadModel(const std::string &path, camxiom::CameraModel &model_out)
{
  std::ifstream file(path, std::ios::binary);
  std::string text((std::size_t{1} << 20U) + 1U, '\0');
  file.read(text.data(), static_cast<std::streamsize>(text.size()));
  text.resize(static_cast<std::size_t>(file.gcount()));

  camxiom::CameraModel model;
  camxiom::CamxiomCameraInfo description;
  camxiom::CameraInfo ros;
  if (camera_yaml::fromYaml(text, description))
  {
    model = camxiom::makeCameraModel(description);
  }
  else if (camera_yaml::fromRosYaml(text, ros))
  {
    model = camxiom::makeCameraModel(ros);
  }
  else
  {
    return false;  // not a camera file in either layout
  }
  if (camxiom::validateCameraModel(model) != camxiom::StatusCode::OK)
  {
    return false;  // the file does not describe a valid model
  }
  model_out = model;
  return true;
}
```

Saving in the ROS layout is the same with `camxiom::CameraInfo` and
`toRosYaml`; `exportCameraInfo` returns `INVALID_MODEL` for a lens ROS
cannot hold (see [CameraInfo interchange](../api/camera-info.md)), and
`makeCameraModel64` / `exportCameraInfo64` keep a double model in double.

## Points to take care of

- **Required keys and counts.** yaml-cpp reads any YAML; check that every
  key the layout needs is there and that each sequence has the expected
  number of values (nine for K, five intrinsics, `rows × cols` for a ROS
  matrix). `makeCameraModel` then checks that the names and the number of
  coefficients fit the model.
- **One document.** `YAML::Load` reads only the first document of a
  file; `YAML::LoadAll` lets the reader refuse a second one.
- **Untrusted files.** Bound the input size before parsing. yaml-cpp
  limits nesting depth itself, but accepts repeated keys and aliases (which
  can expand a small file into a large tree); check those too if the files
  come from outside.
- **Numbers.** Write the fewest digits that read back as the same double
  (`std::to_chars`), and read them with `std::from_chars`. Both ignore the
  program's locale; yaml-cpp's own `as<double>()` follows the global C++
  locale.
- **Exceptions.** yaml-cpp throws `YAML::Exception` on malformed YAML and
  on a failed `as<T>()`; catch it at the reader's boundary.
- **`theta_max`** is not part of either layout; `makeCameraModel` derives it
  from the model. A narrower limit set by the application has to be stored
  and applied by the application.
