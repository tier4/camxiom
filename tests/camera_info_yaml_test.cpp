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

// Tests for the ROS-free CameraInfo YAML writer (camxiom/camera_info_yaml.hpp).
//
// The layout must match the intrinsic camera calibrator's emitter: the
// calibration-parsers section structure, fixed-point numbers (matrices at 6
// decimals, distortion at 12), and column-aligned multi-line matrix blocks.

#include "camxiom/camera_info_yaml.hpp"

#include "camxiom/compat.hpp"
#include "camxiom/model.hpp"

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <locale>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using camxiom::CameraInfo;
using camxiom::StatusCode;

namespace
{

CameraInfo makeTestInfo()
{
  CameraInfo info;
  info.width = 1920U;
  info.height = 1080U;
  info.k = {501.25, 0.0, 960.5, 0.0, 502.75, 540.25, 0.0, 0.0, 1.0};
  info.d = {-0.1, 0.01, 1.0 / 3.0, -0.002, 0.0005};
  info.distortion_model = "plumb_bob";
  // r stays identity; p carries K with a stereo baseline Tx.
  info.p = {501.25, 0.0, 960.5, -35.0875, 0.0, 502.75, 540.25, 0.0, 0.0, 0.0, 1.0, 0.0};
  return info;
}

// Global-locale saboteur: formats numbers like a comma-decimal European
// locale (1.920,50) without depending on any OS locale being installed.
class CommaDecimalPunct : public std::numpunct<char>
{
protected:
  char do_decimal_point() const override { return ','; }
  char do_thousands_sep() const override { return '.'; }
  std::string do_grouping() const override { return "\3"; }
};

}  // namespace

TEST(CameraInfoYaml, EmitsTheCalibrationParsersLayout)
{
  const std::string yaml = camxiom::toCameraInfoYaml(makeTestInfo(), "front_camera");

  EXPECT_NE(yaml.find("image_width: 1920\n"), std::string::npos);
  EXPECT_NE(yaml.find("image_height: 1080\n"), std::string::npos);
  EXPECT_NE(yaml.find("camera_name: front_camera\n"), std::string::npos);
  EXPECT_NE(yaml.find("distortion_model: plumb_bob\n"), std::string::npos);

  for (const char *section :
       {"camera_matrix", "distortion_coefficients", "rectification_matrix", "projection_matrix"})
  {
    const std::size_t pos = yaml.find(std::string(section) + ":\n");
    ASSERT_NE(pos, std::string::npos) << section;
  }

  // Matrix shapes.
  EXPECT_NE(yaml.find("camera_matrix:\n  rows: 3\n  cols: 3\n"), std::string::npos);
  EXPECT_NE(yaml.find("distortion_coefficients:\n  rows: 1\n  cols: 5\n"), std::string::npos);
  EXPECT_NE(yaml.find("projection_matrix:\n  rows: 3\n  cols: 4\n"), std::string::npos);

  // Fixed-point spot values: matrices at 6 decimals, distortion at 12.
  EXPECT_NE(yaml.find("501.250000"), std::string::npos);
  EXPECT_NE(yaml.find("-35.087500"), std::string::npos);
  EXPECT_NE(yaml.find("-0.100000000000"), std::string::npos);
}

TEST(CameraInfoYaml, MatchesTheCalibratorAlignedMatrixLayout)
{
  // The matrix blocks must reproduce the intrinsic calibrator's emitter:
  // one source row per line, continuation lines indented under the opening
  // bracket (9 spaces), cells right-padded per column. The identity
  // rectification matrix has uniform cell widths, so its whole block can be
  // pinned verbatim.
  const std::string yaml = camxiom::toCameraInfoYaml(makeTestInfo());

  // Note the trailing ", " before each line break — the calibrator's emitter
  // writes the separator after the row's last cell too (except the very
  // last), and we reproduce that byte-for-byte.
  const std::string expected_r_block =
    "rectification_matrix:\n"
    "  rows: 3\n"
    "  cols: 3\n"
    "  data: [1.000000, 0.000000, 0.000000, \n"
    "         0.000000, 1.000000, 0.000000, \n"
    "         0.000000, 0.000000, 1.000000]\n";
  EXPECT_NE(yaml.find(expected_r_block), std::string::npos)
    << "rectification block does not match the calibrator layout:\n"
    << yaml;

  // K has mixed cell widths: check the column alignment on its first row —
  // "0.000000" in column 1 is padded to the width of "502.750000".
  EXPECT_NE(yaml.find("  data: [501.250000, 0.000000  , 960.500000, \n"), std::string::npos)
    << "camera_matrix first row is not column-aligned:\n"
    << yaml;
}

TEST(CameraInfoYaml, NumbersRoundTripToWriterPrecision)
{
  const CameraInfo info = makeTestInfo();
  const std::string yaml = camxiom::toCameraInfoYaml(info);

  // Distortion coefficients are written at 12 fixed decimals, so a re-parse
  // must agree to 5e-13 (1/3 exercises the truncation).
  const std::size_t dist = yaml.find("distortion_coefficients:");
  ASSERT_NE(dist, std::string::npos);
  const std::size_t open = yaml.find('[', dist);
  const std::size_t close = yaml.find(']', dist);
  ASSERT_NE(open, std::string::npos);
  ASSERT_NE(close, std::string::npos);

  std::istringstream data(yaml.substr(open + 1, close - open - 1));
  for (std::size_t i = 0; i < info.d.size(); ++i)
  {
    std::string token;
    ASSERT_TRUE(std::getline(data, token, ','));
    EXPECT_NEAR(std::stod(token), info.d[i], 5e-13) << "coefficient " << i;
  }
}

TEST(CameraInfoYaml, OutputIgnoresACommaDecimalGlobalLocale)
{
  // A consuming process may set a comma-decimal global locale (Qt/X11 init
  // is a classic offender). The writer must keep emitting `.` decimals and
  // ungrouped integers, or every number corrupts the comma-delimited flow
  // sequences on read-back.
  const CameraInfo info = makeTestInfo();
  const std::string reference = camxiom::toCameraInfoYaml(info, "cam0");

  const std::locale saboteur(std::locale::classic(), new CommaDecimalPunct);
  const std::locale previous = std::locale::global(saboteur);
  const std::string sabotaged = camxiom::toCameraInfoYaml(info, "cam0");
  std::locale::global(previous);

  EXPECT_EQ(sabotaged, reference);
  EXPECT_EQ(sabotaged.find("1.920"), std::string::npos)
    << "integer fields picked up thousands grouping";
  EXPECT_NE(sabotaged.find("501.250000"), std::string::npos)
    << "decimal point switched away from '.'";
}

TEST(CameraInfoYaml, QuotesNonIdentifierStrings)
{
  // Identifier-like names stay plain (pinned byte-for-byte by the layout
  // tests above); anything YAML-structural must come out double-quoted so a
  // real parser reading the file back cannot mis-nest the document.
  CameraInfo info = makeTestInfo();
  info.distortion_model = "plumb bob";  // embedded space

  const std::string yaml = camxiom::toCameraInfoYaml(info, "front: left #1 \"quoted\"");

  EXPECT_NE(yaml.find("camera_name: \"front: left #1 \\\"quoted\\\"\"\n"), std::string::npos)
    << yaml;
  EXPECT_NE(yaml.find("distortion_model: \"plumb bob\"\n"), std::string::npos) << yaml;
}

TEST(CameraInfoYaml, EscapesControlCharacters)
{
  // Raw C0 control bytes inside a double-quoted scalar are invalid YAML:
  // strict parsers reject the document and yaml-cpp silently corrupts the
  // string (verified: "bad\x01char" round-trips mangled). They must come
  // out as \xHH escapes.
  const CameraInfo info = makeTestInfo();
  const std::string yaml = camxiom::toCameraInfoYaml(info, std::string("bad\x01") + "char\x7f");

  EXPECT_NE(yaml.find("camera_name: \"bad\\x01char\\x7F\"\n"), std::string::npos) << yaml;
  // No raw control byte may survive anywhere in the document.
  for (const char c : yaml)
  {
    const unsigned char byte = static_cast<unsigned char>(c);
    EXPECT_TRUE(byte == '\n' || byte >= 0x20U) << "raw control byte 0x" << std::hex << +byte;
  }
}

TEST(CameraInfoYaml, SaveWritesTheExactStringAndFailsOnBadPath)
{
  const CameraInfo info = makeTestInfo();
  const std::string path = testing::TempDir() + "/camxiom_camera_info_yaml_test.yaml";

  ASSERT_EQ(camxiom::saveCameraInfoYaml(path, info, "cam0"), StatusCode::OK);

  std::ifstream file(path);
  ASSERT_TRUE(file.is_open());
  std::ostringstream read_back;
  read_back << file.rdbuf();
  EXPECT_EQ(read_back.str(), camxiom::toCameraInfoYaml(info, "cam0"));
  std::remove(path.c_str());

  EXPECT_EQ(
    camxiom::saveCameraInfoYaml("/nonexistent-dir-for-camxiom-test/x.yaml", info),
    StatusCode::INVALID_INPUT
  );
}

namespace
{
std::vector<double> parseNumericSequence(const std::string &yaml, const std::string &section)
{
  const auto section_start = yaml.find(section + ":");
  const auto start = yaml.find('[', section_start);
  const auto end = yaml.find(']', start);
  EXPECT_NE(section_start, std::string::npos);
  EXPECT_NE(start, std::string::npos);
  EXPECT_NE(end, std::string::npos);
  if (end == std::string::npos || start == std::string::npos) return {};
  auto data = yaml.substr(start + 1, end - start - 1);
  std::replace(data.begin(), data.end(), ',', ' ');
  std::istringstream input(data);
  input.imbue(std::locale::classic());
  std::vector<double> values;
  double value;
  while (input >> value) values.push_back(value);
  EXPECT_TRUE(input.eof());
  return values;
}

template <typename Values>
void expectExactValues(const std::vector<double> &actual, const Values &expected)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_EQ(actual[i], expected[i]) << i;
    EXPECT_EQ(std::signbit(actual[i]), std::signbit(expected[i])) << i;
  }
}
}  // namespace

TEST(CameraInfoYaml, RoundTripModePreservesEveryFiniteDouble)
{
  CameraInfo info = makeTestInfo();
  info.k[0] = 513.1234567890123;
  info.k[1] = -0.0;
  info.k[2] = std::nextafter(320.0, 321.0);
  info.r[1] = 1.0 / 3.0;
  info.p[3] = -35.087512345678901;
  info.d = {
    0.0,
    -0.0,
    1.0 / 3.0,
    std::numeric_limits<double>::min(),
    std::numeric_limits<double>::denorm_min(),
    std::numeric_limits<double>::max(),
    -std::numeric_limits<double>::denorm_min(),
    std::nextafter(1.0, 2.0)};
  // Exercise a reproducible spread of mantissas and exponents, independently
  // of the formatting implementation, including values fixed decimals erase.
  std::mt19937_64 random(42);
  for (int i = 0; i < 256; ++i)
  {
    const std::uint64_t bits = random();
    double value;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    if (std::isfinite(value)) info.d.push_back(value);
  }
  const auto yaml =
    camxiom::toCameraInfoYaml(info, "camera", camxiom::CameraInfoYamlPrecision::ROUND_TRIP);
  expectExactValues(parseNumericSequence(yaml, "camera_matrix"), info.k);
  expectExactValues(parseNumericSequence(yaml, "distortion_coefficients"), info.d);
  expectExactValues(parseNumericSequence(yaml, "rectification_matrix"), info.r);
  expectExactValues(parseNumericSequence(yaml, "projection_matrix"), info.p);
  EXPECT_EQ(
    camxiom::toCameraInfoYaml(info, "camera", camxiom::CameraInfoYamlPrecision::FIXED),
    camxiom::toCameraInfoYaml(info)
  );
}

TEST(CameraInfoYaml, RoundTripModeIsLocaleIndependentAndSavesExactText)
{
  auto info = makeTestInfo();
  info.k[0] = 513.1234567890123;
  const auto precision = camxiom::CameraInfoYamlPrecision::ROUND_TRIP;
  const auto reference = camxiom::toCameraInfoYaml(info, "cam0", precision);
  const auto old_locale = std::locale();
  std::locale::global(std::locale(old_locale, new CommaDecimalPunct));
  const auto actual = camxiom::toCameraInfoYaml(info, "cam0", precision);
  std::locale::global(old_locale);
  EXPECT_EQ(actual, reference);
  const auto path = testing::TempDir() + "/camxiom_camera_info_round_trip_test.yaml";
  ASSERT_EQ(camxiom::saveCameraInfoYaml(path, info, "cam0", precision), StatusCode::OK);
  std::ifstream file(path);
  ASSERT_TRUE(file.is_open());
  std::ostringstream contents;
  contents << file.rdbuf();
  EXPECT_EQ(contents.str(), reference);
  std::remove(path.c_str());
  EXPECT_EQ(
    camxiom::saveCameraInfoYaml(
      "/nonexistent-dir-for-camxiom-test/x.yaml", info, "cam0", precision
    ),
    StatusCode::INVALID_INPUT
  );
}

TEST(CameraInfoYaml, BracedNameStillSelectsTheNameOverload)
{
  // An empty brace list once value-initialized the precision and picked the
  // explicit-precision overload, which replaced the empty name with "camera".
  const auto info = makeTestInfo();
  EXPECT_EQ(camxiom::toCameraInfoYaml(info, {}), camxiom::toCameraInfoYaml(info, std::string()));
  EXPECT_NE(camxiom::toCameraInfoYaml(info, {}), camxiom::toCameraInfoYaml(info));
}

TEST(CameraInfoYaml, RoundTripModeUsesYamlNonFiniteSpellings)
{
  auto info = makeTestInfo();
  info.d = {
    std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity()};
  const auto yaml =
    camxiom::toCameraInfoYaml(info, "camera", camxiom::CameraInfoYamlPrecision::ROUND_TRIP);
  EXPECT_NE(yaml.find("data: [.nan, .inf, -.inf]"), std::string::npos);
}

// ===========================================================================
// camxiom's own layout, and reading both layouts back.
// ===========================================================================

namespace
{

camxiom::CamxiomCameraInfo makeCamxiomInfo()
{
  camxiom::CamxiomCameraInfo info;
  info.image_width = 1280U;
  info.image_height = 960U;
  info.projection_model = "equidistant";
  info.distortion_model = "kb4";
  info.distortion_parameters = {0.02, -0.005, 0.0, 1.0 / 3.0};
  info.intrinsic_parameters = {320.0, 320.5, 639.5, 479.5, 0.0};
  return info;
}

std::string writeTemp(const std::string &text)
{
  char path[] = "/tmp/camxiom_yaml_test_XXXXXX";
  const int fd = ::mkstemp(path);
  EXPECT_GE(fd, 0);
  ::close(fd);
  std::ofstream file(path, std::ios::out | std::ios::trunc);
  file << text;
  return path;
}

}  // namespace

TEST(CamxiomCameraInfoYaml, EmitsOneKeyPerField)
{
  EXPECT_EQ(
    camxiom::toCamxiomCameraInfoYaml(makeCamxiomInfo()),
    "image_width: 1280\n"
    "image_height: 960\n"
    "projection_model: equidistant\n"
    "projection_parameters: []\n"
    "distortion_model: kb4\n"
    "distortion_parameters: [0.02, -0.005, 0.0, 0.3333333333333333]\n"
    "intrinsic_parameters: [320.0, 320.5, 639.5, 479.5, 0.0]\n"
  );
}

TEST(CamxiomCameraInfoYaml, RoundTripsEveryDoubleExactly)
{
  std::mt19937_64 rng(11U);
  std::uniform_int_distribution<std::uint64_t> bits;
  camxiom::CamxiomCameraInfo info = makeCamxiomInfo();
  info.projection_model = "double_sphere";
  for (int trial = 0; trial < 200; ++trial)
  {
    std::vector<double> values;
    while (values.size() < 11U)
    {
      double value = 0.0;
      const std::uint64_t raw = bits(rng);
      std::memcpy(&value, &raw, sizeof(value));
      if (std::isfinite(value))
      {
        values.push_back(value);
      }
    }
    info.projection_parameters = {values[0], values[1]};
    info.distortion_parameters = {values[2], values[3], values[4], values[5]};
    std::copy(values.begin() + 6, values.end(), info.intrinsic_parameters.begin());
    camxiom::CamxiomCameraInfo back;
    ASSERT_EQ(
      camxiom::parseCamxiomCameraInfoYaml(camxiom::toCamxiomCameraInfoYaml(info), back),
      StatusCode::OK
    );
    EXPECT_EQ(back.projection_parameters, info.projection_parameters);
    EXPECT_EQ(back.distortion_parameters, info.distortion_parameters);
    EXPECT_EQ(back.intrinsic_parameters, info.intrinsic_parameters);
  }
  info.distortion_parameters = {-0.0, 1e-300, 5e-324, -1.7976931348623157e308};
  camxiom::CamxiomCameraInfo back;
  ASSERT_EQ(
    camxiom::parseCamxiomCameraInfoYaml(camxiom::toCamxiomCameraInfoYaml(info), back),
    StatusCode::OK
  );
  EXPECT_EQ(back.distortion_parameters, info.distortion_parameters);
  EXPECT_TRUE(std::signbit(back.distortion_parameters[0]));
}

// A model written in camxiom's layout reads back as the same model with its
// image size, from text and from a file, in float and in double.
TEST(CamxiomCameraInfoYaml, ModelsRoundTripThroughTheFile)
{
  const camxiom::CameraModel model = camxiom::makeCameraModel(makeCamxiomInfo());
  ASSERT_EQ(camxiom::validateCameraModel(model), StatusCode::OK);
  camxiom::CamxiomCameraInfo info;
  info.image_width = 1280U;
  info.image_height = 960U;
  ASSERT_EQ(camxiom::exportCameraInfo(model, info), StatusCode::OK);
  const std::string path = writeTemp("");
  ASSERT_EQ(camxiom::saveCamxiomCameraInfoYaml(path, info), StatusCode::OK);

  camxiom::CameraModel back;
  camxiom::ImageSize size;
  ASSERT_EQ(camxiom::loadCameraModelYaml(path, back, size), StatusCode::OK);
  EXPECT_EQ(size.width, 1280);
  EXPECT_EQ(size.height, 960);
  EXPECT_EQ(back.projection.type, camxiom::ProjectionModelType::EQUIDISTANT);
  EXPECT_EQ(back.distortion.type, camxiom::DistortionModelType::KB4);
  EXPECT_EQ(back.distortion.coeffs, model.distortion.coeffs);
  EXPECT_EQ(back.intrinsics.fy, model.intrinsics.fy);

  camxiom::CameraModel64 back64;
  ASSERT_EQ(camxiom::loadCameraModelYaml64(path, back64, size), StatusCode::OK);
  // Double reads the decimal the float model was written as.
  EXPECT_EQ(back64.distortion.coeffs[3], info.distortion_parameters[3]);
  EXPECT_EQ(static_cast<float>(back64.distortion.coeffs[3]), model.distortion.coeffs[3]);
  std::remove(path.c_str());
}

// A ROS CameraInfo YAML, as written by toCameraInfoYaml or by hand with
// comments, reads back through the same entry point.
TEST(CamxiomCameraInfoYaml, ReadsRosCameraInfoFiles)
{
  CameraInfo info = makeTestInfo();
  CameraInfo back;
  ASSERT_EQ(
    camxiom::parseCameraInfoYaml(
      camxiom::toCameraInfoYaml(info, "front", camxiom::CameraInfoYamlPrecision::ROUND_TRIP), back
    ),
    StatusCode::OK
  );
  EXPECT_EQ(back.width, info.width);
  EXPECT_EQ(back.height, info.height);
  EXPECT_EQ(back.k, info.k);
  EXPECT_EQ(back.d, info.d);
  EXPECT_EQ(back.distortion_model, info.distortion_model);
  EXPECT_EQ(back.r, info.r);
  EXPECT_EQ(back.p, info.p);

  const std::string handwritten =
    "# calibrated 2026-10-28\n"
    "image_width: 640\n"
    "image_height: 480\n"
    "camera_name: \"left camera\"  # quoted\n"
    "camera_matrix:\n"
    "  rows: 3\n"
    "  cols: 3\n"
    "  data: [500.0, 0.0, 319.5,\n"
    "         0.0, 501.0, 239.5,\n"
    "         0.0, 0.0, 1.0]\n"
    "distortion_model: plumb_bob\n"
    "distortion_coefficients:\n"
    "  rows: 1\n"
    "  cols: 5\n"
    "  data:\n"
    "  - -0.2\n"
    "  - 0.05\n"
    "  - 0.001\n"
    "  - -0.001\n"
    "  - 0.0\n";
  camxiom::CameraModel model;
  camxiom::ImageSize size;
  ASSERT_EQ(camxiom::parseCameraModelYaml(handwritten, model, size), StatusCode::OK);
  EXPECT_EQ(size.width, 640);
  EXPECT_EQ(size.height, 480);
  EXPECT_EQ(model.projection.type, camxiom::ProjectionModelType::PINHOLE);
  EXPECT_EQ(model.distortion.type, camxiom::DistortionModelType::RADTAN5);
  EXPECT_FLOAT_EQ(model.intrinsics.fy, 501.0f);
  EXPECT_FLOAT_EQ(model.distortion.coeffs[0], -0.2f);
}

// The layouts do not mix, and anything that is not YAML or not the layout
// is refused rather than guessed at.
TEST(CamxiomCameraInfoYaml, RefusesMalformedDocuments)
{
  const std::string good = camxiom::toCamxiomCameraInfoYaml(makeCamxiomInfo());
  camxiom::CamxiomCameraInfo info;
  ASSERT_EQ(camxiom::parseCamxiomCameraInfoYaml(good, info), StatusCode::OK);
  CameraInfo ros;
  EXPECT_EQ(camxiom::parseCameraInfoYaml(good, ros), StatusCode::INVALID_INPUT);
  EXPECT_EQ(
    camxiom::parseCamxiomCameraInfoYaml(camxiom::toCameraInfoYaml(makeTestInfo()), info),
    StatusCode::INVALID_INPUT
  );

  const auto replaced = [&good](const std::string &from, const std::string &to) {
    std::string text = good;
    const std::size_t at = text.find(from);
    EXPECT_NE(at, std::string::npos) << from;
    return text.replace(at, from.size(), to);
  };
  const std::vector<std::pair<const char *, std::string>> bad{
    {"missing key", replaced("image_height: 960\n", "")},
    {"extra key", good + "camera_name: front\n"},
    {"duplicate key", good + "image_width: 1280\n"},
    {"four intrinsics", replaced("479.5, 0.0]", "479.5]")},
    {"quoted number", replaced("[320.0,", "[\"320.0\",")},
    {"word for a number", replaced("[320.0,", "[fx,")},
    {"flow mapping", replaced("projection_parameters: []", "projection_parameters: {}")},
    {"unclosed sequence", replaced("479.5, 0.0]", "479.5, 0.0")},
    {"negative size", replaced("image_width: 1280", "image_width: -1280")},
    {"size beyond 32 bits", replaced("image_width: 1280", "image_width: 4294967296")},
    {"tab indentation", replaced("image_width: 1280\n", "image_width: 1280\n\timage_height: 9\n")},
    {"second document", good + "---\nimage_width: 1\n"},
    {"not a mapping", std::string("- 1\n- 2\n")},
    {"empty", std::string("")},
  };
  for (const auto &[what, text] : bad)
  {
    EXPECT_EQ(camxiom::parseCamxiomCameraInfoYaml(text, info), StatusCode::INVALID_INPUT) << what;
    camxiom::CameraModel model;
    camxiom::ImageSize size;
    EXPECT_EQ(camxiom::parseCameraModelYaml(text, model, size), StatusCode::INVALID_INPUT) << what;
  }

  // An anchor is plain YAML and does not change the value it marks.
  ASSERT_EQ(
    camxiom::parseCamxiomCameraInfoYaml(replaced("kb4\n", "&a kb4\n"), info), StatusCode::OK
  );
  EXPECT_EQ(info.distortion_model, "kb4");

  // Well-formed, but not a valid model: the validation status, not OK.
  camxiom::CameraModel model;
  camxiom::ImageSize size;
  EXPECT_EQ(
    camxiom::parseCameraModelYaml(
      replaced("distortion_model: kb4", "distortion_model: radtan4"), model, size
    ),
    StatusCode::INVALID_MODEL
  );
  EXPECT_EQ(
    camxiom::loadCameraModelYaml("/nonexistent/dir/camera.yaml", model, size),
    StatusCode::INVALID_INPUT
  );
}

// strtod follows the C locale's decimal point; reading must not.
TEST(CamxiomCameraInfoYaml, ReadingIgnoresACommaDecimalCLocale)
{
  const char *previous = std::setlocale(LC_NUMERIC, nullptr);
  const std::string saved = previous != nullptr ? previous : "C";
  if (std::setlocale(LC_NUMERIC, "de_DE.UTF-8") == nullptr && std::setlocale(LC_NUMERIC, "de_DE") == nullptr)
  {
    GTEST_SKIP() << "no comma-decimal locale installed";
  }
  camxiom::CamxiomCameraInfo info;
  const StatusCode status =
    camxiom::parseCamxiomCameraInfoYaml(camxiom::toCamxiomCameraInfoYaml(makeCamxiomInfo()), info);
  std::setlocale(LC_NUMERIC, saved.c_str());
  ASSERT_EQ(status, StatusCode::OK);
  EXPECT_EQ(info.intrinsic_parameters[1], 320.5);
  EXPECT_EQ(info.distortion_parameters[3], 1.0 / 3.0);
}

// The ROS writer stores binning and ROI; the reader folds them into K, P and
// the image size as the ROS layer does for a message (binning 2, ROI offset
// (100, 50), ROI 640 x 480 of a 1280 x 960 sensor).
TEST(CamxiomCameraInfoYaml, FoldsRosBinningAndRoiLikeTheMessagePath)
{
  const std::string yaml =
    "image_width: 1280\n"
    "image_height: 960\n"
    "camera_name: binned\n"
    "camera_matrix:\n"
    "  rows: 3\n"
    "  cols: 3\n"
    "  data: [800.0, 0.0, 640.0, 0.0, 800.0, 480.0, 0.0, 0.0, 1.0]\n"
    "distortion_model: plumb_bob\n"
    "distortion_coefficients:\n"
    "  rows: 1\n"
    "  cols: 5\n"
    "  data: [0.0, 0.0, 0.0, 0.0, 0.0]\n"
    "binning_x: 2\n"
    "binning_y: 2\n"
    "roi:\n"
    "  x_offset: 100\n"
    "  y_offset: 50\n"
    "  height: 480\n"
    "  width: 640\n"
    "  do_rectify: false\n";
  CameraInfo info;
  ASSERT_EQ(camxiom::parseCameraInfoYaml(yaml, info), StatusCode::OK);
  EXPECT_EQ(info.k[0], 400.0);
  EXPECT_EQ(info.k[2], 270.0);  // (640 - 100) / 2
  EXPECT_EQ(info.k[4], 400.0);
  EXPECT_EQ(info.k[5], 215.0);  // (480 - 50) / 2
  EXPECT_EQ(info.width, 320U);
  EXPECT_EQ(info.height, 240U);

  camxiom::CameraModel model;
  camxiom::ImageSize size;
  ASSERT_EQ(camxiom::parseCameraModelYaml(yaml, model, size), StatusCode::OK);
  EXPECT_FLOAT_EQ(model.intrinsics.fx, 400.0f);
  EXPECT_FLOAT_EQ(model.intrinsics.cx, 270.0f);
  EXPECT_EQ(size.width, 320);
  EXPECT_EQ(size.height, 240);

  // The writer's file is in the image's own frame and reads back unchanged.
  CameraInfo back;
  ASSERT_EQ(
    camxiom::parseCameraInfoYaml(
      camxiom::toCameraInfoYaml(info, "binned", camxiom::CameraInfoYamlPrecision::ROUND_TRIP), back
    ),
    StatusCode::OK
  );
  EXPECT_EQ(back.k, info.k);
  EXPECT_EQ(back.width, info.width);
}

// A block sequence at its key's own indentation may be followed by more keys.
TEST(CamxiomCameraInfoYaml, ReadsKeysAfterABlockSequenceAtTheKeyIndent)
{
  const std::string yaml =
    "image_width: 640\n"
    "image_height: 480\n"
    "projection_model: double_sphere\n"
    "projection_parameters:\n"
    "- -0.2\n"
    "- 0.6\n"
    "distortion_model: none\n"
    "distortion_parameters: []\n"
    "intrinsic_parameters: [400, 400, 320, 240, 0]\n";
  camxiom::CamxiomCameraInfo info;
  ASSERT_EQ(camxiom::parseCamxiomCameraInfoYaml(yaml, info), StatusCode::OK);
  EXPECT_EQ(info.projection_parameters, (std::vector<double>{-0.2, 0.6}));
  EXPECT_EQ(info.distortion_model, "none");
}

// A quote is a quoting character only where a scalar starts; inside a plain
// scalar it is an ordinary character (camera_name: driver's_camera is valid
// YAML), also before a comment, while a quoted scalar keeps its quotes.
TEST(CamxiomCameraInfoYaml, ReadsQuotesInsidePlainScalars)
{
  const std::string ros =
    "image_width: 640\n"
    "image_height: 480\n"
    "camera_name: driver's_camera  # the left one\n"
    "camera_matrix:\n"
    "  rows: 3\n"
    "  cols: 3\n"
    "  data: [500.0, 0.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.0, 1.0]\n"
    "distortion_model: plumb_bob\n"
    "distortion_coefficients:\n"
    "  rows: 1\n"
    "  cols: 5\n"
    "  data: [-0.2, 0.05, 0.0, 0.0, 0.0]\n";
  CameraInfo info;
  EXPECT_EQ(camxiom::parseCameraInfoYaml(ros, info), StatusCode::OK);
  std::string no_comment = ros;
  no_comment.replace(no_comment.find("  # the left one"), 16, "");
  EXPECT_EQ(camxiom::parseCameraInfoYaml(no_comment, info), StatusCode::OK);

  std::string quoted = ros;
  quoted.replace(quoted.find("driver's_camera"), 15, "'driver''s camera'");
  EXPECT_EQ(camxiom::parseCameraInfoYaml(quoted, info), StatusCode::OK);
  // '' inside single quotes is a quote, so the scalar goes on past it.
  std::string escaped = ros;
  escaped.replace(escaped.find("driver's_camera"), 15, "'driver''s camera #1'");
  EXPECT_EQ(camxiom::parseCameraInfoYaml(escaped, info), StatusCode::OK);
  // The same in a flow sequence (under a key the ROS reader ignores): the
  // bracket inside 'it''s ]' is part of the string, not the closing one.
  EXPECT_EQ(camxiom::parseCameraInfoYaml(ros + "extra: ['it''s ]', 1]\n", info), StatusCode::OK);
  std::string unterminated = ros;
  unterminated.replace(unterminated.find("driver's_camera"), 15, "'driver");
  EXPECT_EQ(camxiom::parseCameraInfoYaml(unterminated, info), StatusCode::INVALID_INPUT);
}

// A stream that cannot report its size (a pipe) is read up to the limit
// rather than refused.
TEST(CamxiomCameraInfoYaml, LoadsFromAPipe)
{
  char dir[] = "/tmp/camxiom_yaml_fifo_XXXXXX";
  ASSERT_NE(::mkdtemp(dir), nullptr);
  const std::string path = std::string(dir) + "/camera.yaml";
  ASSERT_EQ(::mkfifo(path.c_str(), 0600), 0);
  const std::string yaml = camxiom::toCamxiomCameraInfoYaml(makeCamxiomInfo());
  std::thread writer([&] {
    std::ofstream out(path, std::ios::out | std::ios::binary);
    out << yaml;
  });
  camxiom::CameraModel model;
  camxiom::ImageSize size;
  const StatusCode status = camxiom::loadCameraModelYaml(path, model, size);
  writer.join();
  ::unlink(path.c_str());
  ::rmdir(dir);
  ASSERT_EQ(status, StatusCode::OK);
  EXPECT_EQ(size.width, 1280);
  EXPECT_EQ(model.distortion.type, camxiom::DistortionModelType::KB4);
}

// Content after the end of the document, and tab indentation, are refused on
// their own account (no duplicate key involved).
TEST(CamxiomCameraInfoYaml, RefusesASecondDocumentAndTabIndentation)
{
  const std::string good = camxiom::toCamxiomCameraInfoYaml(makeCamxiomInfo());
  camxiom::CamxiomCameraInfo info;
  ASSERT_EQ(camxiom::parseCamxiomCameraInfoYaml(good + "...\n", info), StatusCode::OK);
  ASSERT_EQ(camxiom::parseCamxiomCameraInfoYaml("---\n" + good, info), StatusCode::OK);
  EXPECT_EQ(
    camxiom::parseCamxiomCameraInfoYaml(good + "...\n---\n" + good, info), StatusCode::INVALID_INPUT
  );
  EXPECT_EQ(
    camxiom::parseCamxiomCameraInfoYaml(good + "...\nbroken: [\n", info), StatusCode::INVALID_INPUT
  );
  std::string tabbed = good;
  tabbed.replace(tabbed.find("image_height"), 0, "\t");
  EXPECT_EQ(camxiom::parseCamxiomCameraInfoYaml(tabbed, info), StatusCode::INVALID_INPUT);
  std::string spaced_tab = good;
  spaced_tab.replace(spaced_tab.find("image_height"), 0, " \t");
  EXPECT_EQ(camxiom::parseCamxiomCameraInfoYaml(spaced_tab, info), StatusCode::INVALID_INPUT);
}

// Hostile input is refused quickly: deep nesting, a long unterminated flow
// sequence, very many items or keys, aliases that expand without bound, and
// input past the size limit. Each is
// added to a valid ROS document under keys the ROS reader ignores, so only
// the reader's limits refuse it.
TEST(CamxiomCameraInfoYaml, RefusesHostileInputQuickly)
{
  const std::string base = camxiom::toCameraInfoYaml(makeTestInfo());
  CameraInfo ros;
  ASSERT_EQ(camxiom::parseCameraInfoYaml(base + "x:\n y:\n  z: 1\n", ros), StatusCode::OK);
  // Collections nest at most 16 deep (the root mapping is the first).
  ASSERT_EQ(
    camxiom::parseCameraInfoYaml(
      base + "x: " + std::string(15U, '[') + std::string(15U, ']') + "\n", ros
    ),
    StatusCode::OK
  );
  const std::string past_depth =
    base + "x: " + std::string(16U, '[') + std::string(16U, ']') + "\n";

  // About 1400 levels fit under the 1 MiB size limit.
  std::string deep = base;
  for (int level = 0; level < 1400; ++level)
  {
    deep += std::string(static_cast<std::size_t>(level), ' ') + "x:\n";
  }
  ASSERT_LT(deep.size(), std::size_t{1} << 20U);
  const std::string deep_flow =
    base + "x: " + std::string(400000U, '[') + std::string(400000U, ']') + "\n";
  ASSERT_LT(deep_flow.size(), std::size_t{1} << 20U);
  std::string unterminated = base + "x: [\n";
  for (int i = 0; i < 100000; ++i)
  {
    unterminated += "0,\n";
  }
  std::string many_items = base + "x: [0";
  for (int i = 0; i < 10000; ++i)
  {
    many_items += ", 0";
  }
  many_items += "]\n";
  std::string many_keys = base;
  for (int i = 0; i < 40000; ++i)
  {
    many_keys += "k" + std::to_string(i) + ": 0\n";
  }
  // Aliases that expand nine levels of ten-fold repeats (10^9 nodes), and an
  // alias inside the collection it names.
  std::string aliases = base + "l0: &l0 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n";
  for (int level = 1; level <= 9; ++level)
  {
    const std::string below = "*l" + std::to_string(level - 1);
    aliases += "l" + std::to_string(level) + ": &l" + std::to_string(level) + " [" + below;
    for (int i = 1; i < 10; ++i)
    {
      aliases += ", " + below;
    }
    aliases += "]\n";
  }
  const std::string cycle = base + "x: &x [*x]\n";
  std::string huge = base;
  while (huge.size() <= (std::size_t{1} << 20U))
  {
    huge += "# padding padding padding padding padding padding padding padding\n";
  }
  for (const auto &[what, text] : std::vector<std::pair<const char *, std::string>>{
         {"nesting past the limit", past_depth},
         {"deep nesting", deep},
         {"deep flow nesting", deep_flow},
         {"unterminated sequence", unterminated},
         {"many items", many_items},
         {"many keys", many_keys},
         {"expanding aliases", aliases},
         {"alias cycle", cycle},
         {"past the size limit", huge}})
  {
    const auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(camxiom::parseCameraInfoYaml(text, ros), StatusCode::INVALID_INPUT) << what;
    const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    EXPECT_LT(seconds, 5.0) << what;
  }

  // A file past the limit is refused before it is read.
  const std::string path = writeTemp(huge);
  camxiom::CameraModel model;
  camxiom::ImageSize size;
  EXPECT_EQ(camxiom::loadCameraModelYaml(path, model, size), StatusCode::INVALID_INPUT);
  std::remove(path.c_str());
}
