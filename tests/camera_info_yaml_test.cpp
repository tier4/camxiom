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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <locale>
#include <random>
#include <sstream>
#include <string>

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
  const auto yaml = camxiom::toCameraInfoYaml(info, camxiom::CameraInfoYamlPrecision::ROUND_TRIP);
  expectExactValues(parseNumericSequence(yaml, "camera_matrix"), info.k);
  expectExactValues(parseNumericSequence(yaml, "distortion_coefficients"), info.d);
  expectExactValues(parseNumericSequence(yaml, "rectification_matrix"), info.r);
  expectExactValues(parseNumericSequence(yaml, "projection_matrix"), info.p);
  EXPECT_EQ(
    camxiom::toCameraInfoYaml(info, camxiom::CameraInfoYamlPrecision::FIXED),
    camxiom::toCameraInfoYaml(info)
  );
}

TEST(CameraInfoYaml, RoundTripModeIsLocaleIndependentAndSavesExactText)
{
  auto info = makeTestInfo();
  info.k[0] = 513.1234567890123;
  const auto precision = camxiom::CameraInfoYamlPrecision::ROUND_TRIP;
  const auto reference = camxiom::toCameraInfoYaml(info, precision, "cam0");
  const auto old_locale = std::locale();
  std::locale::global(std::locale(old_locale, new CommaDecimalPunct));
  const auto actual = camxiom::toCameraInfoYaml(info, precision, "cam0");
  std::locale::global(old_locale);
  EXPECT_EQ(actual, reference);
  const auto path = testing::TempDir() + "/camxiom_camera_info_round_trip_test.yaml";
  ASSERT_EQ(camxiom::saveCameraInfoYaml(path, info, precision, "cam0"), StatusCode::OK);
  std::ifstream file(path);
  ASSERT_TRUE(file.is_open());
  std::ostringstream contents;
  contents << file.rdbuf();
  EXPECT_EQ(contents.str(), reference);
  std::remove(path.c_str());
  EXPECT_EQ(
    camxiom::saveCameraInfoYaml("/nonexistent-dir-for-camxiom-test/x.yaml", info, precision),
    StatusCode::INVALID_INPUT
  );
}

TEST(CameraInfoYaml, RoundTripModeUsesYamlNonFiniteSpellings)
{
  auto info = makeTestInfo();
  info.d = {
    std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity()};
  const auto yaml = camxiom::toCameraInfoYaml(info, camxiom::CameraInfoYamlPrecision::ROUND_TRIP);
  EXPECT_NE(yaml.find("data: [.nan, .inf, -.inf]"), std::string::npos);
}
