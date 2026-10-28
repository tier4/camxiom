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

#include "remap_rectify.hpp"

#include "camxiom/internal/constants.hpp"
#include "camxiom/projection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

// Scale fitting for makeRectifiedOutputModel(). The output camera is the
// template at scale s; a larger s is a narrower view. With q the output's
// normalized coordinates of a ray (the template at s = 1), the ray lands
// inside the output image exactly when s * |q|_K <= 1, where |q|_K is the
// larger of its two offsets from the image centre, each in units of half
// the image size (the distance from the centre to the outermost pixel
// centres). So
//   alpha = 1: s = 1 / the largest |q|_K over the source rays kept, and
//   alpha = 0: s = 1 / the smallest |q|_K over the rays the output can show
//              but the source does not.
// Both extremes are found by branch and bound over the sphere of directions:
// interval enclosures of a patch of directions' source pixel, validity and q
// bound the extreme over the patch; patches that cannot hold it are dropped
// and the rest split, until the bound is within kTolerance of a value an
// actual ray attains. The bound itself is returned; it is safe for the model
// formulas evaluated in double, up to floating-point rounding (the returned
// scale keeps headroom for that, enough for ordinary models; near-singular
// parameters are outside it).

namespace camxiom::detail
{
namespace
{

// Rays this close (radians) to where the output projection's coordinates
// grow without bound (the pinhole horizon, the stereographic antipode, the
// cylinder's poles, the end of the sphere models' domain) or become singular
// (the antipode of the equidistant and equisolid laws, which they draw as a
// whole circle) are not shown in two cases: alpha = 1 does not try to
// contain them, and alpha = 0 stops there when the source covers the output
// all the way to that edge (no widest view exists then).
constexpr float kHorizonMargin = 0.05f;
// Extra narrowing of the alpha = 0 view, and widening of the alpha = 1 view,
// for float rounding of the returned model.
constexpr float kInscribedSafety = 1.0001f;
constexpr float kContainedSafety = 1.0001f;
// Relative gap between the bound returned and a value attained by a ray.
constexpr double kTolerance = 2e-4;
// Patches narrower than this (radians) are not split further.
constexpr double kSmallestPatch = 1e-13;
// Lat-long rays this close to a pole are not certain to project.
constexpr double kPoleSlack = 1e-5;
constexpr int kConeBisectionSteps = 48;

// ===========================================================================
// Interval arithmetic
// ===========================================================================

/// A closed interval. Every operation widens each end of its result by a
/// few units in its last place (a floating-point result is off by at most
/// half a unit of its own last place), so a result holds every value its
/// operands' values can produce. (Point evaluations of the model formulas
/// that feed it are rounded, not enclosed.)
struct Interval
{
  double lo;
  double hi;
};

Interval widened(const double lo, const double hi)
{
  constexpr double kUlps = 8.0 * std::numeric_limits<double>::epsilon();
  constexpr double kTiny = std::numeric_limits<double>::min();
  return Interval{lo - kUlps * std::abs(lo) - kTiny, hi + kUlps * std::abs(hi) + kTiny};
}

// Terms that are exactly zero (a coefficient a model does not use) are
// kept exact: widening them would only feed subnormal numbers, which are
// slow, into the arithmetic.
bool isZero(const Interval a) { return a.lo == 0.0 && a.hi == 0.0; }

Interval operator+(const Interval a, const Interval b)
{
  if (isZero(b) || isZero(a))
  {
    return isZero(b) ? a : b;
  }
  return widened(a.lo + b.lo, a.hi + b.hi);
}

Interval operator+(const Interval a, const double s)
{
  return s == 0.0 ? a : widened(a.lo + s, a.hi + s);
}

Interval operator-(const Interval a, const Interval b) { return widened(a.lo - b.hi, a.hi - b.lo); }

Interval operator*(const Interval a, const Interval b)
{
  if (isZero(a) || isZero(b))
  {
    return Interval{0.0, 0.0};
  }
  if (a.lo >= 0.0 && b.lo >= 0.0)
  {
    return widened(a.lo * b.lo, a.hi * b.hi);
  }
  const double p1 = a.lo * b.lo;
  const double p2 = a.lo * b.hi;
  const double p3 = a.hi * b.lo;
  const double p4 = a.hi * b.hi;
  return widened(std::min({p1, p2, p3, p4}), std::max({p1, p2, p3, p4}));
}

Interval operator*(const Interval a, const double s)
{
  if (s == 0.0)
  {
    return Interval{0.0, 0.0};
  }
  return s > 0.0 ? widened(a.lo * s, a.hi * s) : widened(a.hi * s, a.lo * s);
}

Interval square(const Interval a)
{
  if (a.lo >= 0.0)
  {
    return widened(a.lo * a.lo, a.hi * a.hi);
  }
  if (a.hi <= 0.0)
  {
    return widened(a.hi * a.hi, a.lo * a.lo);
  }
  return widened(0.0, std::max(a.lo * a.lo, a.hi * a.hi));
}

/// a / b over the values of b that are positive, for 0 in b and b_hi > 0:
/// a times [1 / b_hi, infinity), unbounded on one side or both.
Interval overPositivePart(const Interval a, const double b_hi)
{
  constexpr double kInfinity = std::numeric_limits<double>::infinity();
  const Interval f = widened(1.0 / b_hi, 1.0 / b_hi);
  if (a.lo >= 0.0)
  {
    return Interval{a.lo == 0.0 ? 0.0 : (Interval{a.lo, a.lo} * f).lo, kInfinity};
  }
  if (a.hi <= 0.0)
  {
    return Interval{-kInfinity, a.hi == 0.0 ? 0.0 : (Interval{a.hi, a.hi} * f).hi};
  }
  return Interval{-kInfinity, kInfinity};
}

/// a / b; nullopt when b may be zero.
std::optional<Interval> divide(const Interval a, const Interval b)
{
  if (!(b.lo > 0.0 || b.hi < 0.0))
  {
    return std::nullopt;
  }
  return a * widened(1.0 / b.hi, 1.0 / b.lo);
}

Interval squareRoot(const Interval a)
{
  return widened(std::sqrt(std::max(a.lo, 0.0)), std::sqrt(std::max(a.hi, 0.0)));
}

Interval negated(const Interval a) { return Interval{-a.hi, -a.lo}; }

Interval clamped(const Interval a, const double lo, const double hi)
{
  return Interval{std::clamp(a.lo, lo, hi), std::clamp(a.hi, lo, hi)};
}

/// Whether c + 2 pi n lies in [lo, hi] for some integer n.
bool holdsPeriodic(const double c, const double lo, const double hi)
{
  const double period = 2.0 * constants::kPi;
  return c + period * std::ceil((lo - c) / period) <= hi;
}

Interval cosine(const Interval a)
{
  if (a.hi - a.lo >= 2.0 * constants::kPi)
  {
    return Interval{-1.0, 1.0};
  }
  double lo = std::min(std::cos(a.lo), std::cos(a.hi));
  double hi = std::max(std::cos(a.lo), std::cos(a.hi));
  if (holdsPeriodic(0.0, a.lo, a.hi))
  {
    hi = 1.0;
  }
  if (holdsPeriodic(constants::kPi, a.lo, a.hi))
  {
    lo = -1.0;
  }
  return clamped(widened(lo, hi), -1.0, 1.0);
}

Interval sine(const Interval a) { return cosine(a + (-0.5 * constants::kPi)); }

double maxAbs(const Interval a) { return std::max(std::abs(a.lo), std::abs(a.hi)); }

/// The common part of two enclosures of the same values.
Interval intersection(const Interval a, const Interval b)
{
  const Interval c{std::max(a.lo, b.lo), std::min(a.hi, b.hi)};
  return c.lo <= c.hi ? c : a;
}

double minAbs(const Interval a) { return a.lo > 0.0 ? a.lo : (a.hi < 0.0 ? -a.hi : 0.0); }

/// Three-valued answer for a patch of directions: none, some, all.
enum class Tri : std::uint8_t { NO, MAYBE, YES };

Tri both(const Tri a, const Tri b)
{
  if (a == Tri::NO || b == Tri::NO)
  {
    return Tri::NO;
  }
  return a == Tri::YES && b == Tri::YES ? Tri::YES : Tri::MAYBE;
}

Tri insideRange(const Interval a, const double lo, const double hi)
{
  if (a.lo >= lo && a.hi <= hi)
  {
    return Tri::YES;
  }
  return a.hi < lo || a.lo > hi ? Tri::NO : Tri::MAYBE;
}

/// How a patch of directions is described: angles from the source axis and
/// azimuths; a box of the source's undistorted normalized plane (a pinhole
/// source's front half, which this chart describes without trigonometric
/// overestimation); or longitudes and latitudes.
enum class Chart : std::uint8_t { POLAR, PLANE, LAT_LONG };

// ===========================================================================
// Scalar helpers
// ===========================================================================

Eigen::Vector3f rayAt(const double theta, const double azimuth)
{
  return Eigen::Vector3d(
           std::sin(theta) * std::cos(azimuth), std::sin(theta) * std::sin(azimuth), std::cos(theta)
  )
    .cast<float>();
}

/// Whether `ray` (output frame) can be shown by `output` at least `margin`
/// away from where its coordinates run off to infinity or become singular.
bool representable(const CameraModel &output, const Eigen::Vector3f &ray, const float margin)
{
  if (rayToPixel(output, ray).status != StatusCode::OK)
  {
    return false;
  }
  const Eigen::Vector3f r = ray.normalized();
  switch (output.projection.type)
  {
    case ProjectionModelType::PINHOLE:
      return r.z() >= std::sin(margin);
    case ProjectionModelType::STEREOGRAPHIC:
    case ProjectionModelType::EQUIDISTANT:
    case ProjectionModelType::EQUISOLID:
      return r.z() >= -std::cos(margin);
    case ProjectionModelType::CYLINDRICAL:
      return std::abs(r.y()) <= std::cos(margin);
    case ProjectionModelType::OMNIDIRECTIONAL:
    case ProjectionModelType::DOUBLE_SPHERE:
    case ProjectionModelType::EUCM: {
      // Their domain ends at a cone that depends on the parameters: test the
      // ray tilted `margin` further out.
      const float rho = std::sqrt(r.x() * r.x() + r.y() * r.y());
      const float tilted = std::min(std::atan2(rho, r.z()) + margin, constants::kPiF);
      const float c = rho > 0.0f ? r.x() / rho : 1.0f;
      const float s = rho > 0.0f ? r.y() / rho : 0.0f;
      const Eigen::Vector3f probe(std::sin(tilted) * c, std::sin(tilted) * s, std::cos(tilted));
      return rayToPixel(output, probe).status == StatusCode::OK;
    }
    case ProjectionModelType::ORTHOGRAPHIC:
    case ProjectionModelType::EQUIRECTANGULAR:
      return true;
    case ProjectionModelType::UNKNOWN:
      break;
  }
  return false;
}

/// The largest angle from the axis for which `inside(theta)` still holds
/// along one azimuth, assuming it holds on the axis and the domain is a
/// cone. A domain reaching the back axis (where the direction itself is
/// singular) ends a hair short of it.
template <typename Inside>
std::optional<double> coneLimit(const Inside &inside)
{
  if (!inside(0.0))
  {
    return std::nullopt;
  }
  double lo = 0.0;
  double hi = constants::kPi - 1e-4;
  if (inside(hi))
  {
    return hi;
  }
  for (int step = 0; step < kConeBisectionSteps; ++step)
  {
    const double mid = 0.5 * (lo + hi);
    (inside(mid) ? lo : hi) = mid;
  }
  return lo;
}

// ===========================================================================
// Polynomials and certified roots
// ===========================================================================

/// Univariate polynomial, ascending powers.
using Poly = std::vector<double>;

Poly operator+(const Poly &a, const Poly &b)
{
  Poly r(std::max(a.size(), b.size()), 0.0);
  for (std::size_t i = 0; i < a.size(); ++i) r[i] += a[i];
  for (std::size_t i = 0; i < b.size(); ++i) r[i] += b[i];
  return r;
}

Poly operator-(const Poly &a, const Poly &b)
{
  Poly r(std::max(a.size(), b.size()), 0.0);
  for (std::size_t i = 0; i < a.size(); ++i) r[i] += a[i];
  for (std::size_t i = 0; i < b.size(); ++i) r[i] -= b[i];
  return r;
}

Poly operator*(Poly p, const double s)
{
  for (double &c : p) c *= s;
  return p;
}

Poly operator*(const Poly &a, const Poly &b)
{
  Poly r(a.size() + b.size() - 1, 0.0);
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    for (std::size_t j = 0; j < b.size(); ++j) r[i + j] += a[i] * b[j];
  }
  return r;
}

double evaluate(const Poly &p, const double t)
{
  double value = 0.0;
  for (std::size_t i = p.size(); i-- > 0;)
  {
    value = value * t + p[i];
  }
  return value;
}

/// Sum of |c_i| t^i: bounds the rounding error of evaluate() at t.
double magnitude(const Poly &p, const double t)
{
  double value = 0.0;
  for (std::size_t i = p.size(); i-- > 0;)
  {
    value = value * t + std::abs(p[i]);
  }
  return value;
}

// Binomial coefficients for the Bernstein conversion; the conditions below
// stay under degree 32.
constexpr int kMaxDegree = 40;

struct BinomialTable
{
  constexpr BinomialTable()
  {
    for (int n = 0; n <= kMaxDegree; ++n)
    {
      c[n][0] = 1.0;
      for (int k = 1; k <= n; ++k) c[n][k] = c[n - 1][k - 1] + c[n - 1][k];
      for (int k = 0; k <= n; ++k) inverse[n][k] = 1.0 / c[n][k];
    }
  }
  double c[kMaxDegree + 1][kMaxDegree + 1] = {};
  double inverse[kMaxDegree + 1][kMaxDegree + 1] = {};
};

// Built at compile time: nothing is initialized on first use.
constexpr BinomialTable kBinomials{};

/// Bernstein coefficients of `p` on [a, b], a sub-interval of [0, 1] (where
/// the shift stays well conditioned), and a bound on their rounding error.
/// They bound the polynomial on [a, b]: all positive means p > 0 there.
void bernstein(const Poly &p, const double a, const double b, Poly &out, double &error)
{
  const int n = static_cast<int>(p.size()) - 1;
  double q[kMaxDegree + 1];
  std::copy(p.begin(), p.end(), q);
  for (int i = 0; i < n; ++i)  // Taylor shift: q(u) = p(a + u)
  {
    for (int j = n - 1; j >= i; --j)
    {
      q[j] += a * q[j + 1];
    }
  }
  const BinomialTable &table = kBinomials;
  double scale = 0.0;
  double h_power = 1.0;
  for (int k = 0; k <= n; ++k)
  {
    q[k] *= h_power;
    scale += std::abs(q[k]);
    q[k] *= table.inverse[n][k];
    h_power *= b - a;
  }
  out.assign(static_cast<std::size_t>(n + 1), 0.0);
  for (int k = 0; k <= n; ++k)
  {
    double sum = 0.0;
    for (int i = 0; i <= k; ++i)
    {
      sum += table.c[k][i] * q[i];
    }
    out[static_cast<std::size_t>(k)] = sum;
  }
  error = 1e-12 * std::max(scale, magnitude(p, b));
}

/// The sign of p(t) when its rounding cannot change it, else 0.
int certainSign(const Poly &p, const double t)
{
  const double value = evaluate(p, t);
  const double error = 1e-12 * magnitude(p, t);
  return value > error ? 1 : (value < -error ? -1 : 0);
}

/// Whether Bernstein coefficients, each of certain sign, change sign once:
/// then the polynomial has exactly one root (a simple one) on their
/// interval.
bool oneSignChange(const Poly &coeffs, const double error)
{
  int changes = 0;
  for (std::size_t i = 0; i < coeffs.size(); ++i)
  {
    if (!(std::abs(coeffs[i]) > error))
    {
      return false;
    }
    changes += i > 0 && (coeffs[i] > 0.0) != (coeffs[i - 1] > 0.0) ? 1 : 0;
  }
  return changes == 1;
}

/// Searching [a, b] (a sub-interval of [0, 1]) from its near end (a, or b
/// when `from_right`), the first point where `p` is not certified positive:
/// the near end of the first sub-interval at most `tol` wide whose Bernstein
/// coefficients are not all positive. Everything between the near end of
/// [a, b] and that point is certified positive. nullopt when `p` is
/// certified positive on all of [a, b]. A sub-interval holding exactly one
/// simple root (its coefficients change sign once) is narrowed by
/// bisection on the sign of p, which certifies the same way.
std::optional<double> firstNotPositive(
  const Poly &p, const double a, const double b, const double tol, const bool from_right
)
{
  std::vector<std::pair<double, double>> stack{{a, b}};
  Poly coeffs;
  while (!stack.empty())
  {
    auto [lo, hi] = stack.back();
    stack.pop_back();
    double error = 0.0;
    bernstein(p, lo, hi, coeffs, error);
    if (*std::min_element(coeffs.begin(), coeffs.end()) > error)
    {
      continue;
    }
    const double near_value = from_right ? coeffs.back() : coeffs.front();
    if (near_value <= error || hi - lo <= tol)
    {
      return from_right ? hi : lo;
    }
    if (oneSignChange(coeffs, error))
    {
      // p is positive from the near end up to the root and negative past it.
      while (hi - lo > tol)
      {
        const double mid = 0.5 * (lo + hi);
        const int sign = certainSign(p, mid);
        if (sign == 0)
        {
          break;
        }
        ((sign > 0) == from_right ? hi : lo) = mid;
      }
      return from_right ? hi : lo;
    }
    const double mid = 0.5 * (lo + hi);
    if (from_right)  // The near half is examined first.
    {
      stack.emplace_back(lo, mid);
      stack.emplace_back(mid, hi);
    }
    else
    {
      stack.emplace_back(mid, hi);
      stack.emplace_back(lo, mid);
    }
  }
  return std::nullopt;
}

/// A point t in [t0, t1] (0 <= t0 < t1, t1 may be infinite) such that `p`
/// is certified positive on [t0, t); nullopt when it is certified positive
/// on all of [t0, t1]. The search stops at a sub-interval narrower than
/// 1e-12 that it cannot certify, or whose near end is not certified
/// positive, so t is a safe lower bound on the first root, not the root
/// itself: where `p` is flat near a root (a double root) it can sit well
/// below it. Beyond 1 the search runs on the reversed polynomial
/// u^n p(1 / u) (same sign for u > 0), so every Bernstein expansion stays
/// on a sub-interval of [0, 1].
std::optional<double> firstNotPositiveIn(Poly p, const double t0, const double t1)
{
  while (p.size() > 1 && p.back() == 0.0)
  {
    p.pop_back();
  }
  if (p.size() > static_cast<std::size_t>(kMaxDegree + 1))
  {
    return t0;  // Out of the table's range: certify nothing.
  }
  constexpr double kTol = 1e-12;
  if (t0 < 1.0)
  {
    if (const auto t = firstNotPositive(p, t0, std::min(t1, 1.0), kTol, false))
    {
      return t;
    }
    if (t1 <= 1.0)
    {
      return std::nullopt;
    }
  }
  const Poly reversed(p.rbegin(), p.rend());
  // t in [max(t0, 1), t1] is u = 1 / t in [1 / t1, 1 / max(t0, 1)]; the
  // first t is the largest u.
  if (const auto u = firstNotPositive(reversed, 1.0 / t1, 1.0 / std::max(t0, 1.0), kTol, true))
  {
    return 1.0 / *u;
  }
  return std::nullopt;
}

/// Radius up to which `p` (positive at 0) is certified positive; `cap` (or
/// infinity) when no root is found up to it.
double positiveUpTo(const Poly &p, const double cap)
{
  const auto t = firstNotPositiveIn(p, 0.0, cap);
  return t ? *t : cap;
}

/// Bivariate polynomial sum c(a, b) x^a y^b, total degree <= degree().
class Poly2
{
public:
  explicit Poly2(const int degree = 0)
  : degree_(degree), c_(static_cast<std::size_t>((degree + 1) * (degree + 1)), 0.0)
  {
  }

  static Poly2 constant(const double value)
  {
    Poly2 p(0);
    p.at(0, 0) = value;
    return p;
  }
  static Poly2 x()
  {
    Poly2 p(1);
    p.at(1, 0) = 1.0;
    return p;
  }
  static Poly2 y()
  {
    Poly2 p(1);
    p.at(0, 1) = 1.0;
    return p;
  }

  int degree() const { return degree_; }
  double &at(const int a, const int b)
  {
    return c_[static_cast<std::size_t>(a * (degree_ + 1) + b)];
  }
  double at(const int a, const int b) const
  {
    return c_[static_cast<std::size_t>(a * (degree_ + 1) + b)];
  }

  Poly2 operator+(const Poly2 &o) const { return combine(o, 1.0); }
  Poly2 operator-(const Poly2 &o) const { return combine(o, -1.0); }
  Poly2 operator*(const double s) const
  {
    Poly2 r = *this;
    for (double &v : r.c_) v *= s;
    return r;
  }
  Poly2 operator*(const Poly2 &o) const
  {
    if (degree_ > trimmedDegree() || o.degree_ > o.trimmedDegree())
    {
      return trimmed() * o.trimmed();
    }
    Poly2 r(degree_ + o.degree_);
    for (int a = 0; a <= degree_; ++a)
    {
      for (int b = 0; a + b <= degree_; ++b)
      {
        const double ca = at(a, b);
        if (ca == 0.0)
        {
          continue;
        }
        for (int c = 0; c <= o.degree_; ++c)
        {
          for (int d = 0; c + d <= o.degree_; ++d)
          {
            r.at(a + c, b + d) += ca * o.at(c, d);
          }
        }
      }
    }
    return r;
  }
  Poly2 dx() const
  {
    Poly2 r(std::max(degree_ - 1, 0));
    for (int a = 1; a <= degree_; ++a)
    {
      for (int b = 0; a + b <= degree_; ++b) r.at(a - 1, b) += a * at(a, b);
    }
    return r;
  }
  Poly2 dy() const
  {
    Poly2 r(std::max(degree_ - 1, 0));
    for (int a = 0; a <= degree_; ++a)
    {
      for (int b = 1; a + b <= degree_; ++b) r.at(a, b - 1) += b * at(a, b);
    }
    return r;
  }

  /// The highest total degree with a nonzero coefficient (0 for zero).
  int trimmedDegree() const
  {
    for (int k = degree_; k > 0; --k)
    {
      for (int a = 0; a <= k; ++a)
      {
        if (at(a, k - a) != 0.0)
        {
          return k;
        }
      }
    }
    return 0;
  }

  Poly2 trimmed() const
  {
    Poly2 r(trimmedDegree());
    for (int a = 0; a <= r.degree_; ++a)
    {
      for (int b = 0; a + b <= r.degree_; ++b) r.at(a, b) = at(a, b);
    }
    return r;
  }

  bool isZero() const
  {
    return std::all_of(c_.begin(), c_.end(), [](const double v) { return v == 0.0; });
  }

  /// The polynomial in t along the direction (cos_phi, sin_phi).
  Poly along(const double cos_phi, const double sin_phi) const
  {
    std::vector<double> cp(static_cast<std::size_t>(degree_ + 1), 1.0);
    std::vector<double> sp(static_cast<std::size_t>(degree_ + 1), 1.0);
    for (int i = 1; i <= degree_; ++i)
    {
      cp[static_cast<std::size_t>(i)] = cp[static_cast<std::size_t>(i - 1)] * cos_phi;
      sp[static_cast<std::size_t>(i)] = sp[static_cast<std::size_t>(i - 1)] * sin_phi;
    }
    Poly r(static_cast<std::size_t>(degree_ + 1), 0.0);
    for (int a = 0; a <= degree_; ++a)
    {
      for (int b = 0; a + b <= degree_; ++b)
      {
        r[static_cast<std::size_t>(a + b)] +=
          at(a, b) * cp[static_cast<std::size_t>(a)] * sp[static_cast<std::size_t>(b)];
      }
    }
    return r;
  }

  /// Per power t^k, the sum of |c(a, b)| over a + b = k, times k when
  /// `derivative`: along every direction this bounds the polynomial (and
  /// its derivative with respect to the direction angle, since
  /// |d(cos^a sin^b) / d(phi)| <= a + b) coefficient by coefficient.
  Poly absAlong(const bool derivative) const
  {
    Poly r(static_cast<std::size_t>(degree_ + 1), 0.0);
    for (int a = 0; a <= degree_; ++a)
    {
      for (int b = 0; a + b <= degree_; ++b)
      {
        r[static_cast<std::size_t>(a + b)] +=
          std::abs(at(a, b)) * (derivative ? static_cast<double>(a + b) : 1.0);
      }
    }
    return r;
  }

private:
  Poly2 combine(const Poly2 &o, const double sign) const
  {
    Poly2 r(std::max(degree_, o.degree_));
    for (int a = 0; a <= degree_; ++a)
    {
      for (int b = 0; a + b <= degree_; ++b) r.at(a, b) += at(a, b);
    }
    for (int a = 0; a <= o.degree_; ++a)
    {
      for (int b = 0; a + b <= o.degree_; ++b) r.at(a, b) += sign * o.at(a, b);
    }
    return r;
  }

  int degree_;
  std::vector<double> c_;
};

// ===========================================================================
// The source's valid directions
// ===========================================================================

double coefficient(const DistortionModel &d, const int i)
{
  return i < static_cast<int>(d.count) ? static_cast<double>(d.coeffs[static_cast<std::size_t>(i)])
                                       : 0.0;
}

bool isRational(const DistortionModel &d)
{
  return d.type == DistortionModelType::RATIONAL8 || d.type == DistortionModelType::THIN_PRISM12 ||
         d.type == DistortionModelType::TILTED14;
}

bool hasPrism(const DistortionModel &d)
{
  return d.type == DistortionModelType::THIN_PRISM12 || d.type == DistortionModelType::TILTED14;
}

/// Polynomials, in the undistorted normalized plane (x, y), that stay
/// positive from the axis out to a point exactly when the plane distortion
/// has not folded over or broken down on the way: the Jacobian determinant
/// of the distortion (Q^4 det J, the rational denominator Q cleared), Q
/// itself, and for a tilt the third component of the tilted point (Q
/// cleared, signed to be positive on the axis; the tilt's own determinant
/// is positive). With `direction_terms` false the tangential, prism and tilt
/// terms are left out, which gives their direction-independent part.
struct PlaneConditions
{
  Poly2 det;
  Poly2 denominator;
  std::optional<Poly2> tilt;
};

PlaneConditions planeConditions(const DistortionModel &d, const bool direction_terms)
{
  const double w = direction_terms ? 1.0 : 0.0;
  const double k1 = coefficient(d, 0);
  const double k2 = coefficient(d, 1);
  const double p1 = w * coefficient(d, 2);
  const double p2 = w * coefficient(d, 3);
  const double k3 = coefficient(d, 4);
  const bool rational = isRational(d);
  const bool prism = hasPrism(d);
  const double k4 = rational ? coefficient(d, 5) : 0.0;
  const double k5 = rational ? coefficient(d, 6) : 0.0;
  const double k6 = rational ? coefficient(d, 7) : 0.0;
  const double s1 = prism ? w * coefficient(d, 8) : 0.0;
  const double s2 = prism ? w * coefficient(d, 9) : 0.0;
  const double s3 = prism ? w * coefficient(d, 10) : 0.0;
  const double s4 = prism ? w * coefficient(d, 11) : 0.0;

  const Poly2 x = Poly2::x();
  const Poly2 y = Poly2::y();
  const Poly2 one = Poly2::constant(1.0);
  const Poly2 r2 = x * x + y * y;
  const Poly2 r4 = r2 * r2;
  const Poly2 r6 = r4 * r2;
  const Poly2 num = one + r2 * k1 + r4 * k2 + r6 * k3;
  const Poly2 den = one + r2 * k4 + r4 * k5 + r6 * k6;
  const Poly2 tan_x = x * y * (2.0 * p1) + (r2 + x * x * 2.0) * p2 + r2 * s1 + r4 * s2;
  const Poly2 tan_y = (r2 + y * y * 2.0) * p1 + x * y * (2.0 * p2) + r2 * s3 + r4 * s4;
  // Distorted point (X / Q, Y / Q).
  const Poly2 big_x = x * num + den * tan_x;
  const Poly2 big_y = y * num + den * tan_y;
  const Poly2 a11 = big_x.dx() * den - big_x * den.dx();
  const Poly2 a12 = big_x.dy() * den - big_x * den.dy();
  const Poly2 a21 = big_y.dx() * den - big_y * den.dx();
  const Poly2 a22 = big_y.dy() * den - big_y * den.dy();

  PlaneConditions c{a11 * a22 - a12 * a21, den, std::nullopt};
  if (d.has_tilt)
  {
    if (direction_terms)
    {
      const auto &t = d.tilt_matrix;
      const double sign = t[8] >= 0.0f ? 1.0 : -1.0;
      c.tilt = (big_x * static_cast<double>(t[6]) + big_y * static_cast<double>(t[7]) +
                den * static_cast<double>(t[8])) *
               sign;
    }
    else
    {
      c.tilt = den;
    }
  }
  return c;
}

/// A point where `upper`, an upper bound on a condition, is certified
/// negative, so the condition has failed by then along every direction it
/// bounds; infinity when none is found just past its first root (a root it
/// only touches, for one).
double certifiedEnd(const Poly &upper)
{
  if (const auto t = firstNotPositiveIn(upper, 0.0, std::numeric_limits<double>::infinity()))
  {
    for (const double step : {1e-12, 1e-9, 1e-6, 1e-3})
    {
      const double probe = *t * (1.0 + step) + 1e-300;
      if (std::isfinite(probe) && evaluate(upper, probe) < -1e-12 * magnitude(upper, probe))
      {
        return probe;
      }
    }
  }
  return std::numeric_limits<double>::infinity();
}

// ===========================================================================
// Projection laws symmetric about the axis
// ===========================================================================

/// The normalized image radius of a projection symmetric about its axis
/// (every law but the lat-long ones), with unit intrinsics and no plane
/// correction but with its angle correction (Kannala-Brandt) if it has one,
/// at angle theta from the axis. The domain is where the library's forward
/// projection succeeds along an azimuth. The radius increases on it (for a
/// sphere model, the domain is cut where a sample shows otherwise; a
/// Kannala-Brandt fold is the source's to find).
class RadialLaw
{
public:
  explicit RadialLaw(const CameraModel &model)
  : type_(model.projection.type),
    xi_(model.projection.xi),
    alpha_(model.projection.alpha),
    beta_(model.projection.beta)
  {
    if (distortionSpaceOf(model.distortion.type) == DistortionSpace::ANGLE)
    {
      const int count = model.distortion.type == DistortionModelType::KB8 ? 8 : 4;
      for (int i = 0; i < count; ++i)
      {
        kb_.push_back(coefficient(model.distortion, i));
      }
    }
    CameraModel law = model;
    law.intrinsics = IntrinsicsModel{};
    if (distortionSpaceOf(model.distortion.type) == DistortionSpace::PLANE)
    {
      law.distortion = DistortionModel{};
    }
    limit_ = coneLimit([&](const double theta) { return rayToPixel(law, rayAt(theta, 0.0)).ok(); }
    ).value_or(0.0);
    if (type_ == ProjectionModelType::PINHOLE)
    {
      limit_ = 0.5 * constants::kPi;  // the horizon itself, not reached
    }
    // The sphere models' radius increases on their domain for the usual
    // parameters; cut the domain where a sample says otherwise.
    const bool sphere = type_ == ProjectionModelType::OMNIDIRECTIONAL ||
                        type_ == ProjectionModelType::DOUBLE_SPHERE ||
                        type_ == ProjectionModelType::EUCM;
    constexpr int kChecks = 1024;
    double previous = radius(0.0);
    for (int i = 1; sphere && i <= kChecks; ++i)
    {
      const double theta = limit_ * i / kChecks;
      const double r = radius(theta);
      if (!(r > previous))
      {
        limit_ = limit_ * (i - 1) / kChecks;
        break;
      }
      previous = r;
    }
  }

  /// Largest angle of the domain.
  double limit() const { return limit_; }

  double radius(const double theta) const
  {
    const auto s = [theta] { return std::sin(theta); };
    const auto c = [theta] { return std::cos(theta); };
    switch (type_)
    {
      case ProjectionModelType::PINHOLE:
        return std::tan(theta);
      case ProjectionModelType::EQUIDISTANT: {
        double factor = 0.0;
        const double t2 = theta * theta;
        for (std::size_t i = kb_.size(); i-- > 0;)
        {
          factor = (factor + kb_[i]) * t2;
        }
        return theta * (1.0 + factor);
      }
      case ProjectionModelType::EQUISOLID:
        return 2.0 * std::sin(0.5 * theta);
      case ProjectionModelType::STEREOGRAPHIC:
        return 2.0 * std::tan(0.5 * theta);
      case ProjectionModelType::ORTHOGRAPHIC:
        return s();
      case ProjectionModelType::OMNIDIRECTIONAL:
        return s() / (c() + xi_);
      case ProjectionModelType::DOUBLE_SPHERE: {
        const double sn = s();
        const double w = xi_ + c();
        return sn / (alpha_ * std::sqrt(sn * sn + w * w) + (1.0 - alpha_) * w);
      }
      case ProjectionModelType::EUCM: {
        const double sn = s();
        const double cs = c();
        return sn / (alpha_ * std::sqrt(beta_ * sn * sn + cs * cs) + (1.0 - alpha_) * cs);
      }
      default:
        return 0.0;
    }
  }

  /// The radius over [theta0, theta1] inside the domain (it increases).
  Interval radius(const double theta0, const double theta1) const
  {
    const Interval r = widened(radius(theta0), radius(theta1));
    return widened(r.lo, r.hi);  // the formula's own rounding
  }

  /// The radius' derivative over the angles `t` inside the domain; nullopt
  /// for the pinhole law (its output has bounds of its own) or where a
  /// denominator may vanish.
  std::optional<Interval> slope(const Interval t) const
  {
    const Interval s = sine(t);
    const Interval c = cosine(t);
    const Interval one{1.0, 1.0};
    switch (type_)
    {
      case ProjectionModelType::EQUIDISTANT: {
        // theta + sum k_i theta^(2 i + 3)
        Interval sum = one;
        Interval power = one;
        for (std::size_t i = 0; i < kb_.size(); ++i)
        {
          power = power * square(t);
          sum = sum + power * ((2.0 * static_cast<double>(i) + 3.0) * kb_[i]);
        }
        return sum;
      }
      case ProjectionModelType::EQUISOLID:
        return cosine(t * 0.5);
      case ProjectionModelType::STEREOGRAPHIC:
        return divide(one, square(cosine(t * 0.5)));
      case ProjectionModelType::ORTHOGRAPHIC:
        return c;
      case ProjectionModelType::OMNIDIRECTIONAL:
        // sin / (cos + xi): (1 + xi cos) / (cos + xi)^2
        return divide(c * xi_ + 1.0, square(c + xi_));
      case ProjectionModelType::DOUBLE_SPHERE: {
        // sin / D, D = alpha sqrt(sin^2 + w^2) + (1 - alpha) w, w = xi + cos:
        // D' = -alpha xi sin / sqrt(sin^2 + w^2) - (1 - alpha) sin.
        const Interval w = c + xi_;
        const Interval root = squareRoot(square(s) + square(w));
        const Interval den = root * alpha_ + w * (1.0 - alpha_);
        const auto lean = divide(s * (-alpha_ * xi_), root);
        if (!lean)
        {
          return std::nullopt;
        }
        const Interval den_slope = *lean + s * (alpha_ - 1.0);
        return divide(c * den - s * den_slope, square(den));
      }
      case ProjectionModelType::EUCM: {
        // sin / D, D = alpha sqrt(beta sin^2 + cos^2) + (1 - alpha) cos:
        // D' = alpha (beta - 1) sin cos / sqrt(...) - (1 - alpha) sin.
        const Interval root = squareRoot(square(s) * beta_ + square(c));
        const Interval den = root * alpha_ + c * (1.0 - alpha_);
        const auto lean = divide(s * c * (alpha_ * (beta_ - 1.0)), root);
        if (!lean)
        {
          return std::nullopt;
        }
        const Interval den_slope = *lean + s * (alpha_ - 1.0);
        return divide(c * den - s * den_slope, square(den));
      }
      default:
        return std::nullopt;
    }
  }

private:
  ProjectionModelType type_;
  double xi_;
  double alpha_;
  double beta_;
  std::vector<double> kb_;
  double limit_{0.0};
};

// ===========================================================================
// Linear forms over a patch
// ===========================================================================

/// A quantity over a patch linearized at its centre: at offsets (oa, ob)
/// from the centre it lies in c + da oa + db ob, with da, db enclosures of
/// its derivatives over the patch (its mean value form).
struct LinearForm
{
  Interval c;
  Interval da;
  Interval db;

  Interval over(const Interval oa, const Interval ob) const { return c + da * oa + db * ob; }
};

/// Where `form` may be at least `bound` (`above`) or at most it.
struct HalfPlane
{
  LinearForm form;
  double bound;
  bool above;
};

/// The largest value of `f` (the smallest, with `lowest`) over the offsets
/// oa x ob at which every constraint may hold; nullopt where none may. In
/// each quadrant of the offsets the forms' extreme sides are linear, so the
/// region there is a convex polygon and the extreme sits at a vertex.
std::optional<double> extremeOver(
  const LinearForm &f, const bool lowest, const Interval oa, const Interval ob,
  const HalfPlane *constraints, const int count
)
{
  using Point = std::pair<double, double>;
  // A rectangle clipped by `count` half-planes has at most 4 + count vertices.
  constexpr int kMaxVertices = 16;
  if (count > kMaxVertices - 4)
  {
    return std::nullopt;
  }
  const double reach_a = maxAbs(oa);
  const double reach_b = maxAbs(ob);
  // Slack well above the rounding of the arithmetic below, on the safe side.
  const auto slack = [&](const double c, const Interval da, const Interval db) {
    return 1e-9 * (1.0 + std::abs(c) + maxAbs(da) * reach_a + maxAbs(db) * reach_b);
  };
  std::optional<double> best;
  for (const bool pa : {false, true})
  {
    for (const bool pb : {false, true})
    {
      const double a0 = pa ? std::max(oa.lo, 0.0) : oa.lo;
      const double a1 = pa ? oa.hi : std::min(oa.hi, 0.0);
      const double b0 = pb ? std::max(ob.lo, 0.0) : ob.lo;
      const double b1 = pb ? ob.hi : std::min(ob.hi, 0.0);
      if (a0 > a1 || b0 > b1)
      {
        continue;
      }
      Point polygon[kMaxVertices] = {{a0, b0}, {a1, b0}, {a1, b1}, {a0, b1}};
      int size = 4;
      for (int k = 0; k < count && size > 0; ++k)
      {
        // Kept where k0 + ka a + kb b >= -slack.
        const HalfPlane &h = constraints[k];
        const Interval &da = h.form.da;
        const Interval &db = h.form.db;
        const double k0 = h.above ? h.form.c.hi - h.bound : h.bound - h.form.c.lo;
        const double ka = h.above ? (pa ? da.hi : da.lo) : -(pa ? da.lo : da.hi);
        const double kb = h.above ? (pb ? db.hi : db.lo) : -(pb ? db.lo : db.hi);
        const double s = slack(k0, da, db);
        const auto side = [&](const Point &p) { return k0 + ka * p.first + kb * p.second + s; };
        Point clipped[kMaxVertices];
        int kept = 0;
        for (int i = 0; i < size; ++i)
        {
          const Point &p = polygon[i];
          const Point &q = polygon[(i + 1) % size];
          const double sp = side(p);
          const double sq = side(q);
          if (sp >= 0.0)
          {
            clipped[kept++] = p;
          }
          if ((sp >= 0.0) != (sq >= 0.0))
          {
            const double t = sp / (sp - sq);
            clipped[kept++] =
              Point{p.first + t * (q.first - p.first), p.second + t * (q.second - p.second)};
          }
        }
        std::copy(clipped, clipped + kept, polygon);
        size = kept;
      }
      const double ga = lowest == pa ? f.da.lo : f.da.hi;
      const double gb = lowest == pb ? f.db.lo : f.db.hi;
      for (int i = 0; i < size; ++i)
      {
        const Point &p = polygon[i];
        const double value = (lowest ? f.c.lo : f.c.hi) + ga * p.first + gb * p.second;
        best = !best ? value : (lowest ? std::min(*best, value) : std::max(*best, value));
      }
    }
  }
  if (best)
  {
    const double s = slack(maxAbs(f.c), f.da, f.db);
    *best += lowest ? -s : s;
  }
  return best;
}

// ===========================================================================
// The source
// ===========================================================================

/// The source over patches of directions: their rays, whether they are
/// valid source directions, and their source pixels.
///
/// A ray is valid when the forward projection succeeds and, along its
/// direction out from the source axis, keeps the orientation it has near the
/// axis: past the first point where it turns back (a distortion polynomial
/// that folds over, e.g. a strong negative radtan k1), pixels unproject onto
/// the folded branch, to rays the lens does not see there. That edge comes
/// from the distortion polynomials, bracketed with certified Bernstein
/// bounds: the slope of the Kannala-Brandt angle polynomial; for a plane
/// correction its Jacobian determinant (rational denominator cleared), the
/// denominator and the tilt's third component, either exactly (radial terms
/// only) or bounded over the patch's directions (with tangential, prism or
/// tilt terms). A patch is valid, invalid, or (straddling the edge, or near
/// a root the bracket could not settle) undecided; the searches treat
/// undecided patches on the safe side for each end.
///
/// Patches are angles from the source axis and azimuths, or longitudes and
/// latitudes for a lat-long source.
class Source
{
public:
  struct Patch
  {
    // Directions of the patch's rays: unit vectors, or for the plane chart
    // (x, y, 1).
    Interval x;
    Interval y;
    Interval z;
    bool unit;
    Tri valid;
    bool has_pixel;
    // Certainly no ray of the patch is both valid and inside the image
    // (widened by a negative margin), with or without a pixel enclosure.
    bool lost;
    Interval u;
    Interval v;
  };

  /// `margin`: the searches' inner edge, in pixels inside the image.
  Source(const CameraModel &model, const ImageSize &size, const float margin)
  : model_(model),
    lat_long_(isLatLongProjection(model.projection.type)),
    law_(model),
    fx_(model.intrinsics.fx),
    fy_(model.intrinsics.fy),
    cx_(model.intrinsics.cx),
    cy_(model.intrinsics.cy),
    skew_(model.intrinsics.skew),
    edge_(std::min(static_cast<double>(margin), 0.0)),
    u_end_(static_cast<double>(size.width - 1) - edge_),
    v_end_(static_cast<double>(size.height - 1) - edge_),
    inner_(std::max(static_cast<double>(margin), 0.0)),
    inner_u_end_(static_cast<double>(size.width - 1) - inner_),
    inner_v_end_(static_cast<double>(size.height - 1) - inner_)
  {
    if (lat_long_)
    {
      return;
    }
    // The image's largest distorted normalized radius (at a corner).
    for (const double u : {edge_, u_end_})
    {
      for (const double v : {edge_, v_end_})
      {
        const double y = (v - cy_) / fy_;
        const double x = (u - cx_ - skew_ * y) / fx_;
        image_radius_ = std::max(image_radius_, std::sqrt(x * x + y * y));
      }
    }
    const DistortionSpace space = distortionSpaceOf(model.distortion.type);
    if (space == DistortionSpace::ANGLE)
    {
      // theta_d = theta (1 + k1 t + k2 t^2 + ...), t = theta^2: its slope is
      // 1 + 3 k1 t + 5 k2 t^2 + ...
      Poly slope{1.0};
      const int count = model.distortion.type == DistortionModelType::KB8 ? 8 : 4;
      for (int i = 0; i < count; ++i)
      {
        slope.push_back((2.0 * i + 3.0) * coefficient(model.distortion, i));
      }
      const double t_max = constants::kPi * constants::kPi;
      const auto t = firstNotPositiveIn(slope, 0.0, t_max);
      fold_lo_ = t ? std::sqrt(*t) : std::numeric_limits<double>::infinity();
      const double end = certifiedEnd(slope);
      fold_hi_ = std::isfinite(end) ? std::sqrt(end) : std::numeric_limits<double>::infinity();
    }
    if (space == DistortionSpace::PLANE)
    {
      initPlane();
    }
    else
    {
      // Without a plane correction a ray's normalized point is its pixel's:
      // beyond the image's largest radius, none lands inside.
      lost_from_ = image_radius_ * (1.0 + 1e-9);
    }
  }

  bool latLong() const { return lat_long_; }

  /// No ray further than this from the source axis is valid.
  double thetaEnd() const { return std::min(law_.limit(), fold_hi_); }

  Patch evaluate(
    const Chart chart, const double a0, const double a1, const double b0, const double b1
  ) const
  {
    switch (chart)
    {
      case Chart::POLAR:
        return polarPatch(a0, a1, b0, b1);
      case Chart::PLANE:
        return planePatch(a0, a1, b0, b1);
      case Chart::LAT_LONG:
        break;
    }
    return latLongPatch(a0, a1, b0, b1);
  }

  /// The ray at chart coordinates (a, b).
  Eigen::Vector3f rayAt(const Chart chart, const double a, const double b) const
  {
    switch (chart)
    {
      case Chart::POLAR:
        return detail::rayAt(a, b);
      case Chart::PLANE:
        return Eigen::Vector3f(static_cast<float>(a), static_cast<float>(b), 1.0f);
      case Chart::LAT_LONG:
        break;
    }
    return Eigen::Vector3d(std::cos(b) * std::sin(a), std::sin(b), std::cos(b) * std::cos(a))
      .cast<float>();
  }

  /// The source pixel at chart coordinates (a, b), if it projects.
  std::optional<Eigen::Vector2f> pixelAt(const Chart chart, const double a, const double b) const
  {
    const PixelResult px = rayToPixel(model_, rayAt(chart, a, b));
    return px.ok() ? std::optional<Eigen::Vector2f>(Eigen::Vector2f(px.pixel.u, px.pixel.v))
                   : std::nullopt;
  }

  /// A pinhole source: its front half is searched on the plane chart, out to
  /// this undistorted normalized radius (the law's own cut, see RadialLaw).
  bool pinholeLaw() const { return model_.projection.type == ProjectionModelType::PINHOLE; }
  /// The plane chart's reach: past any float pinhole image (2^31 pixels at
  /// the smallest float focal length reach about 1e54), with room for the
  /// squares of the interval arithmetic.
  static constexpr double planeExtent() { return 1e150; }

  /// Every ray out past this normalized radius is outside the image or
  /// invalid.
  double lostFrom() const { return std::min(lost_from_, invalid_from_); }

  /// The source pixel (u, v) over a patch of valid rays as linear forms;
  /// nullopt for charts and corrections without them here (a lat-long
  /// source, whose pixel is already exact, a tilt, a plane correction on a
  /// polar patch).
  std::optional<std::pair<LinearForm, LinearForm>> pixelForms(
    const Chart chart, const double a0, const double a1, const double b0, const double b1
  ) const
  {
    const double ac = 0.5 * (a0 + a1);
    const double bc = 0.5 * (b0 + b1);
    Interval xc;
    Interval yc;
    Interval j[2][2];  // d(xd, yd) / d(a, b)
    if (chart == Chart::PLANE)
    {
      if (plane_)
      {
        const Interval px{ac, ac};
        const Interval py{bc, bc};
        if (model_.distortion.has_tilt || !distort(px, py, square(px) + square(py), xc, yc) || !jacobian(Interval{a0, a1}, Interval{b0, b1}, j))
        {
          return std::nullopt;
        }
      }
      else
      {
        xc = Interval{ac, ac};
        yc = Interval{bc, bc};
        j[0][0] = Interval{1.0, 1.0};
        j[0][1] = Interval{0.0, 0.0};
        j[1][0] = Interval{0.0, 0.0};
        j[1][1] = Interval{1.0, 1.0};
      }
    }
    else if (chart == Chart::POLAR && !plane_ && a1 <= std::min(law_.limit(), fold_lo_))
    {
      // (rho(theta) cos(phi), rho(theta) sin(phi)).
      const auto slope = law_.slope(Interval{a0, a1});
      if (!slope)
      {
        return std::nullopt;
      }
      const Interval rho = law_.radius(a0, a1);
      const Interval cp = cosine(Interval{b0, b1});
      const Interval sp = sine(Interval{b0, b1});
      j[0][0] = *slope * cp;
      j[0][1] = negated(rho * sp);
      j[1][0] = *slope * sp;
      j[1][1] = rho * cp;
      const Interval rc = law_.radius(ac, ac);
      xc = rc * cosine(Interval{bc, bc});
      yc = rc * sine(Interval{bc, bc});
    }
    else
    {
      return std::nullopt;
    }
    const LinearForm u{
      xc * fx_ + yc * skew_ + cx_, j[0][0] * fx_ + j[1][0] * skew_,
      j[0][1] * fx_ + j[1][1] * skew_};
    const LinearForm v{yc * fy_ + cy_, j[1][0] * fy_, j[1][1] * fy_};
    return std::pair{u, v};
  }

private:
  void initPlane()
  {
    plane_ = true;
    const PlaneConditions full = planeConditions(model_.distortion, true);
    const PlaneConditions radial = planeConditions(model_.distortion, false);
    det_ = full.det;
    const Poly2 det_dir = full.det - radial.det;
    if (full.tilt)
    {
      tilt_ = *full.tilt;
    }
    const Poly2 tilt_dir = full.tilt ? *full.tilt - *radial.tilt : Poly2(0);
    directional_ = !det_dir.isZero() || !tilt_dir.isZero();
    det_slack_ = det_dir.absAlong(true);
    tilt_slack_ = tilt_dir.absAlong(true);

    const double infinity = std::numeric_limits<double>::infinity();
    const Poly denominator = radial.denominator.along(1.0, 0.0);
    const double denominator_positive = positiveUpTo(denominator, infinity);
    // Every direction: the radial part minus (or plus) a bound on the rest.
    const Poly det_lower = radial.det.along(1.0, 0.0) - det_dir.absAlong(false);
    const Poly det_upper = radial.det.along(1.0, 0.0) + det_dir.absAlong(false);
    valid_below_ = std::min(denominator_positive, positiveUpTo(det_lower, infinity));
    invalid_from_ = std::min(certifiedEnd(denominator), certifiedEnd(det_upper));
    if (radial.tilt)
    {
      const Poly tilt_lower = radial.tilt->along(1.0, 0.0) - tilt_dir.absAlong(false);
      const Poly tilt_upper = radial.tilt->along(1.0, 0.0) + tilt_dir.absAlong(false);
      valid_below_ = std::min(valid_below_, positiveUpTo(tilt_lower, infinity));
      invalid_from_ = std::min(invalid_from_, certifiedEnd(tilt_upper));
    }
    denominator_end_ = std::min(denominator_positive, certifiedEnd(denominator));
    // Below this the radial part alone, r P(r^2) / Q(r^2), increases.
    radial_increasing_below_ =
      std::min(denominator_positive, positiveUpTo(radial.det.along(1.0, 0.0), infinity));
    const DistortionModel &d = model_.distortion;
    const double p = std::abs(coefficient(d, 2)) + std::abs(coefficient(d, 3));
    offset_r2_ = 4.0 * p;
    if (d.has_thin_prism)
    {
      offset_r2_ += std::abs(coefficient(d, 8)) + std::abs(coefficient(d, 10));
      offset_r4_ = std::abs(coefficient(d, 9)) + std::abs(coefficient(d, 11));
    }
    if (!d.has_tilt)
    {
      // Where r P - Q (R + offset) > 0, a ray with Q > 0 has its radial part,
      // less the most the other terms can move it, beyond the image R, and a
      // ray with Q <= 0 is invalid: none is a valid ray inside the image.
      // From where that holds on to the fold, every ray is lost.
      const auto c = [&](const int i) { return coefficient(d, i); };
      const Poly numerator{0.0, 1.0, 0.0, c(0), 0.0, c(1), 0.0, c(4)};
      const Poly reach{image_radius_ * (1.0 + 1e-9), 0.0, offset_r2_, 0.0, offset_r4_};
      const Poly beyond = numerator - denominator * reach;
      const double from = certifiedEnd(beyond * -1.0);
      if (std::isfinite(from))
      {
        const auto to = firstNotPositiveIn(beyond, from, infinity);
        if (invalid_from_ <= (to ? *to : infinity))
        {
          lost_from_ = std::min(from, invalid_from_);
        }
      }
    }
  }

  /// The radial part of the plane correction, r P(r^2) / Q(r^2).
  double radialPart(const double r) const
  {
    const DistortionModel &d = model_.distortion;
    const auto c = [&](const int i) { return coefficient(d, i); };
    const double s = r * r;
    double radial = 1.0 + s * (c(0) + s * (c(1) + s * c(4)));
    if (d.is_rational)
    {
      radial /= 1.0 + s * (c(5) + s * (c(6) + s * c(7)));
    }
    return r * radial;
  }

  /// Validity of undistorted normalized radii `r` at the azimuths
  /// [b0, b1] that `azimuths()` returns (asked for only when needed).
  template <typename Azimuths>
  Tri planeValidity(const Interval r, const Azimuths &azimuths) const
  {
    if (r.hi < valid_below_)
    {
      return Tri::YES;
    }
    if (r.lo >= invalid_from_)
    {
      return Tri::NO;
    }
    if (!directional_)
    {
      return Tri::MAYBE;
    }
    const auto [b0, b1] = azimuths();
    if (b1 - b0 > kBoundedAzimuths)
    {
      return Tri::MAYBE;
    }
    // Bounds over the patch's azimuths: the conditions along its middle
    // azimuth, minus or plus half its width times a bound on their change
    // with the azimuth.
    const double half = 0.5 * (b1 - b0);
    const double c = std::cos(b0 + half);
    const double s = std::sin(b0 + half);
    const Poly det = det_.along(c, s);
    const Poly det_slack = det_slack_ * half;
    bool positive = r.hi < denominator_end_ && !firstNotPositiveIn(det - det_slack, 0.0, r.hi);
    double end = certifiedEnd(det + det_slack);
    if (tilt_)
    {
      const Poly tilt = tilt_->along(c, s);
      const Poly tilt_slack = tilt_slack_ * half;
      positive = positive && !firstNotPositiveIn(tilt - tilt_slack, 0.0, r.hi);
      end = std::min(end, certifiedEnd(tilt + tilt_slack));
    }
    if (positive)
    {
      return Tri::YES;
    }
    return end <= r.lo ? Tri::NO : Tri::MAYBE;
  }

  /// The plane correction over a patch of undistorted points (xn, yn) with
  /// squared radius s. Where a denominator (the rational Q, the tilt's third
  /// component) may be zero over the patch, the enclosure covers only the
  /// rays on its valid side, the others being invalid, and is then
  /// unbounded. False when no ray of the patch is on that side.
  bool distort(const Interval xn, const Interval yn, const Interval s, Interval &xd, Interval &yd)
    const
  {
    const DistortionModel &d = model_.distortion;
    const auto c = [&](const int i) { return coefficient(d, i); };
    const Interval xy = xn * yn;
    Interval tan_x = xy * (2.0 * c(2)) + (s + square(xn) * 2.0) * c(3);
    Interval tan_y = (s + square(yn) * 2.0) * c(2) + xy * (2.0 * c(3));
    if (d.has_thin_prism)
    {
      const Interval s2 = square(s);
      tan_x = tan_x + s * c(8) + s2 * c(9);
      tan_y = tan_y + s * c(10) + s2 * c(11);
    }
    const Interval radial = (((s * c(4)) + c(1)) * s + c(0)) * s + 1.0;
    const Interval den =
      d.is_rational ? (((s * c(7)) + c(6)) * s + c(5)) * s + 1.0 : Interval{1.0, 1.0};
    if (den.lo > 0.0)
    {
      const Interval ratio = *divide(radial, den);
      xd = xn * ratio + tan_x;
      yd = yn * ratio + tan_y;
    }
    else if (den.hi > 0.0)
    {
      xd = overPositivePart(xn * radial, den.hi) + tan_x;
      yd = overPositivePart(yn * radial, den.hi) + tan_y;
    }
    else
    {
      return false;
    }
    if (d.has_tilt)
    {
      // Signed so that the valid side has a positive third component.
      const auto &t = d.tilt_matrix;
      const double sign = t[8] >= 0.0f ? 1.0 : -1.0;
      const auto row = [&](const int i) {
        return xd * (sign * t[3 * i]) + yd * (sign * t[3 * i + 1]) + sign * t[3 * i + 2];
      };
      const Interval mx = row(0);
      const Interval my = row(1);
      const Interval mz = row(2);
      if (mz.lo > 0.0)
      {
        xd = *divide(mx, mz);
        yd = *divide(my, mz);
      }
      else if (mz.hi > 0.0)
      {
        xd = overPositivePart(mx, mz.hi);
        yd = overPositivePart(my, mz.hi);
      }
      else
      {
        return false;
      }
    }
    return !std::isnan(xd.lo) && !std::isnan(xd.hi) && !std::isnan(yd.lo) && !std::isnan(yd.hi);
  }

  Patch polarPatch(const double a0, const double a1, const double b0, const double b1) const
  {
    Patch patch{};
    patch.unit = true;
    const Interval st = sine(Interval{a0, a1});
    const Interval ct = cosine(Interval{a0, a1});
    const Interval sp = sine(Interval{b0, b1});
    const Interval cp = cosine(Interval{b0, b1});
    patch.x = st * cp;
    patch.y = st * sp;
    patch.z = ct;
    const double limit = law_.limit();
    if (a0 > limit || a0 >= fold_hi_)
    {
      patch.valid = Tri::NO;
      return patch;
    }
    patch.valid = a1 <= limit && a1 < fold_lo_ ? Tri::YES : Tri::MAYBE;
    const Interval r = law_.radius(a0, std::min({a1, limit, fold_hi_}));
    const Sector sector{cp, sp};
    normalizedPart(
      patch, r * cp, r * sp, r,
      [&] {
        return std::pair{b0, b1};
      },
      &sector
    );
    return patch;
  }

  /// A box [x0, x1] x [y0, y1] of a pinhole source's undistorted
  /// normalized plane (rays (x, y, 1)).
  Patch planePatch(const double x0, const double x1, const double y0, const double y1) const
  {
    Patch patch{};
    const Interval x{x0, x1};
    const Interval y{y0, y1};
    const Interval squared = square(x) + square(y);
    patch.x = x;
    patch.y = y;
    patch.z = Interval{1.0, 1.0};
    patch.unit = false;
    const Interval r = squareRoot(squared);
    const double extent = planeExtent();
    if (r.lo > extent)
    {
      patch.valid = Tri::NO;
      return patch;
    }
    patch.valid = r.hi <= extent ? Tri::YES : Tri::MAYBE;
    // Azimuths of the box (all of them when it holds the origin).
    const auto azimuths = [&] {
      if (x0 <= 0.0 && x1 >= 0.0 && y0 <= 0.0 && y1 >= 0.0)
      {
        return std::pair{0.0, 2.0 * constants::kPi};
      }
      double c[4] = {
        std::atan2(y0, x0), std::atan2(y0, x1), std::atan2(y1, x0), std::atan2(y1, x1)};
      if (x1 < 0.0 && y0 < 0.0 && y1 >= 0.0)
      {
        // Across the negative x axis: shift the lower half by a turn.
        for (double &a : c)
        {
          a = a < 0.0 ? a + 2.0 * constants::kPi : a;
        }
      }
      return std::pair{*std::min_element(c, c + 4), *std::max_element(c, c + 4)};
    };
    normalizedPart(patch, x, y, r, azimuths, nullptr);
    return patch;
  }

  /// Cosine and sine of a sector's azimuths.
  struct Sector
  {
    Interval cos;
    Interval sin;
  };

  /// Pixel and validity of a patch's undistorted normalized points (xn, yn)
  /// at radii `r` and the azimuths `azimuths()` returns. A `sector` patch's
  /// points lie along its azimuths (a tighter enclosure). The plane
  /// correction's validity is not examined for a lost patch (neither search
  /// needs it there).
  template <typename Azimuths>
  void normalizedPart(
    Patch &patch, const Interval xn, const Interval yn, const Interval r, const Azimuths &azimuths,
    const Sector *sector
  ) const
  {
    if (r.lo >= lost_from_)
    {
      patch.lost = true;
      patch.valid = both(patch.valid, r.lo >= invalid_from_ ? Tri::NO : Tri::MAYBE);
      return;
    }
    const bool enclosed = pixelPart(patch, xn, yn, r, sector);
    patch.lost = patch.lost || (enclosed && (patch.u.hi < edge_ || patch.u.lo > u_end_ ||
                                             patch.v.hi < edge_ || patch.v.lo > v_end_));
    if (plane_)
    {
      patch.valid = both(patch.valid, patch.lost ? Tri::MAYBE : planeValidity(r, azimuths));
    }
  }

  /// Whether it made a pixel enclosure (maybe unbounded) for the patch.
  bool pixelPart(
    Patch &patch, const Interval xn, const Interval yn, const Interval r, const Sector *sector
  ) const
  {
    Interval xd = xn;
    Interval yd = yn;
    if (plane_)
    {
      const DistortionModel &d = model_.distortion;
      if (r.hi < radial_increasing_below_ && !d.has_tilt)
      {
        // The radial part increases over the patch: bounded by its ends.
        const Interval g = widened(radialPart(std::max(r.lo, 0.0)), radialPart(r.hi));
        // (Its formula's own rounding, relative to each end.)
        const Interval radial_g{g.lo - 1e-12 * std::abs(g.lo), g.hi + 1e-12 * std::abs(g.hi)};
        // The other terms move the point by at most this much: far enough
        // out, the patch cannot reach the image.
        const double s1 = r.hi * r.hi;
        const double offset = offset_r2_ * s1 + offset_r4_ * s1 * s1;
        if (radial_g.lo - offset > image_radius_ * (1.0 + 1e-9))
        {
          patch.lost = true;
          return false;
        }
        if (sector != nullptr && !directional_)
        {
          xd = radial_g * sector->cos;
          yd = radial_g * sector->sin;
        }
        else if (!distort(xn, yn, square(r), xd, yd))
        {
          return false;
        }
      }
      else if (!distort(xn, yn, square(r), xd, yd))
      {
        return false;
      }
    }
    patch.u = xd * fx_ + yd * skew_ + cx_;
    patch.v = yd * fy_ + cy_;
    const bool point =
      xn.hi - xn.lo <= 1e-9 * (1.0 + maxAbs(xn)) && yn.hi - yn.lo <= 1e-9 * (1.0 + maxAbs(yn));
    if (plane_ && !model_.distortion.has_tilt && !(sector != nullptr && !directional_) && !point &&
        nearEdges(patch.u, patch.v))
    {
      // Where it decides the patch, tighten the enclosure by its mean value
      // form (its error shrinks with the square of the patch size; the
      // natural one's only with the size).
      Interval xm;
      Interval ym;
      if (meanValue(xn, yn, xm, ym))
      {
        xd = intersection(xd, xm);
        yd = intersection(yd, ym);
        patch.u = xd * fx_ + yd * skew_ + cx_;
        patch.v = yd * fy_ + cy_;
      }
    }
    patch.has_pixel = std::isfinite(patch.u.lo) && std::isfinite(patch.u.hi) &&
                      std::isfinite(patch.v.lo) && std::isfinite(patch.v.hi);
    return true;
  }

  /// Whether a pixel enclosure is neither certainly inside the narrowed
  /// image nor certainly outside the widened one.
  bool nearEdges(const Interval u, const Interval v) const
  {
    const bool inside =
      u.lo >= inner_ && u.hi <= inner_u_end_ && v.lo >= inner_ && v.hi <= inner_v_end_;
    const bool outside = u.hi < edge_ || u.lo > u_end_ || v.hi < edge_ || v.lo > v_end_;
    return !inside && !outside;
  }

  /// The mean value form of the plane correction (without tilt) over the
  /// box xn x yn: its value at the centre plus its Jacobian over the box
  /// times the offsets from the centre.
  bool meanValue(const Interval xn, const Interval yn, Interval &xd, Interval &yd) const
  {
    const double mx = 0.5 * (xn.lo + xn.hi);
    const double my = 0.5 * (yn.lo + yn.hi);
    const Interval px{mx, mx};
    const Interval py{my, my};
    Interval xc;
    Interval yc;
    Interval j[2][2];
    if (!distort(px, py, square(px) + square(py), xc, yc) || !jacobian(xn, yn, j))
    {
      return false;
    }
    const Interval dx = xn + (-mx);
    const Interval dy = yn + (-my);
    xd = xc + j[0][0] * dx + j[0][1] * dy;
    yd = yc + j[1][0] * dx + j[1][1] * dy;
    return std::isfinite(xd.lo) && std::isfinite(xd.hi) && std::isfinite(yd.lo) &&
           std::isfinite(yd.hi);
  }

  /// The Jacobian of the plane correction (without tilt) over the box
  /// x x y: rows xd, yd; columns x, y.
  bool jacobian(const Interval x, const Interval y, Interval (&j)[2][2]) const
  {
    const DistortionModel &d = model_.distortion;
    const auto c = [&](const int i) { return coefficient(d, i); };
    const Interval s = square(x) + square(y);
    // The radial factor P / Q and its derivative with respect to s.
    Interval radial = (((s * c(4)) + c(1)) * s + c(0)) * s + 1.0;
    Interval slope = (s * (3.0 * c(4)) + 2.0 * c(1)) * s + c(0);
    if (d.is_rational)
    {
      const Interval den = (((s * c(7)) + c(6)) * s + c(5)) * s + 1.0;
      const Interval den_slope = (s * (3.0 * c(7)) + 2.0 * c(6)) * s + c(5);
      const auto ratio = divide(radial, den);
      const auto ratio_slope = divide(slope * den - radial * den_slope, square(den));
      if (!ratio || !ratio_slope)
      {
        return false;
      }
      radial = *ratio;
      slope = *ratio_slope;
    }
    const Interval twice_slope = slope * 2.0;
    const Interval xy = x * y;
    const double p1 = c(2);
    const double p2 = c(3);
    j[0][0] = radial + twice_slope * square(x) + y * (2.0 * p1) + x * (6.0 * p2);
    j[0][1] = twice_slope * xy + x * (2.0 * p1) + y * (2.0 * p2);
    j[1][0] = twice_slope * xy + x * (2.0 * p1) + y * (2.0 * p2);
    j[1][1] = radial + twice_slope * square(y) + y * (6.0 * p1) + x * (2.0 * p2);
    if (d.has_thin_prism)
    {
      const Interval sx = s * x;
      const Interval sy = s * y;
      j[0][0] = j[0][0] + x * (2.0 * c(8)) + sx * (4.0 * c(9));
      j[0][1] = j[0][1] + y * (2.0 * c(8)) + sy * (4.0 * c(9));
      j[1][0] = j[1][0] + x * (2.0 * c(10)) + sx * (4.0 * c(11));
      j[1][1] = j[1][1] + y * (2.0 * c(10)) + sy * (4.0 * c(11));
    }
    return true;
  }

  Patch latLongPatch(const double a0, const double a1, const double b0, const double b1) const
  {
    Patch patch{};
    patch.unit = true;
    const Interval cl = cosine(Interval{a0, a1});
    const Interval sl = sine(Interval{a0, a1});
    const Interval cp = cosine(Interval{b0, b1});
    patch.x = cp * sl;
    patch.y = sine(Interval{b0, b1});
    patch.z = cp * cl;
    const double pole = 0.5 * constants::kPi - kPoleSlack;
    patch.valid = b0 > -pole && b1 < pole ? Tri::YES : Tri::MAYBE;
    Interval y{b0, b1};
    if (model_.projection.type == ProjectionModelType::CYLINDRICAL)
    {
      if (!(b0 > -0.5 * constants::kPi && b1 < 0.5 * constants::kPi))
      {
        return patch;
      }
      y = widened(std::tan(b0), std::tan(b1));
    }
    patch.u = Interval{a0, a1} * fx_ + y * skew_ + cx_;
    patch.v = y * fy_ + cy_;
    patch.has_pixel = true;
    return patch;
  }

  // Azimuth width below which a plane-validity patch gets its own bounds.
  static constexpr double kBoundedAzimuths = 2.0 * constants::kPi / 64.0;

  const CameraModel &model_;
  bool lat_long_;
  RadialLaw law_;
  double fx_;
  double fy_;
  double cx_;
  double cy_;
  double skew_;
  // The image with the searches' margins, widened when a margin is negative
  // (edge_ to *_end_) and narrowed when it is positive (inner_ to
  // inner_*_end_).
  double edge_;
  double u_end_;
  double v_end_;
  double inner_;
  double inner_u_end_;
  double inner_v_end_;
  double fold_lo_{std::numeric_limits<double>::infinity()};
  double fold_hi_{std::numeric_limits<double>::infinity()};
  bool plane_{false};
  bool directional_{false};
  double valid_below_{std::numeric_limits<double>::infinity()};
  double invalid_from_{std::numeric_limits<double>::infinity()};
  double denominator_end_{std::numeric_limits<double>::infinity()};
  // From this undistorted normalized radius out, every ray is outside the
  // image or invalid.
  double lost_from_{std::numeric_limits<double>::infinity()};
  double radial_increasing_below_{0.0};
  double offset_r2_{0.0};
  double offset_r4_{0.0};
  double image_radius_{0.0};
  Poly2 det_;
  std::optional<Poly2> tilt_;
  Poly det_slack_;
  Poly tilt_slack_;
};

// ===========================================================================
// The output
// ===========================================================================

/// Ray directions over a box of chart coordinates, with their partial
/// derivatives along the two coordinates (all enclosures over the box).
struct DirectionJet
{
  Interval v[3];
  Interval da[3];
  Interval db[3];
};

DirectionJet chartDirection(const Chart chart, const Interval a, const Interval b)
{
  const Interval zero{0.0, 0.0};
  const Interval one{1.0, 1.0};
  switch (chart)
  {
    case Chart::PLANE:
      return DirectionJet{{a, b, one}, {one, zero, zero}, {zero, one, zero}};
    case Chart::POLAR: {
      // a: angle from the axis, b: azimuth.
      const Interval sa = sine(a);
      const Interval ca = cosine(a);
      const Interval sb = sine(b);
      const Interval cb = cosine(b);
      return DirectionJet{
        {sa * cb, sa * sb, ca}, {ca * cb, ca * sb, negated(sa)}, {negated(sa * sb), sa * cb, zero}};
    }
    case Chart::LAT_LONG:
      break;
  }
  // a: longitude, b: latitude.
  const Interval sa = sine(a);
  const Interval ca = cosine(a);
  const Interval sb = sine(b);
  const Interval cb = cosine(b);
  return DirectionJet{
    {cb * sa, sb, cb * ca},
    {cb * ca, zero, negated(cb * sa)},
    {negated(sb * sa), cb, negated(sb * ca)}};
}

/// atan2(p, z) over p >= 0 and z in their enclosures: the angle from the
/// axis of a ray at distance p from it and height z.
Interval angleFromAxis(const Interval p, const Interval z)
{
  const double p0 = std::max(p.lo, 0.0);
  if (z.lo >= 0.0)
  {
    return widened(std::atan2(p0, z.hi), std::atan2(p.hi, z.lo));
  }
  if (z.hi <= 0.0)
  {
    return widened(std::atan2(p.hi, z.hi), std::atan2(p0, z.lo));
  }
  return widened(std::atan2(p0, z.hi), std::atan2(p0, z.lo));
}

/// The output over patches of source rays: whether it can show them (its
/// domain) or keeps them (inside the horizon margin), and bounds on |q|_K
/// over the ones it shows (or keeps).
class Output
{
public:
  struct Patch
  {
    Tri shown;
    Tri kept;
    bool any;  // the bounds below cover at least one ray
    double upper;
    double lower;
    // The offsets of |q|_K as linear forms, where they were made.
    std::optional<std::pair<LinearForm, LinearForm>> forms;
  };

  Output(
    const CameraModel &unit_output, const ImageSize &out_size,
    const Eigen::Matrix3f &src_from_output
  )
  : type_(unit_output.projection.type),
    lat_long_(isLatLongProjection(unit_output.projection.type)),
    law_(unit_output),
    fx_(unit_output.intrinsics.fx),
    fy_(unit_output.intrinsics.fy),
    skew_(unit_output.intrinsics.skew),
    half_w_(0.5 * static_cast<double>(out_size.width - 1)),
    half_h_(0.5 * static_cast<double>(out_size.height - 1))
  {
    const Eigen::Matrix3d r = src_from_output.cast<double>().transpose();
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        rotation_[i][j] = r(i, j);
        axis_aligned_ = axis_aligned_ && (r(i, j) == 0.0 || std::abs(r(i, j)) == 1.0);
      }
    }
    // The offsets (a, b) = M q, M = [[fx, skew] / half_w, [0, fy / half_h]]:
    // its smallest singular value is at least |det M| / |M|_F.
    const double m00 = fx_ / half_w_;
    const double m01 = skew_ / half_w_;
    const double m11 = fy_ / half_h_;
    offset_floor_ = (1.0 - 1e-9) * std::abs(m00 * m11) /
                    (std::sqrt(m00 * m00 + m01 * m01 + m11 * m11) * std::sqrt(2.0));
    const double margin = kHorizonMargin;
    if (lat_long_)
    {
      // The projection fails at a pole: a ray whose distance from the
      // vertical axis is at most 1e-8 of its length (evaluated in float).
      // Rays clearly farther out are shown, clearly closer in are not, and
      // those in between are undecided.
      shown_latitude_ = std::acos(2e-8);
      unshown_latitude_ = std::acos(0.5e-8);
      kept_latitude_ =
        type_ == ProjectionModelType::CYLINDRICAL ? 0.5 * constants::kPi - margin : shown_latitude_;
      unkept_latitude_ =
        type_ == ProjectionModelType::CYLINDRICAL ? kept_latitude_ : unshown_latitude_;
      return;
    }
    const double limit = law_.limit();
    double kept = limit;
    switch (type_)
    {
      case ProjectionModelType::PINHOLE:
        kept = std::min(limit, 0.5 * constants::kPi - margin);
        break;
      case ProjectionModelType::STEREOGRAPHIC:
      case ProjectionModelType::EQUIDISTANT:
      case ProjectionModelType::EQUISOLID:
        kept = std::min(limit, constants::kPi - margin);
        break;
      case ProjectionModelType::OMNIDIRECTIONAL:
      case ProjectionModelType::DOUBLE_SPHERE:
      case ProjectionModelType::EUCM:
        kept = limit - margin;
        break;
      default:
        break;
    }
    shown_angle_ = limit;
    kept_angle_ = std::max(kept, 0.0);
    // A pinhole output shows every ray in front of it.
    pinhole_shown_ = std::numeric_limits<double>::infinity();
    pinhole_kept_ = std::tan(std::max(kept, 0.0));
  }

  /// Bounds that settle a patch (it is dropped): an upper one at most
  /// `upper_settles`, a lower one at least `lower_settles`. A patch settled
  /// by its direct bounds needs no mean value form.
  struct Settled
  {
    double upper_settles;
    double lower_settles;
  };

  /// The rays of a patch [a0, a1] x [b0, b1] of `chart`, with directions
  /// x, y, z (unit vectors when `unit`, else of any length).
  Patch evaluate(
    const Chart chart, const double a0, const double a1, const double b0, const double b1,
    const Interval x, const Interval y, const Interval z, const bool unit, const bool kept_only,
    const Settled settled
  ) const
  {
    if (type_ == ProjectionModelType::PINHOLE && !unit && !axis_aligned_)
    {
      return pinholeBox(x, y, kept_only);
    }
    const Interval xo = rotated(0, x, y, z);
    const Interval yo = rotated(1, x, y, z);
    const Interval zo = rotated(2, x, y, z);
    if (type_ == ProjectionModelType::PINHOLE)
    {
      return pinhole(xo, yo, zo, kept_only);  // depends on the direction only
    }
    // The others depend on the direction only too; angles are taken from
    // the unnormalized directions.
    const PatchBox box{chart, a0, a1, b0, b1, settled};
    return lat_long_ ? latLong(xo, yo, zo, kept_only, box) : symmetric(xo, yo, zo, kept_only, box);
  }

  /// The two offsets of |q|_K over a patch as linear forms; nullopt where
  /// q is not smooth over it.
  std::optional<std::pair<LinearForm, LinearForm>> offsetForms(
    const Chart chart, const double a0, const double a1, const double b0, const double b1
  ) const
  {
    if (type_ == ProjectionModelType::PINHOLE)
    {
      return std::nullopt;
    }
    return forms(PatchBox{chart, a0, a1, b0, b1, Settled{}});
  }

private:
  /// Row `row` of the rotation applied to (x, y, z); entries of 0 and +-1
  /// (an axis-aligned rotation) are applied exactly.
  Interval rotated(const int row, const Interval x, const Interval y, const Interval z) const
  {
    const Interval in[3] = {x, y, z};
    Interval sum{0.0, 0.0};
    bool first = true;
    for (int j = 0; j < 3; ++j)
    {
      const double c = rotation_[row][j];
      if (c == 0.0)
      {
        continue;
      }
      const Interval term = c == 1.0    ? in[j]
                            : c == -1.0 ? Interval{-in[j].hi, -in[j].lo}
                                        : in[j] * c;
      sum = first ? term : sum + term;
      first = false;
    }
    return sum;
  }

  /// The patch a source patch came from, for its mean value form.
  struct PatchBox
  {
    Chart chart;
    double a0;
    double a1;
    double b0;
    double b1;
    Settled settled;
  };

  /// |q|_K bounds from enclosures of q, tightened by the patch's mean value
  /// form when given (that form holds over the whole patch).
  /// `norm_lo`: a lower bound on |q| known otherwise.
  void bound(
    const Interval qx, const Interval qy, Patch &patch,
    const std::optional<std::pair<Interval, Interval>> &tight = std::nullopt,
    const double norm_lo = 0.0
  ) const
  {
    Interval a = (qx * fx_ + qy * skew_) * (1.0 / half_w_);
    Interval b = qy * (fy_ / half_h_);
    if (tight)
    {
      a = intersection(a, tight->first);
      b = intersection(b, tight->second);
    }
    patch.upper = std::max(maxAbs(a), maxAbs(b));
    // Also |q|_K >= |q| sigma_min / sqrt(2), which holds where skew mixes
    // an unbounded qy into a (rays near a pinhole output's horizon).
    patch.lower = std::max(
      {minAbs(a), minAbs(b), offset_floor_ * std::max(norm_lo, std::hypot(minAbs(qx), minAbs(qy)))}
    );
    patch.any = true;
  }

  /// The two offsets of |q|_K, (fx qx + skew qy) / half_w and fy qy / half_h,
  /// over a patch by their mean value form: their values at its centre plus
  /// enclosures of their derivatives over it times the offsets from the
  /// centre. Its error shrinks with the square of the patch size (the
  /// direct enclosures' only with the size), which settles extremes that lie
  /// along flat stretches of the image edge. nullopt where q is not smooth
  /// over the patch: it holds the output axis, a lat-long pole or seam, or
  /// reaches the end of the law's domain.
  std::optional<std::pair<Interval, Interval>> meanValue(
    const PatchBox &patch, const std::optional<std::pair<LinearForm, LinearForm>> &f
  ) const
  {
    if (!f)
    {
      return std::nullopt;
    }
    const Interval oa = Interval{patch.a0, patch.a1} + (-0.5 * (patch.a0 + patch.a1));
    const Interval ob = Interval{patch.b0, patch.b1} + (-0.5 * (patch.b0 + patch.b1));
    return std::pair{f->first.over(oa, ob), f->second.over(oa, ob)};
  }

  /// The two offsets of |q|_K over a patch as linear forms (see meanValue).
  std::optional<std::pair<LinearForm, LinearForm>> forms(const PatchBox &patch) const
  {
    const Chart chart = patch.chart;
    const double a0 = patch.a0;
    const double a1 = patch.a1;
    const double b0 = patch.b0;
    const double b1 = patch.b1;
    const double ac = 0.5 * (a0 + a1);
    const double bc = 0.5 * (b0 + b1);
    const auto rotate = [&](DirectionJet j) {
      for (Interval *v : {j.v, j.da, j.db})
      {
        const Interval x = v[0];
        const Interval y = v[1];
        const Interval z = v[2];
        v[0] = rotated(0, x, y, z);
        v[1] = rotated(1, x, y, z);
        v[2] = rotated(2, x, y, z);
      }
      return j;
    };
    const DirectionJet box = rotate(chartDirection(chart, Interval{a0, a1}, Interval{b0, b1}));
    const DirectionJet centre = rotate(chartDirection(chart, Interval{ac, ac}, Interval{bc, bc}));
    Interval q[2];    // at the centre
    Interval q_a[2];  // derivatives over the box
    Interval q_b[2];
    if (!(lat_long_ ? latLongSlopes(box, centre, q, q_a, q_b)
                    : symmetricSlopes(box, centre, q, q_a, q_b)))
    {
      return std::nullopt;
    }
    const auto offsets = [&](const Interval *v) {
      return std::pair{(v[0] * fx_ + v[1] * skew_) * (1.0 / half_w_), v[1] * (fy_ / half_h_)};
    };
    const auto [a, b] = offsets(q);
    const auto [a_a, b_a] = offsets(q_a);
    const auto [a_b, b_b] = offsets(q_b);
    const std::pair<LinearForm, LinearForm> forms{LinearForm{a, a_a, a_b}, LinearForm{b, b_a, b_b}};
    for (const LinearForm &f : {forms.first, forms.second})
    {
      for (const Interval &i : {f.c, f.da, f.db})
      {
        if (!std::isfinite(i.lo) || !std::isfinite(i.hi))
        {
          return std::nullopt;
        }
      }
    }
    return forms;
  }

  /// q = rho(theta) (x, y) / p (p = |(x, y)|, theta = atan2(p, z)) at the
  /// centre, and its derivatives over the box:
  /// dq = rho'(theta) dtheta u + rho(theta) dphi n, with u = (x, y) / p, n
  /// its normal, dtheta = (z dp - p dz) / |w|^2, dphi = (x dy - y dx) / p^2.
  bool symmetricSlopes(
    const DirectionJet &box, const DirectionJet &centre, Interval (&q)[2], Interval (&q_a)[2],
    Interval (&q_b)[2]
  ) const
  {
    const auto planar = [](const DirectionJet &j) {
      return squareRoot(square(j.v[0]) + square(j.v[1]));
    };
    const Interval p = planar(box);
    const Interval pc = planar(centre);
    if (!(p.lo > 0.0) || !(pc.lo > 0.0))
    {
      return false;
    }
    const Interval theta = angleFromAxis(p, box.v[2]);
    const Interval theta_c = angleFromAxis(pc, centre.v[2]);
    if (!(theta.hi < law_.limit()))
    {
      return false;
    }
    const Interval rho = law_.radius(std::max(theta.lo, 0.0), theta.hi);
    const Interval rho_c = law_.radius(std::max(theta_c.lo, 0.0), theta_c.hi);
    const auto rho_slope = law_.slope(theta);
    if (!rho_slope)
    {
      return false;
    }
    q[0] = rho_c * clamped(*divide(centre.v[0], pc), -1.0, 1.0);
    q[1] = rho_c * clamped(*divide(centre.v[1], pc), -1.0, 1.0);
    const Interval ux = clamped(*divide(box.v[0], p), -1.0, 1.0);
    const Interval uy = clamped(*divide(box.v[1], p), -1.0, 1.0);
    const Interval p2 = square(box.v[0]) + square(box.v[1]);
    const Interval w2 = p2 + square(box.v[2]);
    const auto along = [&](const Interval *d, Interval(&out)[2]) {
      const auto dp = divide(box.v[0] * d[0] + box.v[1] * d[1], p);
      const auto dtheta = dp ? divide(box.v[2] * *dp - p * d[2], w2) : std::nullopt;
      const auto dphi = divide(box.v[0] * d[1] - box.v[1] * d[0], p2);
      if (!dtheta || !dphi)
      {
        return false;
      }
      const Interval radial = *rho_slope * *dtheta;
      const Interval turn = rho * *dphi;
      out[0] = radial * ux - turn * uy;
      out[1] = radial * uy + turn * ux;
      return true;
    };
    return along(box.da, q_a) && along(box.db, q_b);
  }

  /// q = (longitude, latitude), or (longitude, tan latitude) for the
  /// cylinder, at the centre and its derivatives over the box, away from the
  /// poles and the seam.
  bool latLongSlopes(
    const DirectionJet &box, const DirectionJet &centre, Interval (&q)[2], Interval (&q_a)[2],
    Interval (&q_b)[2]
  ) const
  {
    const Interval &x = box.v[0];
    const Interval &y = box.v[1];
    const Interval &z = box.v[2];
    if (!(z.lo > 0.0 || x.lo > 0.0 || x.hi < 0.0))
    {
      return false;  // may cross the seam
    }
    const auto across = [](const DirectionJet &j) {
      return squareRoot(square(j.v[0]) + square(j.v[2]));
    };
    const Interval h = across(box);
    const Interval hc = across(centre);
    if (!(h.lo > 0.0) || !(hc.lo > 0.0))
    {
      return false;
    }
    const bool cylinder = type_ == ProjectionModelType::CYLINDRICAL;
    const Interval &xc = centre.v[0];
    const Interval &zc = centre.v[2];
    const double c[4] = {
      std::atan2(xc.lo, zc.lo), std::atan2(xc.lo, zc.hi), std::atan2(xc.hi, zc.lo),
      std::atan2(xc.hi, zc.hi)};
    q[0] = widened(*std::min_element(c, c + 4), *std::max_element(c, c + 4));
    const auto ratio = divide(centre.v[1], hc);
    if (!ratio)
    {
      return false;
    }
    q[1] = cylinder ? *ratio : widened(std::atan(ratio->lo), std::atan(ratio->hi));
    const Interval h2 = square(x) + square(z);
    const auto along = [&](const Interval *d, Interval(&out)[2]) {
      const auto dlon = divide(z * d[0] - x * d[2], h2);
      const auto dh = divide(x * d[0] + z * d[2], h);
      if (!dlon || !dh)
      {
        return false;
      }
      const Interval rise = d[1] * h - y * *dh;
      const auto dlat = cylinder ? divide(rise, h2) : divide(rise, h2 + square(y));
      if (!dlat)
      {
        return false;
      }
      out[0] = *dlon;
      out[1] = *dlat;
      return true;
    };
    return along(box.da, q_a) && along(box.db, q_b);
  }

  /// A box x x y of rays (x, y, 1) into a pinhole output: q is a linear
  /// fractional function of (x, y), monotone in each while the rays stay in
  /// front of the output, so its extremes over the box are at the corners
  /// (much tighter than dividing enclosures when the box is large).
  Patch pinholeBox(const Interval x, const Interval y, const bool kept_only) const
  {
    const Interval one{1.0, 1.0};
    Interval qx{std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    Interval qy = qx;
    for (const double cx : {x.lo, x.hi})
    {
      for (const double cy : {y.lo, y.hi})
      {
        const Interval px{cx, cx};
        const Interval py{cy, cy};
        const Interval zo = rotated(2, px, py, one);
        if (!(zo.lo > 0.0))
        {
          return pinhole(
            rotated(0, x, y, one), rotated(1, x, y, one), rotated(2, x, y, one), kept_only
          );
        }
        const Interval cqx = *divide(rotated(0, px, py, one), zo);
        const Interval cqy = *divide(rotated(1, px, py, one), zo);
        qx = Interval{std::min(qx.lo, cqx.lo), std::max(qx.hi, cqx.hi)};
        qy = Interval{std::min(qy.lo, cqy.lo), std::max(qy.hi, cqy.hi)};
      }
    }
    return pinholeQ(qx, qy, true, kept_only);
  }

  /// q = (x / z, y / z); shown while |q| <= tan(limit), kept while
  /// |q| <= cot(margin).
  Patch pinhole(const Interval xo, const Interval yo, const Interval zo, const bool kept_only) const
  {
    if (!(zo.hi > 0.0))
    {
      return Patch{};  // behind the camera
    }
    const Interval zc{std::max(zo.lo, zo.hi * 1e-300), zo.hi};
    return pinholeQ(*divide(xo, zc), *divide(yo, zc), zo.lo > 0.0, kept_only);
  }

  /// From enclosures of q over the rays in front of the output (all of
  /// them when `in_front`).
  Patch pinholeQ(Interval qx, Interval qy, const bool in_front, const bool kept_only) const
  {
    Patch patch{};
    const Interval q2 = square(qx) + square(qy);
    const auto within = [&](const double radius) {
      const double r2 = radius * radius;
      if (in_front && q2.hi <= r2)
      {
        return Tri::YES;
      }
      return q2.lo > r2 ? Tri::NO : Tri::MAYBE;
    };
    patch.shown = within(pinhole_shown_);
    patch.kept = within(pinhole_kept_);
    const double radius = kept_only ? pinhole_kept_ : pinhole_shown_;
    if (q2.lo > radius * radius)
    {
      return patch;
    }
    // Over the part inside that radius, each coordinate is within the
    // radius left by the other's smallest magnitude.
    const Interval rad{radius, radius};
    const auto cap = [&](const double m) {
      return squareRoot((rad + Interval{-m, -m}) * (rad + Interval{m, m})).hi;
    };
    const double cap_x = std::min(radius, cap(std::min(minAbs(qy), radius)));
    const double cap_y = std::min(radius, cap(std::min(minAbs(qx), radius)));
    bound(clamped(qx, -cap_x, cap_x), clamped(qy, -cap_y, cap_y), patch);
    return patch;
  }

  /// Whether a (non-point) patch's direct bounds leave it to be refined.
  static bool open(const Patch &patch, const PatchBox &box)
  {
    return !(box.a0 == box.a1 && box.b0 == box.b1) && patch.upper > box.settled.upper_settles &&
           patch.lower < box.settled.lower_settles;
  }

  /// The mean value form of a patch whose direct enclosures span less than
  /// this (radians) in both angles: a larger one gains nothing from it.
  static constexpr double kMeanValueSpan = 0.25;

  Patch symmetric(
    const Interval xo, const Interval yo, const Interval zo, const bool kept_only,
    const PatchBox &box
  ) const
  {
    Patch patch{};
    const Interval planar = squareRoot(square(xo) + square(yo));
    const Interval theta = angleFromAxis(planar, zo);
    const auto within = [&](const double limit) {
      return theta.hi <= limit ? Tri::YES : (theta.lo > limit ? Tri::NO : Tri::MAYBE);
    };
    patch.shown = within(shown_angle_);
    patch.kept = within(kept_angle_);
    const double limit = kept_only ? kept_angle_ : shown_angle_;
    if (theta.lo > limit)
    {
      return patch;
    }
    const Interval rho = law_.radius(std::max(theta.lo, 0.0), std::min(theta.hi, limit));
    Interval dx{-1.0, 1.0};
    Interval dy{-1.0, 1.0};
    double turn = 2.0 * constants::kPi;
    if (planar.lo > 0.0)
    {
      // Azimuths of the box (xo, yo), which does not hold the axis.
      double c[4] = {
        std::atan2(yo.lo, xo.lo), std::atan2(yo.lo, xo.hi), std::atan2(yo.hi, xo.lo),
        std::atan2(yo.hi, xo.hi)};
      if (xo.hi < 0.0 && yo.lo < 0.0 && yo.hi >= 0.0)
      {
        for (double &a : c)
        {
          a = a < 0.0 ? a + 2.0 * constants::kPi : a;  // across the negative x axis
        }
      }
      const Interval phi = widened(*std::min_element(c, c + 4), *std::max_element(c, c + 4));
      dx = cosine(phi);
      dy = sine(phi);
      turn = phi.hi - phi.lo;
    }
    // |q| is rho itself.
    bound(rho * dx, rho * dy, patch, std::nullopt, rho.lo);
    if (theta.hi - theta.lo < kMeanValueSpan && turn < kMeanValueSpan && open(patch, box))
    {
      patch.forms = forms(box);
      if (const auto tight = meanValue(box, patch.forms))
      {
        bound(rho * dx, rho * dy, patch, tight, rho.lo);
      }
    }
    return patch;
  }

  Patch latLong(
    const Interval xo, const Interval yo, const Interval zo, const bool kept_only,
    const PatchBox &box
  ) const
  {
    Patch patch{};
    // Latitude atan2(y, h), h = |(x, z)|: increasing in y, and away from the
    // equator as h shrinks.
    const Interval h = squareRoot(square(xo) + square(zo));
    const double h0 = std::max(h.lo, 0.0);
    const Interval lat = widened(
      std::atan2(yo.lo, yo.lo < 0.0 ? h0 : h.hi), std::atan2(yo.hi, yo.hi > 0.0 ? h0 : h.hi)
    );
    const auto within = [&](const double yes, const double no) {
      return lat.lo >= -yes && lat.hi <= yes ? Tri::YES
                                             : (lat.hi < -no || lat.lo > no ? Tri::NO : Tri::MAYBE);
    };
    patch.shown = within(shown_latitude_, unshown_latitude_);
    patch.kept = within(kept_latitude_, unkept_latitude_);
    const double limit = kept_only ? unkept_latitude_ : unshown_latitude_;
    if (lat.hi < -limit || lat.lo > limit)
    {
      return patch;
    }
    const Interval lat_c{std::max(lat.lo, -limit), std::min(lat.hi, limit)};
    const Interval qy = type_ == ProjectionModelType::CYLINDRICAL
                          ? widened(std::tan(lat_c.lo), std::tan(lat_c.hi))
                          : lat_c;
    // Longitude atan2(x, z): over a patch away from the origin and from the
    // seam (x = 0, z < 0) its extremes are at the corners; across the seam
    // it lies within `a` of +-pi.
    std::vector<Interval> lons;
    if (zo.lo > 0.0 || xo.lo > 0.0 || xo.hi < 0.0)
    {
      const double c[4] = {
        std::atan2(xo.lo, zo.lo), std::atan2(xo.lo, zo.hi), std::atan2(xo.hi, zo.lo),
        std::atan2(xo.hi, zo.hi)};
      lons.push_back(widened(*std::min_element(c, c + 4), *std::max_element(c, c + 4)));
    }
    else if (zo.hi < 0.0)
    {
      const double a = std::atan2(maxAbs(xo), -zo.hi);
      lons.push_back(widened(-constants::kPi, -constants::kPi + a));
      lons.push_back(widened(constants::kPi - a, constants::kPi));
    }
    else
    {
      lons.push_back(Interval{-constants::kPi, constants::kPi});
    }
    patch.upper = 0.0;
    patch.lower = std::numeric_limits<double>::infinity();
    for (const Interval &qx : lons)
    {
      Patch part{};
      bound(clamped(qx, -constants::kPi, constants::kPi), qy, part);
      patch.upper = std::max(patch.upper, part.upper);
      patch.lower = std::min(patch.lower, part.lower);
    }
    patch.any = true;
    if (lons.size() == 1 && lons[0].hi - lons[0].lo < kMeanValueSpan && lat.hi - lat.lo < kMeanValueSpan && open(patch, box))
    {
      patch.forms = forms(box);
      if (const auto tight = meanValue(box, patch.forms))
      {
        bound(clamped(lons[0], -constants::kPi, constants::kPi), qy, patch, tight);
      }
    }
    return patch;
  }

  ProjectionModelType type_;
  bool lat_long_;
  RadialLaw law_;
  double fx_;
  double fy_;
  double skew_;
  double half_w_;
  double half_h_;
  double rotation_[3][3]{};
  bool axis_aligned_{true};
  double shown_angle_{0.0};
  double kept_angle_{0.0};
  double offset_floor_{0.0};
  double pinhole_shown_{0.0};
  double pinhole_kept_{0.0};
  double shown_latitude_{0.0};
  double unshown_latitude_{0.0};
  double kept_latitude_{0.0};
  double unkept_latitude_{0.0};
};

// ===========================================================================
// Branch and bound over the sphere of directions
// ===========================================================================

struct Node
{
  Chart chart;
  double a0;
  double a1;
  double b0;
  double b1;
  double bound;
  // The patch's pixel enclosure straddles the image's left or right (1) or
  // top or bottom (2) edge, and is this wide (pixels) across those edges.
  std::uint8_t straddles;
  double spread;
};

/// Records which edges of [u0, u1] x [v0, v1] a patch's pixel enclosure
/// straddles.
void markStraddles(Node &n, const Interval u, const Interval v, const Tri in_u, const Tri in_v)
{
  n.straddles =
    static_cast<std::uint8_t>((in_u == Tri::MAYBE ? 1U : 0U) | (in_v == Tri::MAYBE ? 2U : 0U));
  n.spread = (in_u == Tri::MAYBE ? u.hi - u.lo : 0.0) + (in_v == Tri::MAYBE ? v.hi - v.lo : 0.0);
}

/// Where to split [lo, hi] on the plane chart: at zero across it, on a log
/// scale far from it, else in the middle.
double planeSplit(const double lo, const double hi)
{
  if (lo < 0.0 && hi > 0.0)
  {
    return 0.0;
  }
  if (lo >= 0.0)
  {
    return hi > 4.0 * (lo + 1.0) ? std::sqrt((lo + 1.0) * (hi + 1.0)) - 1.0 : 0.5 * (lo + hi);
  }
  return -planeSplit(-hi, -lo);
}

/// Splits a patch across its longer side, or across the image edge it
/// straddles: the side along which its source pixel moves most across that
/// edge (by finite differences at its centre), unless its pixel enclosure is
/// mostly overestimation (then the longer side, which shrinks that).
std::pair<Node, Node> split(const Node &n, const Source &source)
{
  bool first = true;  // split a (else b)
  double m = 0.0;
  if (n.straddles != 0)
  {
    const double a = 0.5 * (n.a0 + n.a1);
    const double b = 0.5 * (n.b0 + n.b1);
    const double ha = 0.25 * (n.a1 - n.a0);
    const double hb = 0.25 * (n.b1 - n.b0);
    const auto c = source.pixelAt(n.chart, a, b);
    const auto pa = source.pixelAt(n.chart, a + ha, b);
    const auto pb = source.pixelAt(n.chart, a, b + hb);
    if (c && pa && pb)
    {
      const Eigen::Vector2f da = *pa - *c;
      const Eigen::Vector2f db = *pb - *c;
      const float ga = ((n.straddles & 1U) ? std::abs(da.x()) : 0.0f) +
                       ((n.straddles & 2U) ? std::abs(da.y()) : 0.0f);
      const float gb = ((n.straddles & 1U) ? std::abs(db.x()) : 0.0f) +
                       ((n.straddles & 2U) ? std::abs(db.y()) : 0.0f);
      // The steps are a quarter of each side.
      if (std::isfinite(ga) && std::isfinite(gb) && ga + gb > 0.0f && n.spread <= 8.0 * (ga + gb))
      {
        first = ga >= gb;
        m = first ? (n.chart == Chart::PLANE ? planeSplit(n.a0, n.a1) : 0.5 * (n.a0 + n.a1))
                  : (n.chart == Chart::PLANE ? planeSplit(n.b0, n.b1) : 0.5 * (n.b0 + n.b1));
        return first
                 ? std::pair<
                     Node,
                     Node>{Node{n.chart, n.a0, m, n.b0, n.b1, 0.0, 0, 0.0}, Node{n.chart, m, n.a1, n.b0, n.b1, 0.0, 0, 0.0}}
                 : std::pair<Node, Node>{
                     Node{n.chart, n.a0, n.a1, n.b0, m, 0.0, 0, 0.0},
                     Node{n.chart, n.a0, n.a1, m, n.b1, 0.0, 0, 0.0}};
      }
    }
  }
  switch (n.chart)
  {
    case Chart::PLANE: {
      const double wa = (n.a1 - n.a0) / (1.0 + std::max(std::abs(n.a0), std::abs(n.a1)));
      const double wb = (n.b1 - n.b0) / (1.0 + std::max(std::abs(n.b0), std::abs(n.b1)));
      first = wa >= wb;
      m = first ? planeSplit(n.a0, n.a1) : planeSplit(n.b0, n.b1);
      break;
    }
    case Chart::POLAR: {
      // a: angle from the axis, b: azimuth; azimuth arcs grow with sin(a).
      const double half_pi = 0.5 * constants::kPi;
      const double widest =
        n.a0 <= half_pi && n.a1 >= half_pi ? 1.0 : std::max(std::sin(n.a0), std::sin(n.a1));
      first = n.a1 - n.a0 >= (n.b1 - n.b0) * widest;
      m = first ? 0.5 * (n.a0 + n.a1) : 0.5 * (n.b0 + n.b1);
      break;
    }
    case Chart::LAT_LONG: {
      // a: longitude, b: latitude; longitude arcs shrink by cos(latitude).
      const double widest =
        n.b0 <= 0.0 && n.b1 >= 0.0 ? 1.0 : std::max(std::cos(n.b0), std::cos(n.b1));
      first = (n.a1 - n.a0) * widest > n.b1 - n.b0;
      m = first ? 0.5 * (n.a0 + n.a1) : 0.5 * (n.b0 + n.b1);
      break;
    }
  }
  if (first)
  {
    return {
      Node{n.chart, n.a0, m, n.b0, n.b1, 0.0, 0, 0.0},
      Node{n.chart, m, n.a1, n.b0, n.b1, 0.0, 0, 0.0}};
  }
  return {
    Node{n.chart, n.a0, n.a1, n.b0, m, 0.0, 0, 0.0},
    Node{n.chart, n.a0, n.a1, m, n.b1, 0.0, 0, 0.0}};
}

bool tooSmall(const Node &n)
{
  const double scale =
    n.chart == Chart::PLANE
      ? 1.0 + std::max({std::abs(n.a0), std::abs(n.a1), std::abs(n.b0), std::abs(n.b1)})
      : 1.0;
  return n.a1 - n.a0 < kSmallestPatch * scale && n.b1 - n.b0 < kSmallestPatch * scale;
}

/// A grid of na x nb patches over [a0, a1] x [b0, b1].
void addGrid(
  std::vector<Node> &nodes, const Chart chart, const double a0, const double a1, const double b0,
  const double b1, const int na, const int nb
)
{
  // The ends exactly (interpolating them would lose a small end next to a
  // huge one).
  const auto at = [](const double lo, const double hi, const int k, const int n) {
    return k == 0 ? lo : (k == n ? hi : lo + (hi - lo) * k / n);
  };
  for (int i = 0; i < na; ++i)
  {
    for (int j = 0; j < nb; ++j)
    {
      nodes.push_back(Node{
        chart, at(a0, a1, i, na), at(a0, a1, i + 1, na), at(b0, b1, j, nb), at(b0, b1, j + 1, nb),
        0.0, 0, 0.0});
    }
  }
}

/// The starting patches: every direction the source might show (`all`
/// false), or every direction (`all` true).
std::vector<Node> startingPatches(const Source &source, const bool all)
{
  std::vector<Node> nodes;
  const double two_pi = 2.0 * constants::kPi;
  if (source.latLong())
  {
    addGrid(
      nodes, Chart::LAT_LONG, -constants::kPi, constants::kPi, -0.5 * constants::kPi,
      0.5 * constants::kPi, 16, 8
    );
    return nodes;
  }
  if (source.pinholeLaw())
  {
    // Boxes cut at the axes and, when that is closer, where the rays are
    // all lost (those out there are needed only for every direction).
    const double e = Source::planeExtent();
    const double l = source.lostFrom();
    std::vector<double> cuts{-e, 0.0, e};
    if (l < e)
    {
      cuts = all ? std::vector<double>{-e, -l, 0.0, l, e} : std::vector<double>{-l, 0.0, l};
    }
    for (std::size_t i = 0; i + 1 < cuts.size(); ++i)
    {
      for (std::size_t j = 0; j + 1 < cuts.size(); ++j)
      {
        addGrid(nodes, Chart::PLANE, cuts[i], cuts[i + 1], cuts[j], cuts[j + 1], 1, 1);
      }
    }
    if (all)
    {
      addGrid(nodes, Chart::POLAR, std::atan(e), constants::kPi, 0.0, two_pi, 4, 16);
    }
    return nodes;
  }
  addGrid(
    nodes, Chart::POLAR, 0.0, all ? constants::kPi : source.thetaEnd(), 0.0, two_pi, all ? 16 : 8,
    all ? 32 : 16
  );
  return nodes;
}

/// Bounds of |q|_K over just one side of the image edges a patch's pixel
/// enclosure straddles (n.straddles), from linear forms of its pixel and of
/// the offsets: with `inside`, the largest |q|_K where the pixel may lie
/// inside [u0, u1] x [v0, v1] (-infinity where it cannot), else the smallest
/// where it may lie outside (infinity where it cannot). Near a flat stretch
/// of the edge this is tight however long the patch runs along it. nullopt
/// where the forms are not available.
std::optional<double> sideBound(
  const Node &n, const Source &source, const Output &output, const Output::Patch &o,
  const double u0, const double u1, const double v0, const double v1, const bool inside
)
{
  if (n.a0 == n.a1 && n.b0 == n.b1)
  {
    return std::nullopt;
  }
  const auto pixel = source.pixelForms(n.chart, n.a0, n.a1, n.b0, n.b1);
  const auto offsets =
    !pixel ? std::nullopt
           : (o.forms ? o.forms : output.offsetForms(n.chart, n.a0, n.a1, n.b0, n.b1));
  if (!offsets)
  {
    return std::nullopt;
  }
  const Interval oa = Interval{n.a0, n.a1} + (-0.5 * (n.a0 + n.a1));
  const Interval ob = Interval{n.b0, n.b1} + (-0.5 * (n.b0 + n.b1));
  HalfPlane edges[4];
  int count = 0;
  if ((n.straddles & 1U) != 0)
  {
    edges[count++] = HalfPlane{pixel->first, u0, inside};
    edges[count++] = HalfPlane{pixel->first, u1, !inside};
  }
  if ((n.straddles & 2U) != 0)
  {
    edges[count++] = HalfPlane{pixel->second, v0, inside};
    edges[count++] = HalfPlane{pixel->second, v1, !inside};
  }
  const double infinity = std::numeric_limits<double>::infinity();
  if (inside)
  {
    double largest = -infinity;
    for (const LinearForm &f : {offsets->first, offsets->second})
    {
      const auto hi = extremeOver(f, false, oa, ob, edges, count);
      const auto lo = extremeOver(f, true, oa, ob, edges, count);
      if (hi && lo)
      {
        largest = std::max({largest, *hi, -*lo});
      }
    }
    return largest;
  }
  double smallest = infinity;
  for (int k = 0; k < count; ++k)
  {
    double here = 0.0;
    bool any = false;
    for (const LinearForm &f : {offsets->first, offsets->second})
    {
      const auto lo = extremeOver(f, true, oa, ob, &edges[k], 1);
      const auto hi = extremeOver(f, false, oa, ob, &edges[k], 1);
      if (lo && hi)
      {
        any = true;
        here = std::max(here, *lo > 0.0 ? *lo : (*hi < 0.0 ? -*hi : 0.0));
      }
    }
    smallest = any ? std::min(smallest, here) : smallest;
  }
  return smallest;
}

// alpha = 1: an upper bound, within kTolerance, on |q|_K over the source
// rays that are valid, land inside the source image (between its outermost
// pixel centres) and that the output keeps; nullopt when there are none.
std::optional<double> largestKeptOffset(
  const Source &source, const ImageSize &src_size, const Output &output, const int max_patches
)
{
  const double u_end = static_cast<double>(src_size.width - 1);
  const double v_end = static_cast<double>(src_size.height - 1);
  double attained = 0.0;  // |q|_K of a kept ray
  // The bound over a patch's possibly kept rays, or nullopt when it has none.
  const auto examine = [&](Node &n, bool &certain) -> std::optional<double> {
    const Source::Patch s = source.evaluate(n.chart, n.a0, n.a1, n.b0, n.b1);
    if (s.valid == Tri::NO)
    {
      return std::nullopt;
    }
    Tri inside = Tri::MAYBE;
    if (s.lost)
    {
      inside = Tri::NO;
    }
    else if (s.has_pixel)
    {
      const Tri in_u = insideRange(s.u, 0.0, u_end);
      const Tri in_v = insideRange(s.v, 0.0, v_end);
      inside = both(in_u, in_v);
      markStraddles(n, s.u, s.v, in_u, in_v);
    }
    if (inside == Tri::NO)
    {
      return std::nullopt;
    }
    const Output::Patch o = output.evaluate(
      n.chart, n.a0, n.a1, n.b0, n.b1, s.x, s.y, s.z, s.unit, true,
      Output::Settled{attained * (1.0 + kTolerance), std::numeric_limits<double>::infinity()}
    );
    if (!o.any || o.kept == Tri::NO)
    {
      return std::nullopt;
    }
    certain = s.valid == Tri::YES && inside == Tri::YES && o.kept == Tri::YES;
    if (certain && n.a0 == n.a1 && n.b0 == n.b1)
    {
      return o.lower;
    }
    double bound = o.upper;
    if (n.straddles != 0 && bound > attained * (1.0 + kTolerance))
    {
      if (const auto side = sideBound(n, source, output, o, 0.0, u_end, 0.0, v_end, true))
      {
        bound = std::min(bound, *side);
      }
    }
    return bound >= 0.0 ? std::optional<double>(bound) : std::nullopt;
  };
  const auto probe = [&](const Node &n) {
    const double a = 0.5 * (n.a0 + n.a1);
    const double b = 0.5 * (n.b0 + n.b1);
    bool certain = false;
    Node point{n.chart, a, a, b, b, 0.0, 0, 0.0};
    const auto value = examine(point, certain);
    if (value && certain)
    {
      attained = std::max(attained, *value);
    }
  };

  const auto later = [](const Node &l, const Node &r) { return l.bound < r.bound; };
  std::priority_queue<Node, std::vector<Node>, decltype(later)> queue(later);
  const auto offer = [&](Node n) {
    bool certain = false;
    const auto bound = examine(n, certain);
    if (bound && *bound > attained * (1.0 + kTolerance))
    {
      probe(n);
      n.bound = *bound;
      queue.push(n);
    }
  };
  for (const Node &n : startingPatches(source, false))
  {
    offer(n);
  }
  double result = 0.0;
  for (int patches = 0;; ++patches)
  {
    if (queue.empty())
    {
      result = attained * (1.0 + kTolerance);
      break;
    }
    const Node top = queue.top();
    if (top.bound <= attained * (1.0 + kTolerance) || tooSmall(top) || patches >= max_patches)
    {
      // Every patch left is bounded by `top`; anything dropped was bounded
      // by the tolerance band above `attained`.
      result = std::max(top.bound, attained * (1.0 + kTolerance));
      break;
    }
    queue.pop();
    const auto [first, second] = split(top, source);
    offer(first);
    offer(second);
  }
  if (!(result > 0.0) || !std::isfinite(result))
  {
    return std::nullopt;
  }
  return result;
}

// alpha = 0: a lower bound, within kTolerance, on |q|_K over the rays the
// output can show that the source does not: rays that are not valid source
// directions or land less than `margin` inside the source image. Infinity
// when there are none.
double smallestUnshownOffset(
  const Source &source, const ImageSize &src_size, const float margin, const Output &output,
  const int max_patches
)
{
  const double m = static_cast<double>(margin);
  const double u0 = m;
  const double u1 = static_cast<double>(src_size.width - 1) - m;
  const double v0 = m;
  const double v1 = static_cast<double>(src_size.height - 1) - m;
  const double infinity = std::numeric_limits<double>::infinity();
  double attained = infinity;  // |q|_K of a ray the output shows but the source does not
  const auto examine = [&](Node &n, bool &certain) -> std::optional<double> {
    const Source::Patch s = source.evaluate(n.chart, n.a0, n.a1, n.b0, n.b1);
    Tri inside = Tri::MAYBE;
    if (s.lost || !(u0 <= u1 && v0 <= v1))
    {
      inside = Tri::NO;
    }
    else if (s.has_pixel)
    {
      const Tri in_u = insideRange(s.u, u0, u1);
      const Tri in_v = insideRange(s.v, v0, v1);
      inside = both(in_u, in_v);
      markStraddles(n, s.u, s.v, in_u, in_v);
    }
    if (s.valid == Tri::YES && inside == Tri::YES)
    {
      return std::nullopt;
    }
    const Output::Patch o = output.evaluate(
      n.chart, n.a0, n.a1, n.b0, n.b1, s.x, s.y, s.z, s.unit, false,
      Output::Settled{-std::numeric_limits<double>::infinity(), attained / (1.0 + kTolerance)}
    );
    if (!o.any || o.shown == Tri::NO)
    {
      return std::nullopt;
    }
    certain = (s.valid == Tri::NO || inside == Tri::NO) && o.shown == Tri::YES;
    if (certain && n.a0 == n.a1 && n.b0 == n.b1)
    {
      return o.upper;
    }
    double bound = o.lower;
    // With every ray valid, only the rays past the straddled edges are bad.
    if (n.straddles != 0 && s.valid == Tri::YES && bound < attained / (1.0 + kTolerance))
    {
      if (const auto side = sideBound(n, source, output, o, u0, u1, v0, v1, false))
      {
        bound = std::max(bound, *side);
      }
    }
    return bound < infinity ? std::optional<double>(bound) : std::nullopt;
  };
  const auto probe = [&](const Node &n) {
    const double a = 0.5 * (n.a0 + n.a1);
    const double b = 0.5 * (n.b0 + n.b1);
    bool certain = false;
    Node point{n.chart, a, a, b, b, 0.0, 0, 0.0};
    const auto value = examine(point, certain);
    if (value && certain)
    {
      attained = std::min(attained, *value);
    }
  };

  const auto later = [](const Node &l, const Node &r) { return l.bound > r.bound; };
  std::priority_queue<Node, std::vector<Node>, decltype(later)> queue(later);
  const auto offer = [&](Node n) {
    bool certain = false;
    const auto bound = examine(n, certain);
    if (bound && *bound < attained / (1.0 + kTolerance))
    {
      probe(n);
      n.bound = *bound;
      queue.push(n);
    }
  };
  for (const Node &n : startingPatches(source, true))
  {
    offer(n);
  }
  for (int patches = 0;; ++patches)
  {
    if (queue.empty())
    {
      return attained / (1.0 + kTolerance);
    }
    const Node top = queue.top();
    if (top.bound >= attained / (1.0 + kTolerance) || tooSmall(top) || patches >= max_patches)
    {
      return std::min(top.bound, attained / (1.0 + kTolerance));
    }
    queue.pop();
    const auto [first, second] = split(top, source);
    offer(first);
    offer(second);
  }
}

/// The smallest float scale (to a relative 1e-7) at which the output's
/// corners pass `corners_ok`, which holds for large scales and is monotone.
/// A passing and a failing scale are found by stepping out from `start` by
/// factors of 16, so the result is one that passed; nullopt when no float
/// scale passes.
template <typename CornersOk>
std::optional<float> smallestScaleWith(const CornersOk &corners_ok, const float start)
{
  constexpr float kStep = 16.0f;
  float passes = start;
  float fails = start;
  if (corners_ok(start))
  {
    for (;;)
    {
      const float next = passes / kStep;
      if (!(next >= std::numeric_limits<float>::min()))
      {
        return passes;  // passes down to the smallest scales
      }
      if (!corners_ok(next))
      {
        fails = next;
        break;
      }
      passes = next;
    }
  }
  else
  {
    for (;;)
    {
      const float next = fails * kStep;
      if (!std::isfinite(next))
      {
        return std::nullopt;
      }
      if (corners_ok(next))
      {
        passes = next;
        break;
      }
      fails = next;
    }
  }
  for (int step = 0; step < 200 && passes > fails * (1.0f + 1e-7f); ++step)
  {
    const float mid = std::sqrt(passes) * std::sqrt(fails);
    if (!(mid < passes && mid > fails))
    {
      break;
    }
    (corners_ok(mid) ? passes : fails) = mid;
  }
  return passes;
}

}  // namespace

CameraModel scaleOutputModel(const CameraModel &unit_output, const float scale)
{
  CameraModel model = unit_output;
  model.intrinsics.fx *= scale;
  model.intrinsics.fy *= scale;
  model.intrinsics.skew *= scale;
  return model;
}

RectifyScaleFit fitRectifyScale(
  const CameraModel &src_model, const ImageSize &src_size, const CameraModel &unit_output,
  const ImageSize &out_size, const Eigen::Matrix3f &src_from_output, const float source_margin_px,
  const SolverOptions &solver_options, const bool need_inscribed, const bool need_contained,
  const int max_patches
)
{
  const Source source(src_model, src_size, source_margin_px);
  const Output output(unit_output, out_size, src_from_output);
  RectifyScaleFit fit{};
  if (need_contained)
  {
    if (const auto offset = largestKeptOffset(source, src_size, output, max_patches))
    {
      const double scale = 1.0 / *offset / kContainedSafety;
      if (std::isfinite(scale) && scale > 0.0 && scale < std::numeric_limits<float>::max())
      {
        fit.contained_scale = static_cast<float>(scale);
        fit.contained_valid = true;
      }
    }
  }
  if (need_inscribed)
  {
    // The output's corners must lie in its own domain (and, when nothing
    // limits the view, keep the horizon margin): the largest normalized
    // offsets of the image are at its corners.
    const float extent = std::max(
      std::abs(0.5f * static_cast<float>(out_size.width - 1) / unit_output.intrinsics.fx),
      std::abs(0.5f * static_cast<float>(out_size.height - 1) / unit_output.intrinsics.fy)
    );
    const float w = static_cast<float>(out_size.width - 1);
    const float h = static_cast<float>(out_size.height - 1);
    const auto corners_within = [&](const float margin) {
      return [&, margin](const float scale) {
        const CameraModel out = scaleOutputModel(unit_output, scale);
        for (const Pixel2 &p : {Pixel2{0, 0}, Pixel2{w, 0}, Pixel2{0, h}, Pixel2{w, h}})
        {
          const RayResult ray = pixelToRay(out, p, solver_options);
          if (!ray.ok() || !representable(out, ray.ray.direction, margin))
          {
            return false;
          }
        }
        return true;
      };
    };
    const double offset =
      smallestUnshownOffset(source, src_size, source_margin_px, output, max_patches);
    const auto domain = extent > 0.0f && std::isfinite(extent) && offset > 0.0
                          ? smallestScaleWith(corners_within(0.0f), extent)
                          : std::nullopt;
    // When nothing limits the view, it stops at the horizon margin.
    const auto horizon = domain && !std::isfinite(offset)
                           ? smallestScaleWith(corners_within(kHorizonMargin), extent)
                           : std::nullopt;
    if (domain && (std::isfinite(offset) || horizon))
    {
      const double scale =
        kInscribedSafety * std::max(
                             static_cast<double>(*domain),
                             std::isfinite(offset) ? 1.0 / offset : static_cast<double>(*horizon)
                           );
      // The scale returned passes the corner tests itself.
      if (std::isfinite(scale) && scale > 0.0 && scale < std::numeric_limits<float>::max() && corners_within(0.0f)(static_cast<float>(scale)) && (std::isfinite(offset) || corners_within(kHorizonMargin)(static_cast<float>(scale))))
      {
        fit.inscribed_scale = static_cast<float>(scale);
        fit.inscribed_valid = true;
      }
    }
  }
  return fit;
}

}  // namespace camxiom::detail
