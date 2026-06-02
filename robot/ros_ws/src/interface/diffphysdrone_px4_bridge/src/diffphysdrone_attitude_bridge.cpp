#include <array>
#include <cmath>
#include <memory>
#include <string>

#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <mav_msgs/msg/attitude_thrust.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

namespace {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;
using Quat = std::array<double, 4>;  // w, x, y, z

double dot(const Vec3 &a, const Vec3 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return {
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  };
}

double norm(const Vec3 &v) {
  return std::sqrt(dot(v, v));
}

Vec3 normalize(const Vec3 &v, const Vec3 &fallback) {
  const double n = norm(v);
  if (n < 1.0e-9 || !std::isfinite(n)) {
    return fallback;
  }
  return {v[0] / n, v[1] / n, v[2] / n};
}

double clamp(double value, double low, double high) {
  return std::max(low, std::min(high, value));
}

Vec3 clamp_norm(const Vec3 &v, double max_norm) {
  const double n = norm(v);
  if (max_norm <= 0.0 || n <= max_norm || n < 1.0e-9) {
    return v;
  }
  const double scale = max_norm / n;
  return {v[0] * scale, v[1] * scale, v[2] * scale};
}

Vec3 clamp_specific_force_tilt(const Vec3 &force, double max_tilt_rad) {
  if (max_tilt_rad <= 0.0) {
    return {0.0, 0.0, std::max(1.0e-6, force[2])};
  }
  if (max_tilt_rad >= M_PI * 0.5) {
    return force;
  }

  double fx = force[0];
  double fy = force[1];
  double fz = std::max(force[2], 1.0e-6);
  const double horizontal = std::hypot(fx, fy);
  const double max_horizontal = fz * std::tan(max_tilt_rad);
  if (horizontal > max_horizontal && horizontal > 1.0e-9) {
    const double scale = max_horizontal / horizontal;
    fx *= scale;
    fy *= scale;
  }
  return {fx, fy, fz};
}

Quat matrix_to_quaternion(const Mat3 &r) {
  double w = 1.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  const double trace = r[0][0] + r[1][1] + r[2][2];

  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    w = 0.25 * s;
    x = (r[2][1] - r[1][2]) / s;
    y = (r[0][2] - r[2][0]) / s;
    z = (r[1][0] - r[0][1]) / s;
  } else if (r[0][0] > r[1][1] && r[0][0] > r[2][2]) {
    const double s = std::sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0;
    w = (r[2][1] - r[1][2]) / s;
    x = 0.25 * s;
    y = (r[0][1] + r[1][0]) / s;
    z = (r[0][2] + r[2][0]) / s;
  } else if (r[1][1] > r[2][2]) {
    const double s = std::sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0;
    w = (r[0][2] - r[2][0]) / s;
    x = (r[0][1] + r[1][0]) / s;
    y = 0.25 * s;
    z = (r[1][2] + r[2][1]) / s;
  } else {
    const double s = std::sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0;
    w = (r[1][0] - r[0][1]) / s;
    x = (r[0][2] + r[2][0]) / s;
    y = (r[1][2] + r[2][1]) / s;
    z = 0.25 * s;
  }

  const double n = std::sqrt(w * w + x * x + y * y + z * z);
  if (n < 1.0e-9 || !std::isfinite(n)) {
    return {1.0, 0.0, 0.0, 0.0};
  }
  return {w / n, x / n, y / n, z / n};
}

double yaw_from_quaternion(double w, double x, double y, double z) {
  return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

}  // namespace

class DiffPhysDroneAttitudeBridge : public rclcpp::Node {
 public:
  DiffPhysDroneAttitudeBridge() : Node("diffphysdrone_attitude_bridge") {
    accel_topic_ = declare_parameter<std::string>("accel_topic", "diffphysdrone/accel_cmd");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "odometry");
    output_topic_ = declare_parameter<std::string>("output_topic", "cmd_attitude_thrust");
    hover_thrust_ = declare_parameter<double>("hover_thrust", 0.5);
    thrust_min_ = declare_parameter<double>("thrust_min", 0.05);
    thrust_max_ = declare_parameter<double>("thrust_max", 0.9);
    max_tilt_rad_ = declare_parameter<double>("max_tilt_deg", 35.0) * M_PI / 180.0;
    max_net_accel_ = declare_parameter<double>("max_net_accel_mps2", 20.0);
    gravity_ = declare_parameter<double>("gravity_mps2", 9.80665);
    use_current_yaw_ = declare_parameter<bool>("use_current_yaw", true);
    yaw_ref_ = declare_parameter<double>("yaw_ref_rad", 0.0);
    timeout_s_ = declare_parameter<double>("command_timeout_s", 0.2);
    publish_stale_hover_ = declare_parameter<bool>("publish_stale_hover", false);

    pub_ = create_publisher<mav_msgs::msg::AttitudeThrust>(output_topic_, 1);
    accel_sub_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
      accel_topic_, 1,
      [this](const geometry_msgs::msg::Vector3Stamped::SharedPtr msg) {
        last_accel_ = {msg->vector.x, msg->vector.y, msg->vector.z};
        last_command_time_ = now();
        have_command_ = true;
      });

    auto odom_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, odom_qos,
      [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
        const auto &q = msg->pose.pose.orientation;
        current_yaw_ = yaw_from_quaternion(q.w, q.x, q.y, q.z);
      });

    const double publish_rate = std::max(1.0, declare_parameter<double>("publish_rate_hz", 50.0));
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / publish_rate),
      [this]() { publish_latest(); });

    RCLCPP_INFO(get_logger(), "DiffPhysDrone C++ attitude bridge publishing to %s", output_topic_.c_str());
  }

 private:
  bool command_is_fresh() const {
    if (!have_command_) {
      return false;
    }
    return (now() - last_command_time_).seconds() <= timeout_s_;
  }

  void publish_latest() {
    if (!have_command_) {
      return;
    }

    Vec3 accel = last_accel_;
    if (!command_is_fresh()) {
      if (!publish_stale_hover_) {
        return;
      }
      accel = {0.0, 0.0, 0.0};
    }

    if (!std::isfinite(accel[0]) || !std::isfinite(accel[1]) || !std::isfinite(accel[2])) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Skipping non-finite acceleration command");
      return;
    }

    accel = clamp_norm(accel, max_net_accel_);
    Vec3 specific_force = {accel[0], accel[1], accel[2] + gravity_};
    if (norm(specific_force) < 1.0e-9) {
      specific_force = {0.0, 0.0, gravity_};
    }
    specific_force = clamp_specific_force_tilt(specific_force, max_tilt_rad_);

    const double thrust = clamp(norm(specific_force) / gravity_ * hover_thrust_, thrust_min_, thrust_max_);
    const Vec3 z_body = normalize(specific_force, {0.0, 0.0, 1.0});

    const double yaw_ref = use_current_yaw_ ? current_yaw_ : yaw_ref_;
    const Vec3 x_heading = {std::cos(yaw_ref), std::sin(yaw_ref), 0.0};
    Vec3 y_body = cross(z_body, x_heading);
    if (norm(y_body) < 1.0e-6) {
      y_body = cross(z_body, {0.0, 1.0, 0.0});
    }
    y_body = normalize(y_body, {0.0, 1.0, 0.0});
    const Vec3 x_body = normalize(cross(y_body, z_body), {1.0, 0.0, 0.0});

    const Mat3 r = {{
      {{x_body[0], y_body[0], z_body[0]}},
      {{x_body[1], y_body[1], z_body[1]}},
      {{x_body[2], y_body[2], z_body[2]}},
    }};
    const Quat q = matrix_to_quaternion(r);

    mav_msgs::msg::AttitudeThrust msg;
    msg.header.stamp = now();
    msg.header.frame_id = "map";
    msg.attitude.w = q[0];
    msg.attitude.x = q[1];
    msg.attitude.y = q[2];
    msg.attitude.z = q[3];
    msg.thrust.x = 0.0;
    msg.thrust.y = 0.0;
    msg.thrust.z = thrust;
    pub_->publish(msg);
  }

  std::string accel_topic_;
  std::string odom_topic_;
  std::string output_topic_;
  double hover_thrust_{0.5};
  double thrust_min_{0.05};
  double thrust_max_{0.9};
  double max_tilt_rad_{35.0 * M_PI / 180.0};
  double max_net_accel_{20.0};
  double gravity_{9.80665};
  bool use_current_yaw_{true};
  double yaw_ref_{0.0};
  double timeout_s_{0.2};
  bool publish_stale_hover_{false};
  bool have_command_{false};
  double current_yaw_{0.0};
  Vec3 last_accel_{0.0, 0.0, 0.0};
  rclcpp::Time last_command_time_{0, 0, RCL_ROS_TIME};

  rclcpp::Publisher<mav_msgs::msg::AttitudeThrust>::SharedPtr pub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr accel_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DiffPhysDroneAttitudeBridge>());
  rclcpp::shutdown();
  return 0;
}
