from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as R

from sc_pose.metrics.error import E_R

from offset_utils.camera_io import parse_rosbag_frame_name
from offset_utils.pose_ops import T_T_C_to_pose

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class PoseStampedRow:
    stamp_ns: int
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float


def _ros_value_to_builtin(value, depth: int = 0):
    """ Convert a ROS message object into plain Python containers """
    if depth > 6:
        return str(type(value).__name__)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_ros_value_to_builtin(item, depth + 1) for item in value]
    if hasattr(value, "__slots__"):
        return {
                    str(slot): _ros_value_to_builtin(getattr(value, slot), depth + 1)
                    for slot in value.__slots__
                    if hasattr(value, slot)
               }
    if hasattr(value, "__dict__"):
        return {
                    str(key): _ros_value_to_builtin(val, depth + 1)
                    for key, val in vars(value).items()
               }
    return str(value)


def _yaml_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def _to_yaml_lines(value, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, val in value.items():
            if isinstance(val, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_to_yaml_lines(val, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(val)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_to_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(value)}"]


def compute_pose_error_metrics(
                                T_T_C_est      : NDArray[np.floating],
                                T_T_C_truth    : NDArray[np.floating],
                             ) -> dict[str, float]:
    """ Compute per-frame translation and rotation errors in the camera frame """
    T_T_C_est       = np.asarray(T_T_C_est, dtype = np.float64)
    T_T_C_truth     = np.asarray(T_T_C_truth, dtype = np.float64)

    q_CAM_2_TARGET_est, r_Co2To_C_est, _       = T_T_C_to_pose(T_T_C_est)
    q_CAM_2_TARGET_truth, r_Co2To_C_truth, _   = T_T_C_to_pose(T_T_C_truth)

    translation_err_vec   = r_Co2To_C_est - r_Co2To_C_truth
    translation_err_mag   = float(np.linalg.norm(translation_err_vec, ord = 2))
    rotation_err_rad      = float(E_R(q_CAM_2_TARGET_truth, q_CAM_2_TARGET_est))
    rotation_err_deg      = float(np.degrees(rotation_err_rad))

    R_C_2_T_est           = T_T_C_est[:3, :3].T
    R_C_2_T_truth         = T_T_C_truth[:3, :3].T
    R_err                 = R_C_2_T_est @ R_C_2_T_truth.T
    euler_err_xyz_deg     = R.from_matrix(R_err).as_euler("xyz", degrees = True)

    return {
                "translation_error_x_m"      : float(translation_err_vec[0]),
                "translation_error_y_m"      : float(translation_err_vec[1]),
                "translation_error_z_m"      : float(translation_err_vec[2]),
                "translation_error_mag_m"    : translation_err_mag,
                "rotation_error_mag_deg"     : rotation_err_deg,
                "rotation_error_x_deg"       : float(euler_err_xyz_deg[0]),
                "rotation_error_y_deg"       : float(euler_err_xyz_deg[1]),
                "rotation_error_z_deg"       : float(euler_err_xyz_deg[2]),
            }


def summarize_frame_metrics(frame_metrics_df: pd.DataFrame) -> dict[str, float | int]:
    """ Aggregate a frame-metrics dataframe into a compact summary """
    summary = {
                "num_frames" : int(len(frame_metrics_df)),
              }
    if len(frame_metrics_df) == 0:
        return summary

    scalar_cols = [
                    "translation_error_x_m",
                    "translation_error_y_m",
                    "translation_error_z_m",
                    "translation_error_mag_m",
                    "rotation_error_mag_deg",
                    "rotation_error_x_deg",
                    "rotation_error_y_deg",
                    "rotation_error_z_deg",
                  ]
    for column in scalar_cols:
        if column not in frame_metrics_df.columns:
            continue
        values = pd.to_numeric(frame_metrics_df[column], errors = "coerce").to_numpy(dtype = float)
        summary[f"{column}_mean"] = float(np.nanmean(values))
        summary[f"{column}_median"] = float(np.nanmedian(values))
        summary[f"{column}_std"] = float(np.nanstd(values))
        summary[f"{column}_max_abs"] = float(np.nanmax(np.abs(values)))
        summary[f"{column}_rmse"] = float(np.sqrt(np.nanmean(values * values)))
    return summary


def write_error_histograms(
                            frame_metrics_df  : pd.DataFrame,
                            output_dir        : Path,
                            num_bins          : int = 30,
                          ) -> dict[str, dict[str, object]]:
    """ Write histogram PNGs and histogram-bin JSON for pose-error metrics """
    output_dir               = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)

    histogram_specs = {
                        "translation_error_x_m"   : ("Translation Error X", "Translation Error X [m]"),
                        "translation_error_y_m"   : ("Translation Error Y", "Translation Error Y [m]"),
                        "translation_error_z_m"   : ("Translation Error Z", "Translation Error Z [m]"),
                        "translation_error_mag_m" : ("Translation Error Magnitude", "Translation Error Magnitude [m]"),
                        "rotation_error_x_deg"    : ("Rotation Error X", "Rotation Error X [deg]"),
                        "rotation_error_y_deg"    : ("Rotation Error Y", "Rotation Error Y [deg]"),
                        "rotation_error_z_deg"    : ("Rotation Error Z", "Rotation Error Z [deg]"),
                        "rotation_error_mag_deg"  : ("Rotation Error Magnitude", "Rotation Error Magnitude [deg]"),
                      }

    histogram_manifest      = {}
    for column_name, (plot_title, x_label) in histogram_specs.items():
        if column_name not in frame_metrics_df.columns:
            continue

        values = pd.to_numeric(frame_metrics_df[column_name], errors = "coerce").to_numpy(dtype = float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        if values.size == 1 or np.allclose(values, values[0]):
            center              = float(values[0])
            pad                 = max(abs(center) * 0.05, 1e-9)
            bin_edges           = np.linspace(center - pad, center + pad, num_bins + 1)
            counts, bin_edges   = np.histogram(values, bins = bin_edges)
        else:
            counts, bin_edges   = np.histogram(values, bins = num_bins)

        mean_value           = float(np.mean(values))
        median_value         = float(np.median(values))
        std_value            = float(np.std(values))

        fig, ax = plt.subplots(figsize = (8, 5), dpi = 140)
        ax.hist(values, bins = bin_edges, color = "#2a6f97", edgecolor = "black", alpha = 0.85)
        ax.axvline(mean_value, color = "#d62828", linestyle = "--", linewidth = 2, label = "mean")
        ax.axvline(median_value, color = "#f77f00", linestyle = "-.", linewidth = 2, label = "median")
        if std_value > 0.0:
            ax.axvline(mean_value - std_value, color = "#6a4c93", linestyle = ":", linewidth = 1.75, label = "mean - std")
            ax.axvline(mean_value + std_value, color = "#6a4c93", linestyle = ":", linewidth = 1.75, label = "mean + std")
            ax.axvspan(mean_value - std_value, mean_value + std_value, color = "#6a4c93", alpha = 0.08)
        ax.set_title(plot_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Count")
        ax.grid(True, alpha = 0.25)
        ax.legend(loc = "best")
        stat_text = f"mean = {mean_value:.6g}\nstd = {std_value:.6g}"
        ax.text(
                    0.98,
                    0.97,
                    stat_text,
                    transform = ax.transAxes,
                    ha = "right",
                    va = "top",
                    bbox = {
                                "boxstyle"  : "round",
                                "facecolor" : "white",
                                "alpha"     : 0.85,
                                "edgecolor" : "#666666",
                           },
               )
        fig.tight_layout()

        hist_path = output_dir / f"{column_name}_hist.png"
        fig.savefig(hist_path)
        plt.close(fig)

        histogram_manifest[column_name] = {
                                            "plot_title"         : plot_title,
                                            "path"               : str(hist_path),
                                            "num_samples"        : int(values.size),
                                            "min"                : float(np.min(values)),
                                            "max"                : float(np.max(values)),
                                            "mean"               : mean_value,
                                            "median"             : median_value,
                                            "std"                : std_value,
                                            "bin_edges"          : bin_edges.astype(float).tolist(),
                                            "counts"             : counts.astype(int).tolist(),
                                         }

    manifest_path = output_dir / "histogram_manifest.json"
    with manifest_path.open("w", encoding = "utf-8") as handle:
        json.dump(histogram_manifest, handle, indent = 4)

    return histogram_manifest


def load_pose_rows_from_rosbag(
                                bag_dir      : Path,
                                topic_name   : str,
                             ) -> list[PoseStampedRow]:
    """ Read PoseStamped rows from a rosbag topic """
    rows = []
    with AnyReader([Path(bag_dir)]) as reader:
        connections = [conn for conn in reader.connections if conn.topic == topic_name]
        if len(connections) == 0:
            raise RuntimeError(f"Topic not found in rosbag: {topic_name}")

        for connection, _, rawdata in reader.messages(connections = connections):
            msg         = reader.deserialize(rawdata, connection.msgtype)
            stamp_ns    = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            rows.append(
                        PoseStampedRow(
                                        stamp_ns = stamp_ns,
                                        x = float(msg.pose.position.x),
                                        y = float(msg.pose.position.y),
                                        z = float(msg.pose.position.z),
                                        qw = float(msg.pose.orientation.w),
                                        qx = float(msg.pose.orientation.x),
                                        qy = float(msg.pose.orientation.y),
                                        qz = float(msg.pose.orientation.z),
                                     )
                       )
    rows.sort(key = lambda row: row.stamp_ns)
    if len(rows) == 0:
        raise RuntimeError(f"No messages found for topic: {topic_name}")
    return rows


def write_topic_yamls(
                        bag_dir        : Path,
                        topic_names    : list[str],
                        output_dir     : Path,
                     ) -> dict[str, object]:
    """ Write YAML dumps for the rosbag topics this workflow consumes """
    bag_dir      = Path(bag_dir)
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)

    topic_entries = []
    with AnyReader([bag_dir]) as reader:
        for topic_name in topic_names:
            connections = [conn for conn in reader.connections if conn.topic == topic_name]
            if len(connections) == 0:
                raise RuntimeError(f"Topic not found in rosbag: {topic_name}")

            msg_count                = 0
            first_bag_time_ns        = None
            last_bag_time_ns         = None
            first_header_stamp_ns    = None
            last_header_stamp_ns     = None
            example_message          = None
            all_messages             = []

            for connection, timestamp, rawdata in reader.messages(connections = connections):
                bag_time_ns = int(timestamp)
                if first_bag_time_ns is None:
                    first_bag_time_ns = bag_time_ns
                last_bag_time_ns = bag_time_ns

                msg = reader.deserialize(rawdata, connection.msgtype)
                msg_builtin = _ros_value_to_builtin(msg)
                if example_message is None:
                    example_message = msg_builtin

                if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
                    header_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
                    if first_header_stamp_ns is None:
                        first_header_stamp_ns = header_stamp_ns
                    last_header_stamp_ns = header_stamp_ns
                else:
                    header_stamp_ns = None

                all_messages.append(
                                    {
                                        "bag_time_ns"       : int(bag_time_ns),
                                        "bag_time_s"        : float(bag_time_ns) / 1e9,
                                        "header_stamp_ns"   : int(header_stamp_ns) if header_stamp_ns is not None else None,
                                        "header_stamp_s"    : float(header_stamp_ns) / 1e9 if header_stamp_ns is not None else None,
                                        "message"           : msg_builtin,
                                    }
                                   )

                msg_count += 1

            topic_summary = {
                                "topic_name"               : str(topic_name),
                                "rosbag_path"              : str(bag_dir),
                                "message_type"             : str(connections[0].msgtype),
                                "connection_count"         : int(len(connections)),
                                "message_count"            : int(msg_count),
                                "first_bag_time_ns"        : int(first_bag_time_ns) if first_bag_time_ns is not None else None,
                                "first_bag_time_s"         : float(first_bag_time_ns) / 1e9 if first_bag_time_ns is not None else None,
                                "last_bag_time_ns"         : int(last_bag_time_ns) if last_bag_time_ns is not None else None,
                                "last_bag_time_s"          : float(last_bag_time_ns) / 1e9 if last_bag_time_ns is not None else None,
                                "first_header_stamp_ns"    : int(first_header_stamp_ns) if first_header_stamp_ns is not None else None,
                                "first_header_stamp_s"     : float(first_header_stamp_ns) / 1e9 if first_header_stamp_ns is not None else None,
                                "last_header_stamp_ns"     : int(last_header_stamp_ns) if last_header_stamp_ns is not None else None,
                                "last_header_stamp_s"      : float(last_header_stamp_ns) / 1e9 if last_header_stamp_ns is not None else None,
                                "example_message"          : example_message,
                                "messages"                 : all_messages,
                            }

            topic_filename = topic_name.strip("/").replace("/", "__") + ".yaml"
            topic_yaml_path = output_dir / topic_filename
            topic_yaml_path.write_text("\n".join(_to_yaml_lines(topic_summary)) + "\n", encoding = "utf-8")

            topic_entries.append(
                                    {
                                        "topic_name"      : str(topic_name),
                                        "message_type"    : str(connections[0].msgtype),
                                        "message_count"   : int(msg_count),
                                        "yaml_path"       : str(topic_yaml_path),
                                    }
                                )

    manifest = {
                    "rosbag_path"    : str(bag_dir),
                    "num_topics"     : int(len(topic_entries)),
                    "topics"         : topic_entries,
               }
    manifest_path = output_dir / "manifest.yaml"
    manifest_path.write_text("\n".join(_to_yaml_lines(manifest)) + "\n", encoding = "utf-8")
    return {
                "output_dir"      : str(output_dir),
                "manifest_path"   : str(manifest_path),
                "topics"          : topic_entries,
           }


def find_nearest_pose_row(
                            image_stamp_ns  : int,
                            pose_rows       : list[PoseStampedRow],
                         ) -> tuple[PoseStampedRow, int]:
    """ Find the nearest rosbag pose row to an image timestamp """
    stamp_array = np.array([row.stamp_ns for row in pose_rows], dtype = np.int64)
    idx         = int(np.searchsorted(stamp_array, image_stamp_ns))
    candidate_indices = []
    if idx < len(pose_rows):
        candidate_indices.append(idx)
    if idx > 0:
        candidate_indices.append(idx - 1)

    best_idx    = min(candidate_indices, key = lambda ii: abs(int(stamp_array[ii]) - int(image_stamp_ns)))
    best_row    = pose_rows[best_idx]
    delta_ns    = int(best_row.stamp_ns) - int(image_stamp_ns)
    return best_row, delta_ns


def build_vicon_dataframe_from_rosbag(
                                        image_paths      : list[Path],
                                        bag_dir          : Path,
                                        cam_topic        : str,
                                        target_topic     : str,
                                     ) -> pd.DataFrame:
    """ Build a Vicon-style dataframe by matching rosbag truth to extracted images """
    cam_rows    = load_pose_rows_from_rosbag(bag_dir, cam_topic)
    soho_rows   = load_pose_rows_from_rosbag(bag_dir, target_topic)

    vicon_rows  = []
    for image_path in image_paths:
        frame_idx, image_stamp_ns = parse_rosbag_frame_name(image_path)
        cam_row, cam_delta_ns     = find_nearest_pose_row(image_stamp_ns, cam_rows)
        soho_row, soho_delta_ns   = find_nearest_pose_row(image_stamp_ns, soho_rows)
        image_timestamp_s         = float(image_stamp_ns) / 1e9
        cam_timestamp_s           = float(cam_row.stamp_ns) / 1e9
        soho_timestamp_s          = float(soho_row.stamp_ns) / 1e9

        vicon_rows.append(
                            {
                                "frame"                 : image_path.name,
                                "frame_index"           : int(frame_idx),
                                "image_number"          : int(frame_idx),
                                "image_timestamp_ns"    : int(image_stamp_ns),
                                "image_timestamp_s"     : image_timestamp_s,
                                "cam_timestamp_ns"      : int(cam_row.stamp_ns),
                                "cam_timestamp_s"       : cam_timestamp_s,
                                "soho_timestamp_ns"     : int(soho_row.stamp_ns),
                                "soho_timestamp_s"      : soho_timestamp_s,
                                "cam_delta_ms"          : float(cam_delta_ns / 1e6),
                                "soho_delta_ms"         : float(soho_delta_ns / 1e6),
                                "cam_x"                 : float(cam_row.x * 1e3),
                                "cam_y"                 : float(cam_row.y * 1e3),
                                "cam_z"                 : float(cam_row.z * 1e3),
                                "cam_qw"                : float(cam_row.qw),
                                "cam_qx"                : float(cam_row.qx),
                                "cam_qy"                : float(cam_row.qy),
                                "cam_qz"                : float(cam_row.qz),
                                "soho_x"                : float(soho_row.x * 1e3),
                                "soho_y"                : float(soho_row.y * 1e3),
                                "soho_z"                : float(soho_row.z * 1e3),
                                "soho_qw"               : float(soho_row.qw),
                                "soho_qx"               : float(soho_row.qx),
                                "soho_qy"               : float(soho_row.qy),
                                "soho_qz"               : float(soho_row.qz),
                            }
                         )

    vicon_df = pd.DataFrame(vicon_rows)
    return vicon_df.sort_values("frame_index").reset_index(drop = True)


def write_trajectory_pack(
                            output_dir           : Path,
                            frame_records        : list[dict[str, object]],
                            camera_settings      : dict[str, float],
                            K                    : NDArray[np.floating],
                            image_width_px       : int,
                            image_height_px      : int,
                            pose_key             : str = "pose",
                         ) -> tuple[Path, Path]:
    """ Write a nav_ros-style trajectory pack from offset-adjusted truth """
    output_dir       = Path(output_dir)
    traj_dir         = output_dir / "trajectory_pack"
    if traj_dir.exists():
        shutil.rmtree(traj_dir)
    traj_dir.mkdir(parents = True, exist_ok = True)

    lifted_records   = []
    for idx, record in enumerate(frame_records):
        token = f"{idx:05d}"
        src_image_path    = Path(record["image_path"])
        out_img_path      = traj_dir / f"image_{token}.png"
        out_meta_path     = traj_dir / f"meta_{token}.json"
        shutil.copy2(src_image_path, out_img_path)

        q_CAM_2_TARGET    = np.asarray(record["q_CAM_2_TARGET"], dtype = float).reshape(4,)
        q_TARGET_2_CAM    = np.asarray(record["q_TARGET_2_CAM"], dtype = float).reshape(4,)
        r_Co2To_C         = np.asarray(record["r_Co2To_C"], dtype = float).reshape(3,)
        T_T_C             = np.asarray(record["T_T_C"], dtype = float).reshape(4, 4)
        cam_delta_s       = float(record["cam_delta_ms"]) / 1000.0
        soho_delta_s      = float(record["soho_delta_ms"]) / 1000.0

        out_meta = {
                    str(pose_key)            : np.concatenate([r_Co2To_C, q_CAM_2_TARGET]).tolist(),
                    "pose_label"             : "translation_in_camera_frame + q_CAM_2_TARGET",
                    "pose_convention"        : {
                                                    "pose"               : "translation_in_camera_frame + q_CAM_2_TARGET",
                                                    "T_T_C"              : "^C T_T homogeneous transform",
                                               },
                    "camera_settings"        : dict(camera_settings),
                    "camera_K"               : np.asarray(K, dtype = float).tolist(),
                    "T_T_C"                  : T_T_C.tolist(),
                    "output_image_size_wh"   : [int(image_width_px), int(image_height_px)],
                    "source_frame_name"      : str(record["frame"]),
                    "source_frame_index"     : int(record["frame_index"]),
                    "image_timestamp_ns"     : int(record["image_timestamp_ns"]),
                    "image_timestamp_s"      : float(record["image_timestamp_ns"]) / 1e9,
                    "cam_timestamp_ns"       : int(record["cam_timestamp_ns"]),
                    "cam_timestamp_s"        : float(record["cam_timestamp_ns"]) / 1e9,
                    "soho_timestamp_ns"      : int(record["soho_timestamp_ns"]),
                    "soho_timestamp_s"       : float(record["soho_timestamp_ns"]) / 1e9,
                    "cam_delta_s"            : cam_delta_s,
                    "soho_delta_s"           : soho_delta_s,
                   }
        with out_meta_path.open("w", encoding = "utf-8") as handle:
            json.dump(out_meta, handle, indent = 4)

        lifted_records.append(
                                {
                                    "token"              : token,
                                    "image_path"         : str(out_img_path),
                                    "meta_path"          : str(out_meta_path),
                                    "source_frame_name"  : str(record["frame"]),
                                    "source_frame_index" : int(record["frame_index"]),
                                    "image_timestamp_ns" : int(record["image_timestamp_ns"]),
                                }
                             )

    lifted_path = output_dir / "trajectory_lifted.json"
    with lifted_path.open("w", encoding = "utf-8") as handle:
        json.dump(
                    {
                        "num_records"        : int(len(lifted_records)),
                        "trajectory_pack_dir": str(traj_dir),
                        "records"            : lifted_records,
                    },
                    handle,
                    indent = 4,
                 )
    return traj_dir, lifted_path
