from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

############################# How This Script Works #############################
# 1) read an input lifted trajectory pack, which already contains a pose per frame
# 2) read the shifted-truth pose from the metadata pose key for each frame
# 3) use that pose directly as the external MegaPose initial estimate so
#    refinement starts from the Vicon / truth-shifted pose instead of bbox coarse
# 4) run MegaPose with the requested number of hypotheses and refiner iterations
# 5) keep the final top-1 refined pose for each frame, or fall back to the input
#    pose if refinement fails
# 6) write a new trajectory pack plus per-frame MegaPose provenance metadata
# 7) optionally write sanity overlays for the refined pack so the projected pose
#    can be inspected frame-by-frame
############################# How This Script Works #############################

HERE                    = Path(__file__).resolve()
PARENT_ROOT             = HERE.parent
CAMCAL_ROOT             = PARENT_ROOT.parent
AGNC_ROOT               = CAMCAL_ROOT.parent
MEGAPOSE_SRC            = AGNC_ROOT / "megapose6d_dev" / "src"
SC_POSE_UTILS_SRC       = AGNC_ROOT / "sc-pose-utils" / "src"

for dependency_src in (MEGAPOSE_SRC, SC_POSE_UTILS_SRC):
    dependency_src_str = str(dependency_src)
    if dependency_src.exists() and dependency_src_str not in sys.path:
        sys.path.insert(0, dependency_src_str)

from PIL import Image
import torch
from megapose.inference.types import ObservationTensor
from megapose.service.server import PoseService, _parse_pose_estimates
try:
    from megapose.utils.logging import set_logging_level as _set_megapose_logging_level
except Exception:
    _set_megapose_logging_level = None
from sc_pose import q2rotm

try:
    from megapose.service.inferface import PoseServiceInterface
except Exception:
    PoseServiceInterface = None

from offset_utils.metrics_and_pack import write_json_payload, write_trajectory_pack
from offset_utils.pose_ops import T_T_C_to_pose, build_transform
from offset_utils.reprojection import load_target_points, write_sanity_overlays


DEFAULT_LIFTED_JSON = (
    CAMCAL_ROOT
    / "results"
    / "run_004"
    / "check_offset_rosbag_v2"
    / "trajectory_export"
    / "trajectory_lifted_truth_shifted.json"
)
DEFAULT_MESHES_DIR = AGNC_ROOT / "nav_ros" / "testing" / "pose_model_artifacts" / "meshes"
DEFAULT_MESH_LABEL = "soho_centered"
DEFAULT_MODEL = "megapose-1.0-RGB-multi-hypothesis"
DEFAULT_OUTPUT_PACK_DIR_NAME = "trajectory_pack_truth_shifted_megapose_h5_r8"
DEFAULT_OUTPUT_LIFTED_FILENAME = "trajectory_lifted_truth_shifted_megapose_h5_r8.json"
DEFAULT_SANITY_CALIBRATION_YAML = CAMCAL_ROOT / "data" / "offset" / "collection_001" / "calibration.yaml"
DEFAULT_SANITY_KPS_FILE         = PARENT_ROOT / "mesh_keypoints" / "soho_centered" / "rendered_keypoints.json"
DEFAULT_SANITY_KPS_UNITS        = "m"

MEGAPOSE_INTERNAL_BATCH_SIZE = 128
REEXEC_ENVVAR = "CAMCAL_MEGAPOSE_REEXEC_UNDER_PYTHON_C"


def _load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MegaPose refinement on a CamCal trajectory pack and write a new trajectory pack."
    )
    parser.add_argument("--lifted-json", type=Path, default=DEFAULT_LIFTED_JSON)
    parser.add_argument("--meshes-dir", type=Path, default=DEFAULT_MESHES_DIR)
    parser.add_argument("--mesh-label", type=str, default=DEFAULT_MESH_LABEL)
    parser.add_argument("--mesh-units", type=str, default="mm")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--n-pose-hypotheses", type=int, default=5)
    parser.add_argument("--n-refiner-iterations", type=int, default=8)
    parser.add_argument("--output-pack-dir-name", type=str, default=DEFAULT_OUTPUT_PACK_DIR_NAME)
    parser.add_argument("--output-lifted-filename", type=str, default=DEFAULT_OUTPUT_LIFTED_FILENAME)
    parser.add_argument("--sanity-calibration-yaml", type=Path, default=DEFAULT_SANITY_CALIBRATION_YAML)
    parser.add_argument("--sanity-kps-file", type=Path, default=DEFAULT_SANITY_KPS_FILE)
    parser.add_argument("--sanity-kps-units", type=str, default=DEFAULT_SANITY_KPS_UNITS)
    parser.add_argument(
        "--skip-sanity-overlays",
        action="store_true",
        help="Skip writing sanity_check overlay images into the output trajectory pack.",
    )
    parser.add_argument(
        "--record-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional zero-based lifted-record indices to process. Defaults to all records.",
    )
    return parser.parse_args()


def _maybe_reexec_under_python_c() -> None:
    if os.environ.get(REEXEC_ENVVAR) == "1":
        return

    env = os.environ.copy()
    env[REEXEC_ENVVAR] = "1"
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(PARENT_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    os.execvpe(
        sys.executable,
        [
            sys.executable,
            "-u",
            "-c",
            "from run_megapose_on_trajectory_pack_VC import main; main()",
            *sys.argv[1:],
        ],
        env,
    )


def _find_mesh_path(meshes_dir: Path, mesh_label: str) -> Path:
    mesh_dir = Path(meshes_dir) / str(mesh_label)
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Mesh label directory not found: {mesh_dir}")
    candidates = sorted(path for path in mesh_dir.iterdir() if path.suffix.lower() in {".obj", ".ply"})
    if len(candidates) == 0:
        raise FileNotFoundError(f"No .obj/.ply mesh found in {mesh_dir}")
    return candidates[0]


def _load_dist_coeffs_from_calibration_yaml(path: Path) -> np.ndarray:
    values: dict[str, float] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"k1", "k2", "k3", "p1", "p2"}:
            values[key] = float(value.strip())

    missing = [key for key in ("k1", "k2", "k3", "p1", "p2") if key not in values]
    if missing:
        raise ValueError(f"Missing distortion coefficient(s) in {path}: {missing}")

    return np.array(
        [
            values["k1"],
            values["k2"],
            values["p1"],
            values["p2"],
            values["k3"],
        ],
        dtype=np.float64,
    )


def write_sanity_overlays_from_lifted(
    lifted_json_path: Path,
    calibration_yaml_path: Path = DEFAULT_SANITY_CALIBRATION_YAML,
    keypoints_path: Path = DEFAULT_SANITY_KPS_FILE,
    keypoints_units: str = DEFAULT_SANITY_KPS_UNITS,
) -> Path:
    lifted_payload = _load_json(lifted_json_path)
    records = lifted_payload.get("records", [])
    if len(records) == 0:
        raise ValueError(f"No records found in lifted JSON: {lifted_json_path}")

    frame_records: list[dict[str, Any]] = []
    bboxes_xyxy: list[list[int] | None] = []
    traj_dir: Path | None = None
    K: np.ndarray | None = None

    for record in records:
        meta = _load_json(Path(record["meta_path"]))
        if traj_dir is None:
            traj_dir = Path(record["meta_path"]).resolve().parent
        if K is None:
            K = np.asarray(meta["camera_K"], dtype=np.float64)
        frame_records.append(
            {
                "T_T_C": meta["T_T_C"],
            }
        )
        bbox = meta.get("megapose_input_bbox_xyxy", [])
        bboxes_xyxy.append(list(bbox) if bbox else None)

    assert traj_dir is not None
    assert K is not None
    dist_coeffs = _load_dist_coeffs_from_calibration_yaml(Path(calibration_yaml_path))
    target_points_with_origin_m = load_target_points(
                                                        kps_file = Path(keypoints_path),
                                                        with_origin = True,
                                                        units = keypoints_units,
                                                    )
    return write_sanity_overlays(
                traj_dir = traj_dir,
                frame_records = frame_records,
                K = K,
                dist_coeffs = dist_coeffs,
                target_pts_with_origin = target_points_with_origin_m,
                bboxes_xyxy = bboxes_xyxy,
           )
def _make_observation(image_path: Path, K: np.ndarray) -> ObservationTensor:
    rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8, copy=True)
    return ObservationTensor.from_numpy(rgb, depth=None, K=np.asarray(K, dtype=np.float32)).cuda()


def _make_coarse_pose_estimates(
    T_T_C: np.ndarray,
    mesh_label: str,
    n_pose_hypotheses: int,
):
    """ seed MegaPose refinement from the pose stored in the shifted-truth pack """
    pose_matrix = np.asarray(T_T_C, dtype = np.float32).reshape(4, 4)
    num_hypotheses = max(int(n_pose_hypotheses), 1)
    payload = {
                "pose_estimates": {
                                    "infos": [
                                                {
                                                    "label": str(mesh_label),
                                                    "batch_im_id": 0,
                                                    "instance_id": 0,
                                                    "hypothesis_id": int(hypothesis_id),
                                                }
                                                for hypothesis_id in range(num_hypotheses)
                                             ],
                                    "poses": [pose_matrix.tolist() for _ in range(num_hypotheses)],
                                  }
              }
    return _parse_pose_estimates(payload).cuda()
def _pose_key_to_T_T_C(pose: np.ndarray) -> np.ndarray:
    pose                = np.asarray(pose, dtype = np.float64).reshape(7,)
    r_Co2To_C           = pose[:3]
    q_CAM_2_TARGET      = pose[3:]
    Trfm_T_2_C          = q2rotm(q_CAM_2_TARGET)
    return build_transform(Trfm_T_2_C, r_Co2To_C)


def _create_megapose_service(
    model_name: str,
    meshes_dir: Path,
    mesh_units: str,
) -> tuple[PoseService, str]:
    init_errors: list[str] = []
    if PoseServiceInterface is not None:
        try:
            interface = PoseServiceInterface(
                model_name=model_name,
                meshes_dir=Path(meshes_dir),
                mesh_units=mesh_units,
            )
            return interface.service, "PoseServiceInterface"
        except Exception as exc:
            init_errors.append(f"PoseServiceInterface failed: {type(exc).__name__}: {exc}")

    try:
        service = PoseService(
            model_name=model_name,
            meshes_dir=Path(meshes_dir),
            mesh_units=mesh_units,
            bsz_images=MEGAPOSE_INTERNAL_BATCH_SIZE,
        )
        return service, "PoseService"
    except Exception as exc:
        init_errors.append(f"PoseService failed: {type(exc).__name__}: {exc}")
        raise RuntimeError("Unable to initialize MegaPose locally. " + " | ".join(init_errors)) from exc


def _run_megapose_single_frame(
    service: PoseService,
    image_path: Path,
    K: np.ndarray,
    initial_T_T_C: np.ndarray,
    mesh_label: str,
    n_pose_hypotheses: int,
    n_refiner_iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    observation = _make_observation(image_path=image_path, K=K)
    coarse_estimates = _make_coarse_pose_estimates(
                                                    T_T_C = initial_T_T_C,
                                                    mesh_label = mesh_label,
                                                    n_pose_hypotheses = n_pose_hypotheses,
                                                )

    final_pose_estimates, extra_data = service.pose_estimator.run_inference_pipeline(
        observation=observation,
        run_detector=False,
        coarse_estimates=coarse_estimates,
        n_refiner_iterations=int(n_refiner_iterations),
        n_pose_hypotheses=int(n_pose_hypotheses),
        keep_all_refiner_outputs=False,
    )
    if len(final_pose_estimates.infos) != 1:
        raise RuntimeError(
            f"Expected exactly one final pose estimate, received {len(final_pose_estimates.infos)}."
        )

    pose_matrix = final_pose_estimates.poses.detach().cpu().numpy()[0].astype(np.float64)
    info_row = final_pose_estimates.infos.iloc[0].to_dict()
    timing_summary = {
        "time": extra_data.get("time"),
        "timing_str": extra_data.get("timing_str"),
    }
    return pose_matrix, {
        "info_row": _to_builtin(info_row),
        "timing": _to_builtin(timing_summary),
    }


def _build_frame_record(input_record: dict[str, Any], input_meta: dict[str, Any], T_T_C: np.ndarray) -> dict[str, Any]:
    q_CAM_2_TARGET, r_Co2To_C, q_TARGET_2_CAM = T_T_C_to_pose(T_T_C)
    return {
        "image_path": str(input_record["image_path"]),
        "frame": str(input_meta["source_frame_name"]),
        "frame_index": int(input_meta["source_frame_index"]),
        "image_timestamp_ns": int(input_meta["image_timestamp_ns"]),
        "cam_timestamp_ns": int(input_meta["cam_timestamp_ns"]),
        "soho_timestamp_ns": int(input_meta["soho_timestamp_ns"]),
        "cam_delta_ms": float(input_meta["cam_delta_s"]) * 1000.0,
        "soho_delta_ms": float(input_meta["soho_delta_s"]) * 1000.0,
        "q_CAM_2_TARGET": np.asarray(q_CAM_2_TARGET, dtype=float).tolist(),
        "q_TARGET_2_CAM": np.asarray(q_TARGET_2_CAM, dtype=float).tolist(),
        "r_Co2To_C": np.asarray(r_Co2To_C, dtype=float).tolist(),
        "T_T_C": np.asarray(T_T_C, dtype=float).tolist(),
    }


def _select_records(records: list[dict[str, Any]], record_indices: list[int] | None) -> list[tuple[int, dict[str, Any]]]:
    if record_indices is None or len(record_indices) == 0:
        return list(enumerate(records))

    selected: list[tuple[int, dict[str, Any]]] = []
    for record_index in record_indices:
        if record_index < 0 or record_index >= len(records):
            raise IndexError(
                f"record index {record_index} is outside the valid range [0, {len(records) - 1}]"
            )
        selected.append((int(record_index), records[record_index]))
    return selected


def _build_common_metadata_extras(
    input_meta: dict[str, Any],
    args: argparse.Namespace,
    megapose_backend: str,
) -> dict[str, Any]:
    metadata_extras = {
        "target_origin_shift_m": input_meta.get("target_origin_shift_m"),
        "target_origin_shift_frame": input_meta.get("target_origin_shift_frame"),
        "origin_shift_applied": input_meta.get("origin_shift_applied"),
        "source_pose_variant": "megapose_refined_from_truth_shifted",
        "input_source_pose_variant": input_meta.get("source_pose_variant", "truth_translation_shifted"),
        "megapose_model": str(args.model),
        "megapose_mesh_label": str(args.mesh_label),
        "megapose_mesh_units": str(args.mesh_units),
        "megapose_n_pose_hypotheses": int(args.n_pose_hypotheses),
        "megapose_n_refiner_iterations": int(args.n_refiner_iterations),
        "megapose_backend": str(megapose_backend),
        "megapose_initial_estimate_source": "input_pose_key",
        "megapose_initial_estimate_source_pose_variant": input_meta.get("source_pose_variant", "truth_translation_shifted"),
    }
    return {key: value for key, value in metadata_extras.items() if value is not None}


def _write_result_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "output_token",
        "input_record_index",
        "source_frame_index",
        "source_frame_name",
        "status",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "error",
        "timing_str",
        "time_s",
        "frame_elapsed_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = _parse_args()
    args.lifted_json = args.lifted_json.expanduser().resolve()
    args.meshes_dir = args.meshes_dir.expanduser().resolve()
    args.sanity_calibration_yaml = args.sanity_calibration_yaml.expanduser().resolve()
    args.sanity_kps_file = args.sanity_kps_file.expanduser().resolve()

    if _set_megapose_logging_level is not None and str(os.environ.get("MEGAPOSE_DEBUG_PROGRESS", "")).strip():
        _set_megapose_logging_level("INFO")

    if args.n_pose_hypotheses <= 0:
        raise ValueError("--n-pose-hypotheses must be > 0")
    if args.n_refiner_iterations <= 0:
        raise ValueError("--n-refiner-iterations must be > 0")

    lifted_payload = _load_json(args.lifted_json)
    input_records = lifted_payload.get("records", [])
    if len(input_records) == 0:
        raise ValueError(f"No records found in lifted JSON: {args.lifted_json}")

    selected_records = _select_records(input_records, args.record_indices)
    first_meta = _load_json(Path(selected_records[0][1]["meta_path"]))
    mesh_path = _find_mesh_path(args.meshes_dir, args.mesh_label)
    megapose_service, megapose_backend = _create_megapose_service(
        model_name=args.model,
        meshes_dir=args.meshes_dir,
        mesh_units=args.mesh_units,
    )
    print(f"Initialized MegaPose backend={megapose_backend}", flush=True)

    output_root = args.lifted_json.parent
    frame_records: list[dict[str, Any]] = []
    record_metadata_extras: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    total_selected_records = len(selected_records)

    print(
        f"Loaded {len(input_records)} input records; processing {total_selected_records} records "
        f"with model={args.model} mesh_label={args.mesh_label}",
        flush=True,
    )

    for output_index, (input_record_index, input_record) in enumerate(selected_records):
        frame_started_s = perf_counter()
        input_meta = _load_json(Path(input_record["meta_path"]))
        K = np.asarray(input_meta["camera_K"], dtype=np.float64)
        if "pose" in input_meta and input_meta["pose"] is not None:
            input_T_T_C = _pose_key_to_T_T_C(np.asarray(input_meta["pose"], dtype = np.float64))
        else:
            input_T_T_C = np.asarray(input_meta["T_T_C"], dtype=np.float64)
        megapose_error: str | None = None
        megapose_status = "success"
        megapose_timing: dict[str, Any] = {}
        megapose_info_row: dict[str, Any] = {}

        try:
            print(
                f"[{output_index + 1}/{total_selected_records}] frame={input_meta['source_frame_name']} "
                f"starting MegaPose refinement from input pose",
                flush=True,
            )
            refined_T_T_C, megapose_result = _run_megapose_single_frame(
                service=megapose_service,
                image_path=Path(input_record["image_path"]),
                K=K,
                initial_T_T_C=input_T_T_C,
                mesh_label=args.mesh_label,
                n_pose_hypotheses=args.n_pose_hypotheses,
                n_refiner_iterations=args.n_refiner_iterations,
            )
            megapose_timing = dict(megapose_result.get("timing", {}))
            megapose_info_row = dict(megapose_result.get("info_row", {}))
        except Exception as exc:
            refined_T_T_C = input_T_T_C
            megapose_status = "fallback_input_pose"
            megapose_error = f"{type(exc).__name__}: {exc}"

        frame_records.append(
            _build_frame_record(
                input_record=input_record,
                input_meta=input_meta,
                T_T_C=refined_T_T_C,
            )
        )

        per_record_metadata = {
            "megapose_input_bbox_xyxy": [],
            "megapose_status": megapose_status,
        }
        for field_name in ("coarse_logit", "coarse_score", "pose_logit", "pose_score", "hypothesis_id"):
            if field_name in megapose_info_row:
                per_record_metadata[f"megapose_{field_name}"] = megapose_info_row[field_name]
        if megapose_error is not None:
            per_record_metadata["megapose_error"] = megapose_error
        record_metadata_extras.append(per_record_metadata)
        frame_elapsed_s = perf_counter() - frame_started_s

        result_rows.append(
            {
                "output_token": f"{output_index:05d}",
                "input_record_index": int(input_record_index),
                "source_frame_index": int(input_meta["source_frame_index"]),
                "source_frame_name": str(input_meta["source_frame_name"]),
                "status": megapose_status,
                "bbox_xmin": "",
                "bbox_ymin": "",
                "bbox_xmax": "",
                "bbox_ymax": "",
                "error": megapose_error or "",
                "timing_str": str(megapose_timing.get("timing_str", "")),
                "time_s": megapose_timing.get("time", ""),
                "frame_elapsed_s": frame_elapsed_s,
            }
        )
        progress_prefix = f"[{output_index + 1}/{total_selected_records}] frame={input_meta['source_frame_name']}"
        if megapose_status == "success":
            print(
                f"{progress_prefix} status=success "
                f"megapose_time_s={megapose_timing.get('time', '')} frame_elapsed_s={frame_elapsed_s:.2f}",
                flush=True,
            )
        else:
            print(
                f"{progress_prefix} status={megapose_status} error={megapose_error} "
                f"frame_elapsed_s={frame_elapsed_s:.2f}",
                flush=True,
            )

    camera_settings = dict(first_meta.get("camera_settings", {}))
    image_width_px, image_height_px = (int(value) for value in first_meta["output_image_size_wh"])
    common_metadata_extras = _build_common_metadata_extras(
        input_meta=first_meta,
        args=args,
        megapose_backend=megapose_backend,
    )
    output_pack_dir, output_lifted_path = write_trajectory_pack(
        output_dir=output_root,
        frame_records=frame_records,
        camera_settings=camera_settings,
        K=np.asarray(first_meta["camera_K"], dtype=np.float64),
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        pack_dir_name=args.output_pack_dir_name,
        lifted_filename=args.output_lifted_filename,
        metadata_extras=common_metadata_extras,
        record_metadata_extras=record_metadata_extras,
    )
    sanity_check_dir: Path | None = None
    if not args.skip_sanity_overlays:
        dist_coeffs = _load_dist_coeffs_from_calibration_yaml(args.sanity_calibration_yaml)
        target_points_with_origin_m = load_target_points(
                                                            kps_file = args.sanity_kps_file,
                                                            with_origin = True,
                                                            units = args.sanity_kps_units,
                                                        )
        sanity_check_dir = write_sanity_overlays(
                                traj_dir = output_pack_dir,
                                frame_records = frame_records,
                                K = np.asarray(first_meta["camera_K"], dtype = np.float64),
                                dist_coeffs = dist_coeffs,
                                target_pts_with_origin = target_points_with_origin_m,
                                bboxes_xyxy = [
                                                metadata.get("megapose_input_bbox_xyxy", [])
                                                if metadata is not None
                                                else None
                                                for metadata in record_metadata_extras
                                             ],
                           )
        print(f"Wrote sanity overlays to {sanity_check_dir}", flush=True)

    summary_stem = Path(args.output_lifted_filename).stem
    result_csv_path = output_root / f"{summary_stem}_frame_results.csv"
    result_json_path = output_root / f"{summary_stem}_frame_results.json"
    summary_path = output_root / f"{summary_stem}_summary.json"

    _write_result_csv(result_csv_path, result_rows)
    write_json_payload(result_json_path, result_rows)

    success_count = sum(1 for row in result_rows if row["status"] == "success")
    fallback_count = sum(1 for row in result_rows if row["status"] != "success")
    summary_payload = {
        "input_lifted_json": str(args.lifted_json),
        "input_num_records_total": int(len(input_records)),
        "processed_num_records": int(len(selected_records)),
        "output_trajectory_pack_dir": str(output_pack_dir),
        "output_lifted_json": str(output_lifted_path),
        "frame_results_csv": str(result_csv_path),
        "frame_results_json": str(result_json_path),
        "sanity_check_dir": str(sanity_check_dir) if sanity_check_dir is not None else "",
        "mesh_path": str(mesh_path),
        "mesh_label": str(args.mesh_label),
        "mesh_units": str(args.mesh_units),
        "megapose_model": str(args.model),
        "megapose_backend": str(megapose_backend),
        "n_pose_hypotheses": int(args.n_pose_hypotheses),
        "n_refiner_iterations": int(args.n_refiner_iterations),
        "sanity_calibration_yaml": str(args.sanity_calibration_yaml),
        "sanity_kps_file": str(args.sanity_kps_file),
        "sanity_kps_units": str(args.sanity_kps_units),
        "success_count": int(success_count),
        "fallback_count": int(fallback_count),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    write_json_payload(summary_path, summary_payload)

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    _maybe_reexec_under_python_c()
    main()
