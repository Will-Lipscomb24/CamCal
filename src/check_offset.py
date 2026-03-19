""" An example script of how to use the Pinhole Camera model with calibration imagery data, see CamCal repo, that have all the same camera parameters """
from pathlib import Path
import json 
import cv2
from scipy.spatial.transform import Rotation as R
import os, shutil
import pandas as pd
import numpy as np
from numpy.typing import NDArray

# local imports
from sc_pose.mathtils.quaternion import rotm2q, q2trfm, q2rotm
from sc_pose.sensors.camera import PinholeCamera
from sc_pose.sensors.camera_projections import PoseProjector, draw_uv_points_on_image


################################ Helper Functions ################################
def T4x4_inv(T4x4: NDArray) -> NDArray:
    """ 
    Invert a 4x4 homogeneous transformation matrix
    of the form:
    [ R | t ]
    [ 0 | 1 ]
    where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
    The inverse is given by:
    [ R^T | -R^T t ]
    [ 0   | 1       ] 
    """
    Rmat            = T4x4[:3,:3]
    tvec            = T4x4[:3,3]
    Rmat_inv        = Rmat.T
    t_inv           = -Rmat_inv @ tvec
    T4x4_inv        = np.eye(4)
    T4x4_inv[:3,:3] = Rmat_inv
    T4x4_inv[:3,3]  = t_inv
    return T4x4_inv



def T4x4_2_uv(T4x4_A_B: NDArray, cam: PinholeCamera, pts_A: NDArray) -> NDArray:
    """ Project 3D points in A frame to 2D pixel coordinates in the camera frame, given the 4x4 homogeneous transformation from A to B and the camera object with calibration data """
    # Trfm_A_B    = T4x4_A_B[:3,:3]
    # tr_B_A_in_B = T4x4_A_B[:3,3]
    # # pts_B       = Trfm_A_B @ pts_A + tr_B_A_in_B
    pts_B           = np.empty_like(pts_A, dtype = float)
    for i, p in enumerate(pts_A):
        p_h_A       = np.array([p[0], p[1], p[2], 1.0]) # homogeneous point
        p_h_B       = T4x4_A_B @ p_h_A
        pts_B[i]    = p_h_B[:3]
    uv              = cam.project_camera3Dxyz_to_imageUV(pts_B)
    return uv

def _process_vicon_offset_v01(row, T_CvC, T_TvT, vicon_keys):
    """ 
    Process Vicon data using 4x4 homogeneous transformation matrices.

    Returns:
    q_C_2_T
        Active quaternion from the true target frame to the true camera frame.
    r_Co2To_C
        Translation from the true camera origin to the true target origin,
        expressed in the true camera frame.
    T_T_C
        Homogeneous transform ^C T_T.

    Notation:
    ^ A T_B is a 4x4 homogeneous transformation matrix that transforms points from frame B to frame A
    [ A^R_B | A^t_{Ao -> B}]: 
    A^R_B is a passive rotation matrix from frame B to frame A, meaning it rotates the coordinate axes of frame B to align with frame A
    A^t_{Ao -> B} is a translation vector from the origin of frame A to the origin of frame B, expressed in frame A's coordinates 
    """

    soho_x          = float(row[vicon_keys['x_target']]) * 1E-3
    soho_y          = float(row[vicon_keys['y_target']]) * 1E-3
    soho_z          = float(row[vicon_keys['z_target']]) * 1E-3
    soho_qw         = float(row[vicon_keys['qw_target']])
    soho_qx         = float(row[vicon_keys['qx_target']])
    soho_qy         = float(row[vicon_keys['qy_target']])
    soho_qz         = float(row[vicon_keys['qz_target']])
    # ^V t_{Vo->Tvo}
    soho_VTv        = np.array( [ soho_x, soho_y, soho_z ] )
    # q_V_2_Tv, representing the rotation from the vicon frame to target vicon frame
    soho_quatVTv    = np.array( [ soho_qw, soho_qx, soho_qy, soho_qz ] )
    Trfm_TvV        = q2rotm( soho_quatVTv )
    # q2rotm gives active rotation matrix that rotates from Vicon frame to Target Vicon Frame: Rotm_{V\to Tv}
    # Rotm_{V\to Tv} = (Trfm_VTv)^T = Trfm_TvV (which is what we want )

    cam_x           = float(row[vicon_keys['x_cam']]) * 1E-3
    cam_y           = float(row[vicon_keys['y_cam']]) * 1E-3
    cam_z           = float(row[vicon_keys['z_cam']]) * 1E-3
    cam_qw          = float(row[vicon_keys['qw_cam']])
    cam_qx          = float(row[vicon_keys['qx_cam']])
    cam_qy          = float(row[vicon_keys['qy_cam']])
    cam_qz          = float(row[vicon_keys['qz_cam']])    
    # ^V t_{Vo->Cvo}
    cam_VCv         = np.array( [ cam_x, cam_y, cam_z ] )
    # q_V_2_Cv, representing the rotation from the vicon frame to camera vicon frame
    cam_quatVCv     = np.array( [ cam_qw, cam_qx, cam_qy, cam_qz ] )
    Trfm_CvV        = q2rotm( cam_quatVCv ) 
    # q2rotm gives active rotation matrix that rotates from Vicon frame to Camera Vicon Frame: Rotm_{V\to Cv}
    # Rotm_{V\to Cv} = (Trfm_VCv)^T = Trfm_CvV (which is what we want )
    
    # ==========================================================
    # build T_TvV = ^V T_Tv  (transform from Tv coordinates to V)
    # ==========================================================
    # we have soho_VTv, which is in the V frame
    # this implies the T 4x4 Homogenous relates Target Vicon frame to the Vicon frame, T_TvV
    T_TvV           = np.eye(4)
    T_TvV[:3, 3]    = soho_VTv
    T_TvV[:3, :3]   = Trfm_TvV

    # ==========================================================
    # Build T_CvV = ^V T_Cv  (transform from Cv coordinates to V)
    # ==========================================================
    # we have cam_VCv, which is in the V frame
    # this implies the T 4x4 Homogenous relates Camera Vicon frame to the Vicon frame, T_CvV
    T_CvV           = np.eye(4)
    T_CvV[:3, 3]    = cam_VCv
    T_CvV[:3, :3]   = Trfm_CvV # R_CvV
    
    T_T_C           = T_CvC @ (T4x4_inv(T_CvV) @ T_TvV) @ T4x4_inv(T_TvT)  # ^C T_T
    Trfm_T_C        = T_T_C[:3, :3]
    Rotm_C_T        = Trfm_T_C.T
    q_C_2_T         = rotm2q(Rotm_C_T)
    r_Co2To_C       = T_T_C[:3, 3]

    return q_C_2_T, r_Co2To_C, T_T_C

def _process_vicon_offset_v02(row, T_CvC, T_TvT, vicon_keys):
        """
        Process vicon data using passive rotation matrices and translation vectors
        
        T_CvC: transformation from Vicon camera frame to true camera frame
        T_TvT: transformation from Vicon target frame to true target frame
        """
    
        soho_x          = float(row[vicon_keys['x_target']]) * 1E-3
        soho_y          = float(row[vicon_keys['y_target']]) * 1E-3
        soho_z          = float(row[vicon_keys['z_target']]) * 1E-3
        soho_qw         = float(row[vicon_keys['qw_target']])
        soho_qx         = float(row[vicon_keys['qx_target']])
        soho_qy         = float(row[vicon_keys['qy_target']])
        soho_qz         = float(row[vicon_keys['qz_target']])
        soho_VTv        = np.array( [ soho_x, soho_y, soho_z ] ).T
        soho_quatVTv    = np.array( [ soho_qw, soho_qx, soho_qy, soho_qz ] ).T

        cam_x           = float(row[vicon_keys['x_cam']]) * 1E-3
        cam_y           = float(row[vicon_keys['y_cam']]) * 1E-3
        cam_z           = float(row[vicon_keys['z_cam']]) * 1E-3
        cam_qw          = float(row[vicon_keys['qw_cam']])
        cam_qx          = float(row[vicon_keys['qx_cam']])
        cam_qy          = float(row[vicon_keys['qy_cam']])
        cam_qz          = float(row[vicon_keys['qz_cam']])
        cam_VCv         = np.array( [ cam_x, cam_y, cam_z ] ).T
        cam_quatVCv     = np.array( [ cam_qw, cam_qx, cam_qy, cam_qz ] ).T

        R_CCv           = T_CvC[:3,:3]
        R_TTv           = T_TvT[:3,:3]
        t_CCv           = T_CvC[:3,3]
        t_TTv           = T_TvT[:3,3]
        Trfm_TvV        = q2trfm( soho_quatVTv )
        Trfm_CvV        = q2trfm( cam_quatVCv )
        Trfm_VTv        = Trfm_TvV.T
        Trfm_VCv        = Trfm_CvV.T
        Trfm_CCv        = R_CCv.T
        Trfm_TTv        = R_TTv.T
        Trfm_VC         = ( Trfm_CCv @ Trfm_CvV ).T
        Trfm_VT         = ( Trfm_TTv @ Trfm_TvV ).T
        
        # post-offset
        translation_posto   = Trfm_VC.T \
                            @ ( \
                                ( soho_VTv + Trfm_VTv @ t_TTv ) \
                              - ( cam_VCv  + Trfm_VCv @ t_CCv ) \
                            )
        Trfm_TC_posto       = ( Trfm_VT.T @ Trfm_VC )
        quaternion_posto    = rotm2q( Trfm_TC_posto.T )

        r_Co2To_CAM         = translation_posto
        q_C_2_T             = quaternion_posto

        # # pre-offset
        # translation_preo    = Trfm_CvV \
        #                     @ ( \
        #                         soho_VTv  \
        #                       - cam_VCv   \
        #                     )
        # Trfm_TC_preo        =  ( Trfm_TvV @ Trfm_CvV.T )
        # quaternion_preo     = rotm2q( Trfm_TC_preo.T )
        T_T_C           = np.eye(4)
        T_T_C[:3, :3]   = q2trfm(q_C_2_T).T
        T_T_C[:3, 3]    = r_Co2To_CAM

        return q_C_2_T, r_Co2To_CAM, T_T_C


def _load_offset_estimates(offset_data_path, offset_keys):
    # extract offset estimates from json
    with open(offset_data_path, 'r') as f:
        offset_json                         = json.load(f)
    offset_key_list                     = list(offset_keys)
    CV_C_key                            = offset_keys[offset_key_list[0]] # "Trf_4x4_CamViconDef_Cam"
    TV_T_key                            = offset_keys[offset_key_list[1]] # "Trf_4x4_TargetViconDef_Target"
    Trf4x4_CAMVICON_2_CAM_TRUE          = np.array( offset_json[CV_C_key] )
    Trf4x4_TARGETVICON_2_TARGET_TRUE    = np.array( offset_json[TV_T_key] )
    
    return Trf4x4_CAMVICON_2_CAM_TRUE, Trf4x4_TARGETVICON_2_TARGET_TRUE

def _process_opencv_pose_v01(rvec, tvec):
    """Process OpenCV pose estimates as T_B_C = ^C T_B using scipy."""
    Rotm_C_B        = R.from_rotvec(rvec).as_matrix()
    Rt              = R.from_matrix(Rotm_C_B)
    q_xyzw          = Rt.as_quat()
    q_C_2_B         = np.roll(q_xyzw, 1)
    r_Co2Bo_C       = np.asarray(tvec, dtype=float).reshape(3,)
    T_B_C           = np.eye(4)
    T_B_C[:3, :3]   = Rotm_C_B
    T_B_C[:3, 3]    = r_Co2Bo_C
    return q_C_2_B, r_Co2Bo_C, T_B_C

def _process_opencv_pose_v02(rvec, tvec):
    """Process OpenCV pose estimates as T_B_C = ^C T_B using cv2."""
    Rotm_C_B, _     = cv2.Rodrigues(rvec)
    q_C_2_B         = rotm2q(Rotm_C_B)
    r_Co2Bo_C       = np.asarray(tvec, dtype = float).reshape(3,)
    T_B_C           = np.eye(4)
    T_B_C[:3, :3]   = Rotm_C_B
    T_B_C[:3, 3]    = r_Co2Bo_C
    return q_C_2_B, r_Co2Bo_C, T_B_C


def _select_processor(processor_name, processor_map, processor_kind):
    """ resolve a version string from the Inputs section into a callable function """
    try:
        return processor_map[processor_name]
    except KeyError as exc:
        available   = ", ".join(sorted(processor_map))
        raise ValueError(
            f"Unknown {processor_kind} '{processor_name}'. Expected one of: {available}"
        ) from exc

################################ Helper Functions ################################
def main():
    HERE                = Path(__file__).parent.resolve()
    PARENT              = HERE.parent 
    ##################################### Inputs #####################################
    HERE                    = Path(__file__).resolve().parent
    PARENT                  = HERE.parent
    DATA_ROOT               = PARENT / "data" / "offset"
    EXP_NAME                = "collection_001"
    DATA_FOLDER             = DATA_ROOT / EXP_NAME
    IMAGE_DIR               = DATA_FOLDER / "images"
    IMG_SUFFIX              = ".png"
    VICON_CSV_PATH          = DATA_FOLDER / "vicon_data.csv"
    CALIBRATION_YAML_PATH   = DATA_FOLDER / "calibration.yaml"
    # corresponding results path
    RESULT_PATH             = PARENT / "results" / EXP_NAME
    INPUT_JSON_PATH         = RESULT_PATH / "calc_offset_results.json"
    INPUT_OPENCV_PATH       = RESULT_PATH / "calc_camera_poses.csv"
    # output projections     
    OUTPUT_PROJECTION_DIR   = RESULT_PATH / "reprojection"


    # choose type of vicon and opencv pose processing
    selected_vicon_offset_processor = "v01"  # options: v01, v02
    selected_opencv_pose_processor  = "v02"  # options: v01, v02
    # camera parameters 
    img_width       = 4096 
    img_height      = 3000
    focal_length    = 25.0  # in mm 
    sensor_width    = 14.13  # in mm
    sensor_height   = 10.35 # in mm

    # opencv_keys
    opencv_keys     = {
                        'frame': 'frame',
                        'rvec_x': 'rvec_x',
                        'rvec_y': 'rvec_y',
                        'rvec_z': 'rvec_z',
                        'tvec_x': 'tvec_x',
                        'tvec_y': 'tvec_y',
                        'tvec_z': 'tvec_z'
                    }
    # vicon keys
    vicon_keys      = {
                        'frame': 'image_number',
                        'x_target': 'soho_x',
                        'y_target': 'soho_y',
                        'z_target': 'soho_z',
                        'qw_target': 'soho_qw',
                        'qx_target': 'soho_qx',
                        'qy_target': 'soho_qy',
                        'qz_target': 'soho_qz',
                        'x_cam': 'cam_x',
                        'y_cam': 'cam_y',
                        'z_cam': 'cam_z',
                        'qw_cam': 'cam_qw',
                        'qx_cam': 'cam_qx',
                        'qy_cam': 'cam_qy',
                        'qz_cam': 'cam_qz'
                    }

    # offset keys
    offset_keys     = {
                        'Trf_4x4_CamViconDef_Cam': 'T_CvC',
                        'Trf_4x4_TargetViconDef_Target': 'T_TvT'
                    }
    ##################################### Inputs #####################################

    ############################## Secondary Input Setup #############################


    vicon_offset_processors = {
                                "v01": _process_vicon_offset_v01,
                                "v02": _process_vicon_offset_v02
                            }

    opencv_pose_processors  = {
                                "v01": _process_opencv_pose_v01,
                                "v02": _process_opencv_pose_v02
                            }

    process_vicon_offset    = _select_processor(
                                                    selected_vicon_offset_processor,
                                                    vicon_offset_processors,
                                                    "vicon offset pose processor"
                                                )
    process_opencv_pose     = _select_processor(
                                                    selected_opencv_pose_processor,
                                                    opencv_pose_processors,
                                                    "opencv pose processor"
                                                )
    ############################## Secondary Input Setup #############################
    # make results path
    if res_path.exists():
        print(f"Warning: {res_path} already exists, deleting and recreating...")
        shutil.rmtree(res_path)
    os.makedirs(res_path, exist_ok = True)
    print(f"Results will be saved to: {res_path}")


    # create projection object
    cam     = PinholeCamera(
                                sensor_width_mm = sensor_width,
                                sensor_height_mm = sensor_height,
                                image_width_px  = img_width,
                                image_height_px = img_height,
                                focal_length_mm = focal_length
                            )
    # cam.print_state()

    # set calibration
    cam.set_calibration_yaml(calib_data)
    Kmat_cal    = cam.calc_Kmat()
    dist_coeffs = cam._dist_coeffs_as_array()
    proj        = PoseProjector(camera = cam)




    # kps from drew/anand
    with open(kps_file, 'r') as f:
        kps_mm          = np.array( json.load(f) ) # obj in mm
        kps_m           = kps_mm / 1e3
        target_BFF_pts  = kps_m

    # add origin to start of kps
    target_BFF_pts_with_origin  = np.vstack( (np.zeros((1,3)), target_BFF_pts) ) 
    
    # extract pose estimates
    try:
        opencv_df       = pd.read_csv(opencv_pose_est)
    except FileNotFoundError:
        print(f"Error: OpenCV pose estimates file not found at {opencv_pose_est}, going to just do vicon reprojections without opencv comparison...")
        opencv_df   = pd.DataFrame() # empty dataframe to skip opencv reprojection loop

    try:
        vicon_df        = pd.read_csv(vicon_pose_est)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Vicon pose estimates file not found at {vicon_pose_est}, cannot do reprojections without pose estimates, exiting...")


    # load both opencv and vicon data into dataframes
    # iterate through rows
    Rmats           = [] 
    trs             = []
    img_paths       = []
    img_nums        = []

    # matched loop
    for i, row in opencv_df.iterrows():
        img_name    = row[opencv_keys['frame']]
        img_base    = Path(img_name).stem 
        img_num     = int(img_base.split("_")[-1])
        print(f"Processing row {i}: {img_name}")
        img_path    = image_folder / img_name 
        img_outpath = res_path / f"opencv_reproj_{img_base}.png" 
        # extract opencv outputted rotation and translation
        rvec        = np.array([row['rvec_x'], row['rvec_y'], row['rvec_z']])
        tvec        = np.array([row['tvec_x'], row['tvec_y'], row['tvec_z']])

        q_C_2_B, r_Co2Bo_C, T_B_C  = process_opencv_pose(rvec, tvec)
        img_paths.append(img_path)
        img_nums.append(img_num)

        # project 3D points to 2D image coordinates
        uv_cam  = proj.classless_pinhole_project_T4x4_2_uv(
                                                            T_TARGET_CAM       = T_B_C,
                                                            Kmat               = Kmat_cal,
                                                            BC_dist_coeffs     = dist_coeffs,
                                                            points_xyz_TARGET  = target_BFF_pts_with_origin
                                                        )
        
        # draw the points and save the verification image
        # Using str(img_path) to ensure compatibility with cv2/drawing functions
        img_out = draw_uv_points_on_image(
                                            img_or_path     = str(img_path),
                                            points_uv       = uv_cam,
                                            point_color     = (0, 0, 255),
                                            point_radius    = 15, 
                                            point_thickness = 2
                                        )
        # highlight origin point in a different color
        uv_origin   = uv_cam[0]  # assuming the first point is the origin
        img_out     = draw_uv_points_on_image(
                                                img_or_path     = img_out,
                                                points_uv       = uv_origin.reshape(1, 2),
                                                point_color     = (0, 255, 0),  # Green for origin
                                                point_radius    = 20, 
                                                point_thickness = 3
                                            )
        cv2.imwrite(img_outpath, img_out)

        vicon_match = vicon_df[vicon_df[vicon_keys['frame']] == img_num]
        if vicon_match.empty:
            print(f"Image number {img_num} for {img_name} not found in vicon data, skipping...")
            continue
        vicon_row   = vicon_match.iloc[0, :]
        
        vicon_img_outpath       = res_path / f"vicon_reproj_{img_base}.png" 
        combined_img_out_path   = res_path / f"combined_reproj_{img_base}.png" 
        Trf4x4_CAMVICON_2_CAM_TRUE, Trf4x4_TARGETVICON_2_TARGET_TRUE    = _load_offset_estimates(
                                                                                                    offset_data_path = offset_data, 
                                                                                                    offset_keys = offset_keys
                                                                                                )
        q_proc_C_2_T, r_proc_Co2To_C, T_T_C                                    = process_vicon_offset(
                                                                                                        row = vicon_row,
                                                                                                        T_CvC = Trf4x4_CAMVICON_2_CAM_TRUE,
                                                                                                        T_TvT = Trf4x4_TARGETVICON_2_TARGET_TRUE,
                                                                                                        vicon_keys = vicon_keys
                                                                                                  )
        # TODO: patch inputs from q_proc_C_2_T into projection function 
        # uv_cam_vicon    = proj.classless_pinhole_project_to_image(
        #                                                             q_CAM_2_TARGET    = q_proc_C_2_T,
        #                                                             r_Co2To_CAM       = r_proc_Co2To_C,
        #                                                             Kmat              = Kmat_cal,
        #                                                             BC_dist_coeffs    = dist_coeffs,
        #                                                             points_xyz_TARGET = target_BFF_pts_with_origin
        #                                                       )
        #                                                                                             
        uv_cam_vicon    = proj.classless_pinhole_project_T4x4_2_uv(
                                                                    T_TARGET_CAM      = T_T_C,
                                                                    Kmat              = Kmat_cal,
                                                                    BC_dist_coeffs    = dist_coeffs,
                                                                    points_xyz_TARGET = target_BFF_pts_with_origin
                                                                )
        img_vicon_out   = draw_uv_points_on_image(
                                                    img_or_path     = str(img_path),
                                                    points_uv       = uv_cam_vicon,
                                                    point_color     = (255, 0, 0),
                                                    point_radius    = 15, 
                                                    point_thickness = 2
                                                )
        # highlight origin point in a different color
        uv_origin_vicon = uv_cam_vicon[0]  # assuming the first point is the origin
        img_vicon_out   = draw_uv_points_on_image(
                                                    img_or_path     = img_vicon_out,
                                                    points_uv       = uv_origin_vicon.reshape(1, 2),
                                                    point_color     = (0, 255, 0),  # Green for origin
                                                    point_radius    = 20, 
                                                    point_thickness = 3
                                                )
        cv2.imwrite(vicon_img_outpath, img_vicon_out)
        # combined
        img_overlay_out = draw_uv_points_on_image(
                                                    img_or_path     = img_out,
                                                    points_uv       = uv_cam_vicon,
                                                    point_color     = (128, 0, 128),
                                                    point_radius    = 17, 
                                                    point_thickness = 3
                                                )
        cv2.imwrite(combined_img_out_path, img_overlay_out)


    trs_array       = np.array(trs)  # (N, 3)
    Rmats_array     = np.array(Rmats)  # (N, 3, 3)

    # reproject all loop based on the pose estimates from vicon

    print(f'Results located at: {res_path}')

if __name__ == "__main__":
    main()
