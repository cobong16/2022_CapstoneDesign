import argparse
import os
import sys
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
import logging

# CPU 코어 사용제한하기
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

FILE = Path(__file__).resolve()  # 현재 파일 경로
ROOT = FILE.parents[0]  # 프로젝트 ROOT 폴더 경로
WEIGHTS = ROOT / 'weights'  # weights 폴더 경로

# system PATH에 프로젝트 경로 저장하기
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # ROOT 폴더 PATH에 추가
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # yolov5 폴더 PATH에 추가
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # strong_sort 폴더 PATH에 추가

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 상대 경로 저장 ("./")

# 프로젝트 폴더 내 라이브러리 불러오기
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression,
                                  scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path,
                                  strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# 중복 로깅 방지를 위해 핸들러 지우기
logging.getLogger().removeHandler(logging.getLogger().handlers[0])


@torch.no_grad()
def run(
        source=ROOT / 'SampleVideo.mp4',  # 영상 경로
        crosswalk_length=2500,  # 횡단보도 측정 거리
        yolo_weights=WEIGHTS / 'yolov5_crosswalk.pt',  # custom yolo 경로
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # strong sort 경로
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # 추론 사이즈 (height, width)
        conf_thres=0.2,  # confidence 임계값
        iou_thres=0.45,  # IoU 임계값
        max_det=100,  # 이미지 당 최대 탐지 갯수
        device='',  # cuda 지원 유무
        show_vid=False,  # 결과 영상 출력
        save_vid=False,  # 결과 영상 저장
        save_txt=False,  # *.txt로 결과 저장
        save_conf=False,  # --save-txt labels에 confience값 저장
        save_crop=False,  # cropped prediction boxes 저장
        nosave=False,  # 영상 저장하지 않기
        classes=None,  # 클래스 별 필터링: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        eval=False,  # run multi-gpu eval
):
    # 영상파일 경로
    source = str(source)
    # 올바른 확장자인지 검사
    is_file = Path(source).suffix[1:] in VID_FORMATS

    # Export 파일 경로 설정
    if not isinstance(yolo_weights, list):
        exp_name = yolo_weights.stem  # suffix 제거한 경로
    elif type(yolo_weights) is list and len(yolo_weights) == 1:
        exp_name = Path(yolo_weights[0]).stem
    else:
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    # 동일 이름일 경우 뒤에 숫자 붙여주기
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # CUDA 지원 설정하기
    if eval:
        device = torch.device(int(device))
    else:
        device = select_device(device)

    # CNN 모델 불러오기
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz=imgsz, s=stride)

    # 데이터 불러오기
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    vid_path, vid_writer, txt_path = [None], [None], [None]
    outputs = [None]

    # StrongSORT 초기화
    cfg = get_config()
    cfg.merge_from_file(config_strongsort)

    # sort 인스턴스 생성하기
    strongsort_list = [
        StrongSORT(
            strong_sort_weights,
            device,
            half,
            max_dist=cfg.STRONGSORT.MAX_DIST,
            max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
            max_age=cfg.STRONGSORT.MAX_AGE,
            n_init=cfg.STRONGSORT.N_INIT,
            nn_budget=cfg.STRONGSORT.NN_BUDGET,
            mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
            ema_alpha=cfg.STRONGSORT.EMA_ALPHA
        )
    ]

    # 횡단보도 감지 기준 선 설정하기
    crosswalk_points = None
    detecting_lines = None
    ratio = None

    # 트래킹하기
    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))
    is_green = False  # 초록불 탐지 유무
    start_green = 0.0
    waited_people_top = []  # 초록불 시작 전 대기하는 사람들 정보
    waited_people_bottom = []
    passed_top_counter = 0
    passed_bottom_counter = 0
    passed_times = []  # 횡단한 사람들 시간 정보
    passed_id = []
    target_object_id = None  # 탐지된 사람 id
    target_object_speed_meter = None
    target_remain_time = None  # 탐지된 사람 남은 예상 보행시간


    dt = [0.0, 0.0, 0.0, 0.0]
    seen = 0
    curr_frames, prev_frames = [None], [None]
    # dataset 내 정보: 영상 경로, RGB이미지, BGR이미지, 카메라 녹화 정보(사용 안 함), 영상 정보(현프레임, 경로)
    for frame_idx, (path, im, im0, vid_cap, s) in enumerate(dataset):
        # 이미지 정규화
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8을 fp16/32로 변환
        im /= 255.0  # 0 ~ 1로 정규화
        if len(im.shape) == 3:
            im = im[None]  # batch dimension을 위해 4차원 배열로 확장
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS 적용
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t4 = time_sync()
        dt[2] += t4 - t3

        # Process detections
        # Tensor array 구조: x, y, x, y, predict, type
        # type 구조: 0 - crosswalk, 1 - green, 2 - person, 3 - red
        for i, det in enumerate(pred):
            seen += 1

            p, im0c, _ = path, im0.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)
            txt_file_name = p.stem
            save_path = str(save_dir / p.name)

            curr_frames[i] = im0c

            txt_path = str(save_dir / 'tracks' / txt_file_name)
            s += '%gx%g ' % im.shape[2:]
            imc = im0c.copy() if save_crop else im0c

            annotator = Annotator(im0c, line_width=line_thickness, pil=not ascii)
            if cfg.STRONGSORT.ECC:  # Motion Compensation 확인
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):  # 1 프레임 당 감지된 텐서 arr내 객체들
                # 사이즈 리스케일링, 축소된 이미지에서 원본 이미지 좌표로 변환
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0c.shape).round()

                # log용 결과 출력
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # 클래스 당 탐지
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                xywhs = xyxy2xywh(det[:, :4])
                confs = det[:, 4]
                clss = det[:, 5]

                # strongsort 탐지 및 시간 측정
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0c)
                t5 = time_sync()
                dt[3] += t5 - t4

                # 시각화를 위한 박스 그리기
                if len(outputs[i]) > 0:
                    # 1회 횡단 보도 검사
                    if crosswalk_points is None:
                        for output in outputs[i]:
                            if output[5] == 0:
                                crosswalk_points = output[0:4]
                                print("Dectected Crosswalk")

                                crosswalk_top_line = detect_line(im0c, crosswalk_points)
                                temp = detect_points(crosswalk_points, crosswalk_top_line)
                                detecting_lines = temp[0:4]
                                # [상단 시작점, 상단 종료점, 하단 시작점, 하단 종료점, 하단비율]
                                print(f'Detected Lines: {detecting_lines}')
                                break

                    # 보행신초 초록불 후 다음 프레임 부터 계산
                    curr_time = frame_idx
                    pass_time = curr_time - start_green
                    # 특정 지점 통과 시 소요 시간 저장
                    # 10초 = 240
                    fps = 60
                    limit_time = 10 * fps
                    if is_green and pass_time <= limit_time and pass_time % (fps // 2) == 0:
                        for arr_id in waited_people_top:
                            # 상단 종료 지점
                            for output in outputs[i]:
                                if output[4] == arr_id:
                                    if output[3] >= detecting_lines[1]:
                                        if int(arr_id) not in passed_id:
                                          passed_id.append(int(arr_id))                  
                                          passed_times.append([arr_id, pass_time])
                                          passed_top_counter += 1

                        for arr_id in waited_people_bottom:
                            # 하단 종료 지점
                            for output in outputs[i]:
                                if output[4] == arr_id:
                                    if output[3] <= detecting_lines[3]:
                                        if int(arr_id) not in passed_id:
                                          passed_id.append(int(arr_id))                  
                                          passed_times.append([arr_id, pass_time])
                                          passed_bottom_counter += 1

                    # 지정 시간 이후 보행신호 점멸 단계 시간 계산하기
                    if is_green and pass_time == limit_time + 1:
                        print(f"passed_times: {passed_times}")
                        line_length = crosswalk_length / 3
                        temp_id, target_time = passed_times[0]
                        for person_id, time in passed_times:
                            if time >= target_time:
                                temp_id = person_id
                                target_time = time

                        left_time = (pass_time - target_time) // fps
                        print("left time", left_time)

                        target_object_id = int(temp_id)
                        print(f"target_object: {target_object_id}")

                        target_object_speed = round(line_length / ((target_time - 1 * fps) / fps), 2)
                        target_object_speed_meter = round(target_object_speed / 100, 2)

                        print(f"target_speed: {target_object_speed_meter} m/s")
                        remain_length = crosswalk_length - line_length
                        
                        target_remain_time = round(remain_length / target_object_speed, 2) - left_time
                        print(f"target_remain_time: {target_remain_time} sec")

                        # if len(waited_people_top) - passed_top_counter > 0 \
                        #         or len(waited_people_bottom) - passed_bottom_counter > 0:
                        #     print("아직 보행 중인 보행자 발견, 최저 보행 시간으로 출력")
                        #     print(waited_people_top)
                        #     print(waited_people_bottom)
                        # else:
                        #     print(f"passed_times: {passed_times}")
                        #     line_length = crosswalk_length / 3
                        #     temp_id, target_time = passed_times[0]
                        #     for person_id, time in passed_times:
                        #         if time >= target_time:
                        #             temp_id = person_id
                        #             target_time = time

                        #     left_time = (pass_time - target_time) // fps
                        #     print("left time", left_time)

                        #     target_object_id = int(temp_id)
                        #     print(f"target_object: {target_object_id}")

                        #     target_object_speed = round(line_length / ((target_time - 3 * fps) / fps), 2)
                        #     target_object_speed_meter = round(target_object_speed / 100, 2)

                        #     print(f"target_speed: {target_object_speed_meter} m/s")
                        #     remain_length = crosswalk_length - line_length

                        #     target_remain_time = round(remain_length / target_object_speed, 2) - left_time
                        #     print(f"target_remain_time: {target_remain_time} sec")

                    # 보행신호 초록불 탐지하기
                    clss_np = clss.cpu().numpy()
                    if 1 in clss_np and is_green is False:
                        print("Detect GreenLight")
                        is_green = True
                        start_green = frame_idx

                    # 초록불 감지 시 횡단할 사람 검출출
                    if is_green is True and pass_time == fps * 2:
                        # 횡단하기 위해 대기하는 사람들 위 아래로 나눠 저장하기
                        for output in outputs[i]:
                            if output[5] == 2:
                                if output[3] < detecting_lines[1]:
                                    waited_people_top.append(output[4])
                                elif output[3] > detecting_lines[3]:
                                    waited_people_bottom.append(output[4])

                        print(waited_people_top)
                        print(waited_people_bottom)

                    # 한 프레임에서 탐지된 객체들
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                        bboxes = output[0:4]  # 좌표 위치
                        id = output[4]  # 객체 ID값
                        cls = output[5]  # 객체 종류

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:
                            c = int(cls)
                            id = int(id)
                            if c != 0:
                                # 박스 라벨링하기
                                if target_object_id == id:
                                    label = f'{id}: {target_object_speed_meter}m/s {target_remain_time}sec'
                                    annotator.box_label(bboxes, label, color=colors(6, True))
                                else:
                                    label = f'{id}'
                                    # label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else
                                    #                                   (f'{id} {conf:.2f}' if hide_class else
                                    #                                    f'{id} {names[c]} {conf:.2f}'))
                                    annotator.box_label(bboxes, label, color=colors(c, True))
                                if save_crop:
                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                    save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                LOGGER.info(f'{s}Done. YOLO:({dt[1]:.3f}s), StrongSORT:({dt[3]:.3f}s)')
                # print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')
            else:  # 검출 박스가 하나도 없는 경우
                strongsort_list[i].increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()

            if is_green:
                im0 = cv2.line(im0, (0, detecting_lines[0]), (1279, detecting_lines[0]), (0, 0, 255), thickness=2)
                im0 = cv2.line(im0, (0, detecting_lines[2]), (1279, detecting_lines[2]), (0, 0, 255), thickness=2)
                im0 = cv2.line(im0, (0, detecting_lines[1]), (1279, detecting_lines[1]), (255, 0, 0), thickness=2)
                im0 = cv2.line(im0, (0, detecting_lines[3]), (1279, detecting_lines[3]), (255, 0, 0), thickness=2)

            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def detect_line(image, points):
    src = image.copy()
    roi = tuple(map(int, points))
    print(f"crosswalk points: {roi}")

    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])

    img_mask = cv2.inRange(src, lower, upper)
    result = cv2.bitwise_and(src, src, mask=img_mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray = gray[roi[1]:roi[1]+10, roi[0]:roi[2]]

    lines = cv2.HoughLinesP(gray, 0.8, np.pi / 180, 90, minLineLength=10, maxLineGap=50)
    arr = []

    for line in lines:
        length = round(((abs(line[0][2] - line[0][0]) ** 2) + (abs(line[0][3] - line[0][1]) ** 2)) ** 0.5, 3)
        arr.append(length)

    max_length_idx = arr.index(max(arr))
    max_length_p = tuple(lines[max_length_idx][0])

    # return [x, y, x, y]
    print(f"top line points: {max_length_p}")
    return max_length_p


def detect_points(crosswalk_points, top_line_points):
    crosswalk = crosswalk_points
    height = crosswalk[3] - crosswalk[1]
    # print(f"crosswalk height: {height}")
    # 횡단 보도 시작 지점과 끝 지점 저장
    top_start = int(round(crosswalk[1], 0))
    bottom_start = int(round(crosswalk[3], 0))

    # 횡단 보도 시작 지점 너비와 끝 지점 너비 저장
    top_point = top_line_points
    top_width = top_point[2] - top_point[0]
    bottom_width = crosswalk[2] - crosswalk[0]

    # # 두 지점 너비간의 비율 구하기
    # ratio = round(bottom_width / top_width, 2)
    # print(f"ratio is 1 : {ratio}")

    # 비례식을 이용하여 횡단보도 높이에서 위쪽 사다리꼴 높이 찾기
    top_ratio = top_width / (top_width + bottom_width)

    middle_point_length = round(height * top_ratio, 2)
    middle_point = round(top_start + middle_point_length, 0)
    # 중간 지점 너비 = 바닥길이 * 중간지점 높이 / 원래 높이
    middle_point_width = (bottom_width * middle_point_length) / height

    # 비례식을 한 번 더 이용하여 감지할 끝 지점 구하기
    # 위쪽 끝 지점 구하기
    top_ratio = top_width / (top_width + middle_point_width)
    top_end_length = round(middle_point_length * top_ratio, 0)
    top_end = top_start + top_end_length

    # 아래쪽 사다리꼴 높이 구하기
    middle_point_length = height - middle_point_length
    # 아래쪽 끝 지점 구하기
    bottom_ratio = middle_point_width / (middle_point_width + bottom_width)
    bottom_end_length = round(middle_point_length * bottom_ratio, 0)
    bottom_end = middle_point + bottom_end_length

    # bottom_end_length = (height - middle_point_length) * bottom_ratio
    # bottom_end = round(bottom_start - bottom_end_length, 0)

    # 감지할 네 지점 반환하기
    return top_start, int(top_end), bottom_start, int(bottom_end)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov5_crosswalk.pt',
                        help='model.pt path(s)')
    parser.add_argument('--crosswalk-length', type=int, default=2500)
    parser.add_argument('--strong-sort-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--eval', action='store_true', help='run evaluation')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
