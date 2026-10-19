import time
import traceback

import cv2
import numpy as np
import torch
from torch.multiprocessing import Process, Queue

from general_navigation.carla.client import CarlaClient
from general_navigation.carla.utils import start_carla
from general_navigation.common import UIRecorder
from general_navigation.models.model_utils import generate_visual
from general_navigation.schema.environment import DroneControls
from general_navigation.settings import settings

from general_navigation.models.factory import (
    get_default_config,
    get_model,
    get_weights,
)


def async_nomad(input_q: Queue, output_q: Queue, model, device):
    model.to(device)
    model.eval()

    image_history = []
    context_frames = 4

    with torch.no_grad():
        while True:
            try:
                drone_state = input_q.get()
                
                # 從正確的 drone_state 取得車頭前視相機畫面
                img = drone_state.image.cv_image() 
                img = cv2.resize(img, (96, 96))  # 對齊 yaml 的 96x96
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)

                image_history.append(img_tensor)
                if len(image_history) > context_frames:
                    image_history.pop(0)

                # 若歷史幀還沒收集滿，給予停止指令
                if len(image_history) < context_frames:
                    controls = DroneControls(
                        trajectory=[(0, 0)],
                        trajectory_mpc=[(0, 0)],
                        speed=0.0,
                        steer=0.0,
                    )
                    output_q.put(controls.model_dump())
                    continue

                # 1. 將 4 張影像在通道維度串接 (4 * 3 = 12 通道)
                obs_images = torch.cat(image_history, dim=1)
                
                # 2. 目標影像設為當前幀
                goal_images = image_history[-1]  
                # input_goal_mask = torch.zeros(1, device=device)
                input_goal_mask = torch.ones(1, device=device)

                # 3. 透過 NoMaD 取得全局特徵
                global_cond = model(
                    "vision_encoder",
                    obs_img=obs_images,
                    goal_img=goal_images,
                    input_goal_mask=input_goal_mask
                )

                # 4. 透過擴散模型預測軌跡
                pred_horizon = 8  
                noisy_action = torch.randn(1, pred_horizon, 2, device=device)
                timestep = torch.tensor([0], device=device, dtype=torch.long)
                
                pred_noise = model(
                    "noise_pred_net",
                    sample=noisy_action,
                    timestep=timestep,
                    global_cond=global_cond
                )
                
                pred_traj = noisy_action - pred_noise
                trajectory_np = pred_traj.squeeze(0).cpu().numpy() * 5.0

                # 5. 轉換成控制器格式
                trajectory_list = [(float(pt[0]), float(pt[1])) for pt in trajectory_np]
                dx = trajectory_list[0][0]
                steer_val = np.clip(dx * 1.5, -1.0, 1.0)

                controls = DroneControls(
                    trajectory=trajectory_list,
                    trajectory_mpc=trajectory_list,
                    speed=1.0,  # 設定適當的前進車速
                    steer=steer_val,
                )
                
                output_q.put(controls.model_dump())

            except Exception as ex:
                print("Exception while running NoMaD inference:", ex)
                traceback.print_exc()


def main():
    # if you want to simulate a specific scene
    import carla
    native_client = carla.Client("localhost", 2000)
    native_client.set_timeout(10.0)
    native_client.load_world("Town02")
    #

    client = CarlaClient()

    # fix the vehicle to spawn point 0
    world = native_client.get_world()
    spawn_points = world.get_map().get_spawn_points()
    vehicles = world.get_actors().filter('vehicle.*')
    if vehicles:
        vehicles[0].set_transform(spawn_points[0])  # 永遠固定在第 0 個生成點
        print("Vehicle fixed to spawn point 0.")
    #

    rec = UIRecorder()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 取得預設的 nomad.yaml 設定
    print("Loading NoMaD configuration...")
    config = get_default_config("nomad.yaml")
    # breakpoint()

    # 2. 透過 factory 建立模型與子網路
    print("Initializing NoMaD model...")
    model = get_model(config)

    # 3. 自動下載與載入模型權重
    print("Loading model weights...")
    model = get_weights(config, model, device)
    model.to(device)
    model.eval()

    drone_state = client.get_car_state()
    last_update = drone_state.timestamp

    nomad_input_q = Queue(maxsize=1)
    nomad_output_q = Queue(maxsize=1)

    nomad_process = Process(
        target=async_nomad, args=(nomad_input_q, nomad_output_q, model, device)
    )
    nomad_process.daemon = True
    nomad_process.start()

    nomad_controls = DroneControls(
        trajectory=[(0, 0)],
        trajectory_mpc=[(0, 0)],
        speed=0.0,
        steer=0.0,
    )

    try:
        while True:
            start_time = time.time()
            client.game_loop()

            drone_state = client.get_car_state(default=drone_state)

            if nomad_input_q.empty():
                nomad_input_q.put(drone_state)

            # 取得 NoMaD 控制指令
            if not nomad_output_q.empty():
                nomad_controls_dict = nomad_output_q.get()
                nomad_controls = DroneControls(**nomad_controls_dict)
                client.set_car_controls(nomad_controls)
                last_update = nomad_controls.timestamp

            visual = generate_visual(drone_state, nomad_controls)
            rec.write(visual)

            duration = time.time() - start_time
            fps = 1.0 / duration
            print(f"FPS: {fps:.2f} ; Time: {duration:.4f}")

    except KeyboardInterrupt:
        print("Exiting on keyboard interrupt...")
    except Exception:
        traceback.print_exc()
    finally:
        nomad_process.kill()


if __name__ == "__main__":
    from torch.multiprocessing import set_start_method

    try:
        set_start_method("spawn")
    except RuntimeError:
        print("Torch multiprocessing error, cannot proceed")
        exit()

    with start_carla():
        main()