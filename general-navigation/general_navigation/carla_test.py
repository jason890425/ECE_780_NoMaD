from collections import deque
import argparse
import json
import time
import traceback
import cv2
import numpy as np
import torch
from torch.multiprocessing import Process, Queue
from flask import Flask, Response, render_template_string, request, jsonify
import threading
import os
import math
from queue import Empty, Full
from pathlib import Path

import pygame
import yaml
from PIL import Image as PILImage
from torchvision import transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from general_navigation.schema.image import Image

from general_navigation.carla.client import CarlaClient
from general_navigation.carla.utils import start_carla
from general_navigation.common import UIRecorder
from general_navigation.schema.environment import DroneControls
from general_navigation.settings import settings

from general_navigation.models.factory import (
    get_default_config,
    get_model,
    get_weights,
)

import carla

# 全域變數
latest_frame = None
web_throttle = 0.0
web_steer = 0.0
web_capture_triggered = False

def run_web_server():
    global web_throttle, web_steer, web_capture_triggered
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template_string("""
            <html>
                <head>
                    <title>CARLA NoMaD Remote Control</title>
                    <style>
                        body { background: #222; color: #fff; text-align: center; font-family: sans-serif; }
                        img { border: 3px solid #555; margin-top: 10px; }
                        .controls { margin-top: 15px; font-size: 15px; background: #333; display: inline-block; padding: 10px 20px; border-radius: 8px; text-align: left; }
                        #debug { color: #0ff; font-weight: bold; margin-top: 5px; }
                    </style>
                </head>
                <body>
                    <h2>CARLA NoMaD Remote Control (Port 8080)</h2>
                    <img src="/video_feed" width="640" height="480"/><br>
                    <div class="controls">
                        <b>操作說明：</b><br>
                        1. 請用滑鼠點一下上方影像畫面確保焦點。<br>
                        2. <b>W</b>：前進 | <b>S</b>：倒車<br>
                        3. <b>A / D</b>：左右轉向<br>
                        4. <b>C 鍵</b>：拍下 goal.png 並傳送回原點啟動 AI！<br>
                        <div id="debug">狀態: 等待輸入...</div>
                    </div>

                    <script>
                        let keys = {};

                        window.addEventListener('keydown', (e) => {
                            let k = e.key.toLowerCase();
                            if (['w', 'a', 's', 'd'].includes(k)) {
                                keys[k] = true;
                                e.preventDefault();
                            }
                            if (k === 'c') {
                                triggerCapture();
                            }
                        });

                        window.addEventListener('keyup', (e) => {
                            let k = e.key.toLowerCase();
                            if (['w', 'a', 's', 'd'].includes(k)) {
                                keys[k] = false;
                                e.preventDefault();
                            }
                        });

                        setInterval(() => {
                            let throttle = 0.0;
                            let steer = 0.0;

                            if (keys['w']) throttle += 1.0;
                            if (keys['s']) throttle -= 1.0;
                            if (keys['a']) steer -= 1.0;
                            if (keys['d']) steer += 1.0;

                            document.getElementById('debug').innerText = 
                                `Throttle: ${throttle.toFixed(1)} | Steer: ${steer.toFixed(1)}`;

                            fetch('/control', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ throttle: throttle, steer: steer })
                            }).catch(err => {});
                        }, 50);

                        function triggerCapture() {
                            fetch('/control', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ capture: true })
                            });
                            alert("已觸發拍照與重置！車輛將返回起點並交由 AI 導航。");
                        }
                    </script>
                </body>
            </html>
        """)

    @app.route('/video_feed')
    def video_feed():
        def generate():
            global latest_frame
            while True:
                if latest_frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Waiting for CARLA...", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    frame = latest_frame
                
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.03)
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/control', methods=['POST'])
    def control():
        global web_throttle, web_steer, web_capture_triggered
        data = request.get_json()
        if 'throttle' in data:
            web_throttle = data.get('throttle', 0.0)
        if 'steer' in data:
            web_steer = data.get('steer', 0.0)
        if data.get('capture', False):
            web_capture_triggered = True
        return jsonify(success=True)

    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=8080, threaded=True)


# NoMaD waypoints use x forward, y left; CARLA positive steer turns right.
WAYPOINT_SPACING_METERS = 0.25  # Deployment scale; calibrate for CARLA.
NUM_SAMPLES = 8
DIFFUSION_SEED = 0  # Reuse the same noise bank for each observation.
MAX_SPEED = 0.6
MIN_LOOKAHEAD = 1.5
STEER_RATE = 0.5  # Normalized steering units per simulated second.
CONTROL_DT = 0.1
CONTEXT_DT = 0.1  # Fixed simulator-time spacing, independent of inference.
# Evaluation tolerance includes modest lateral tracking error (about 0.55 m
# in the recorded run). This is not an image-based arrival prediction.
GOAL_RADIUS_METERS = 0.8  # Simulator evaluation only, never a model input.
GOAL_APPROACH_RADIUS_METERS = 1.5
GOAL_MISS_INCREASE_METERS = 0.1


def evaluate_goal_progress(distance, closest_distance, goal_radius=GOAL_RADIUS_METERS):
    """Simulator-only termination; a near miss is distinct from arrival."""
    closest_distance = min(distance, closest_distance)
    if distance <= goal_radius:
        return "reached", closest_distance
    if (closest_distance <= GOAL_APPROACH_RADIUS_METERS
            and distance > closest_distance + GOAL_MISS_INCREASE_METERS):
        return "missed", closest_distance
    return None, closest_distance


class FixedImageContext:
    def __init__(self, count, interval=CONTEXT_DT):
        self.frames = deque(maxlen=count)
        self.count = count
        self.interval = interval

    def add(self, timestamp, image, pose):
        if self.frames:
            gap = timestamp - self.frames[-1][0]
            if gap < self.interval - 1e-5:
                return False
            if gap > self.interval + 1e-5:
                # Missing sensor frames must not silently change context spacing.
                self.frames.clear()
        self.frames.append((timestamp, image.copy(), pose))
        return True

    def request(self):
        if len(self.frames) != self.count:
            return None
        return ([entry[1] for entry in self.frames],
                [entry[0] for entry in self.frames], self.frames[-1][2])


def camera_state(client):
    """Use the RGB camera surface before the HUD is composited (BGR for OpenCV)."""
    surface = client.world.camera_manager.surface
    if surface is None:
        return None
    rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    state = client.get_car_state()
    state.image = Image(data=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return state


def stopped_controls():
    return DroneControls(trajectory=[(0, 0)], trajectory_mpc=[(0, 0)],
                         speed=0.0, steer=0.0)


def vehicle_geometry(vehicle):
    wheels = vehicle.get_physics_control().wheels
    steering = [wheel for wheel in wheels if wheel.max_steer_angle > 0]
    fixed = [wheel for wheel in wheels if wheel.max_steer_angle == 0]
    if not steering or not fixed:
        raise RuntimeError("Cannot determine front/rear axles from vehicle physics")
    front = np.mean([[w.position.x, w.position.y, w.position.z] for w in steering], axis=0)
    rear = np.mean([[w.position.x, w.position.y, w.position.z] for w in fixed], axis=0)
    wheelbase = float(np.linalg.norm(front - rear) / 100.0)  # CARLA wheel positions: cm.
    max_angle = math.radians(max(w.max_steer_angle for w in steering))
    if wheelbase <= 0 or max_angle <= 0:
        raise RuntimeError("Invalid vehicle geometry")
    return wheelbase, max_angle


def vehicle_pose(vehicle):
    transform = vehicle.get_transform()
    return (transform.location.x, transform.location.y,
            math.radians(transform.rotation.yaw))


def local_to_world(points, pose):
    """NoMaD local x-forward/y-left to CARLA world XY (y-right at yaw=0)."""
    x, y, yaw = pose
    c, t = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, t], [t, -c]])
    return np.asarray(points, dtype=float) @ rotation.T + [x, y]


def world_to_local(points, pose):
    x, y, yaw = pose
    c, t = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, t], [t, -c]])
    return (np.asarray(points, dtype=float) - [x, y]) @ rotation


def select_trajectory(samples, observation_pose, previous_world, policy="consensus"):
    """Select a sampled path using official first-sample or local heuristic."""
    samples = np.asarray(samples, dtype=float)
    if policy == "official_first":
        return local_to_world(samples[0], observation_pose)
    if policy != "consensus":
        raise ValueError(f"Unknown trajectory selection policy: {policy}")

    flat = samples.reshape(len(samples), -1)
    centrality = np.linalg.norm(flat[:, None] - flat[None, :], axis=-1).mean(axis=1)
    if previous_world is not None:
        previous = world_to_local(previous_world, observation_pose)
        previous = previous[previous[:, 0] > 0.1]
        if len(previous) >= 2:
            # Compare lateral shape at matching forward distances, rather than
            # matching time indices after the vehicle has advanced.
            order = np.argsort(previous[:, 0])
            previous = previous[order]
            continuity = []
            for path in samples:
                overlap = (path[:, 0] >= previous[0, 0]) & (path[:, 0] <= previous[-1, 0])
                if overlap.any():
                    ref_y = np.interp(path[overlap, 0], previous[:, 0], previous[:, 1])
                    continuity.append(float(np.mean(np.abs(path[overlap, 1] - ref_y))))
                else:
                    continuity.append(0.0)
            centrality += 0.35 * np.asarray(continuity)
    selected = samples[int(np.argmin(centrality))]
    world_path = local_to_world(selected, observation_pose)
    return world_path


def tracking_controls(trajectory, wheelbase, max_angle, previous_steer, dt):
    points = np.asarray(trajectory, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        return stopped_controls(), None
    forward = points[points[:, 0] > 0.1]
    if not len(forward):
        return stopped_controls(), None
    distances = np.linalg.norm(forward, axis=1)
    candidates = np.flatnonzero(distances >= MIN_LOOKAHEAD)
    target = forward[candidates[0] if len(candidates) else np.argmax(distances)]
    dx, dy = target
    # A short predicted path uses its actual target distance. Using the
    # nominal lookahead here suppresses steering near a close goal.
    target_distance_squared = float(dx*dx + dy*dy)
    curvature = -2.0 * dy / target_distance_squared
    desired = float(np.clip(math.atan(wheelbase * curvature) / max_angle, -1, 1))
    # Release an obsolete turn immediately. Limit only increases in turn
    # magnitude, so an old left turn cannot persist after a right request.
    base = previous_steer
    if desired * previous_steer < 0:
        base = 0.0
    if abs(desired) <= abs(base):
        steer = desired
    else:
        steer = float(np.clip(desired, base - STEER_RATE*dt,
                              base + STEER_RATE*dt))
    speed = MAX_SPEED * (1.0 - 0.7 * abs(steer))
    controls = DroneControls(trajectory=trajectory, trajectory_mpc=trajectory,
                             speed=speed, steer=steer)
    return controls, (float(dx), float(dy), desired)


def apply_vehicle_controls(client, controls):
    """Bypass KeyboardControl's one-decimal steering rounding."""
    vehicle = client.world.player
    if controls.speed == 0:
        vehicle.disable_constant_velocity()
        client.world.constant_velocity_enabled = False
        vehicle.set_target_velocity(carla.Vector3D())
        vehicle.set_target_angular_velocity(carla.Vector3D())
    else:
        vehicle.enable_constant_velocity(carla.Vector3D(controls.speed, 0, 0))
        client.world.constant_velocity_enabled = True
    command = carla.VehicleControl(steer=float(controls.steer),
                                   brake=1.0 if controls.speed == 0 else 0.0)
    client.controller._control = command
    vehicle.apply_control(command)


def draw_nomad_paths(frame, world_samples, selected_world, pose):
    """Overlay local paths anchored at the image's bottom center (BGR).

    This is a screen-space direction/shape visualization, not a calibrated
    projection onto the road. NoMaD x-forward maps up and y-left maps left.
    """
    overlay = frame.copy()
    height, width = frame.shape[:2]
    origin = np.array([width // 2, height - 1], dtype=float)
    # Fixed scale for a given resolution avoids stretching short predictions.
    pixels_per_meter = min(width * 0.2, height * 0.3)
    yellow, blue = (0, 255, 255), (255, 0, 0)

    def draw(path, color, thickness):
        path = np.asarray(path, dtype=float)
        if (path.ndim != 2 or path.shape[1] != 2 or not len(path)
                or not np.isfinite(path).all()):
            return
        local = world_to_local(path, pose)
        # Retain the forward portion; old waypoints behind the car are not
        # relevant to this overlay. Prepend the vehicle origin to every path.
        local = local[local[:, 0] > 0]
        if not len(local):
            return
        local = np.vstack(([0.0, 0.0], local))
        pixels = origin + np.column_stack((-local[:, 1], -local[:, 0])) * pixels_per_meter
        cv2.polylines(overlay, [np.rint(pixels).astype(np.int32)], False,
                      color, thickness, cv2.LINE_AA)

    if world_samples is not None:
        for path in world_samples:
            draw(path, yellow, 1)
    # Draw last and thicker so the chosen path stays visible over candidates.
    if selected_world is not None:
        draw(selected_world, blue, 2)
    cv2.putText(overlay, "Candidates", (5, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, yellow, 1, cv2.LINE_AA)
    cv2.putText(overlay, "Selected", (5, 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, blue, 1, cv2.LINE_AA)
    return overlay


def sample_nomad_actions(model, condition, scheduler, config, device):
    """Use common random noise across observations, including DDPM steps.

    Reset a local generator on every inference, so identical conditioning does
    not change its candidate paths merely because another request ran first.
    Candidates still receive distinct noise and respond to new observations.
    """
    generator = torch.Generator(device=device).manual_seed(DIFFUSION_SEED)
    action = torch.randn(NUM_SAMPLES, config["len_traj_pred"], 2,
                         device=device, generator=generator)
    condition = condition.repeat(NUM_SAMPLES, 1)
    scheduler.set_timesteps(config["num_diffusion_iters"])
    for timestep in scheduler.timesteps:
        noise = model("noise_pred_net", sample=action,
                      timestep=timestep, global_cond=condition)
        action = scheduler.step(noise, timestep, action,
                                generator=generator).prev_sample
    return action


def nomad_image_transform(config):
    return transforms.Compose([
        transforms.Resize((config["image_size"][1], config["image_size"][0])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def async_nomad(input_q: Queue, output_q: Queue, model, device, goal_path, config):
    model.to(device)
    model.eval()
    context_frames = config["context_size"] + 1
    transform = nomad_image_transform(config)

    def image_tensor(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return transform(PILImage.fromarray(rgb)).unsqueeze(0).to(device)

    # Read the same statistics used to train the checkpoint, without importing
    # the training module and its unrelated training dependencies.
    stats_path = Path(__file__).resolve().parents[2] / "train/vint_train/data/data_config.yaml"
    with stats_path.open() as stream:
        stats = yaml.safe_load(stream)["action_stats"]
    action_min = torch.tensor(stats["min"], device=device)
    action_max = torch.tensor(stats["max"], device=device)
    scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2", clip_sample=True,
        prediction_type="epsilon",
    )
    goal_raw = cv2.imread(goal_path)
    if goal_raw is None:
        raise FileNotFoundError(f"Cannot read goal image: {goal_path}")
    goal_tensor = image_tensor(goal_raw)
    print(f"[NoMaD] Ready: context={context_frames}, diffusion={config['num_diffusion_iters']}, "
          f"samples={NUM_SAMPLES}, fixed_noise_seed={DIFFUSION_SEED}, "
          f"context_dt={CONTEXT_DT:.2f}s (simulator time)")

    with torch.inference_mode():
        while True:
            context_images, context_times, submitted_at, observation_pose = input_q.get()
            inference_start = time.monotonic()
            trajectories = None
            try:
                image_history = [image_tensor(image) for image in context_images]
                if len(context_times) != context_frames or not np.allclose(
                        np.diff(context_times), CONTEXT_DT, atol=1e-5, rtol=0):
                    raise ValueError("Context is not spaced at fixed simulator intervals")
                controls = stopped_controls()
                if len(image_history) == context_frames:
                    condition = model(
                        "vision_encoder", obs_img=torch.cat(image_history, dim=1),
                        goal_img=goal_tensor,
                        input_goal_mask=torch.zeros(1, device=device, dtype=torch.long),
                    )
                    action = sample_nomad_actions(
                        model, condition, scheduler, config, device)
                    deltas = (action + 1) / 2 * (action_max - action_min) + action_min
                    trajectories = (torch.cumsum(deltas, dim=1)
                                    * WAYPOINT_SPACING_METERS).cpu().numpy()
                    if not np.isfinite(trajectories).all():
                        raise ValueError("Non-finite trajectory")
                    # Send all candidates; selection belongs in the main
                    # process, which knows the vehicle's current pose.
                    trajectory_list = [tuple(map(float, point)) for point in trajectories[0]]
                    controls = DroneControls(trajectory=trajectory_list,
                        trajectory_mpc=trajectory_list, speed=MAX_SPEED, steer=0.0)
            except Exception:
                traceback.print_exc()
                controls = stopped_controls()
                trajectories = None
            output_q.put((controls.model_dump(), submitted_at,
                          time.monotonic() - inference_start, observation_pose,
                          None if trajectories is None else trajectories.tolist()))


def path_tracking_error(position, path):
    """Minimum XY distance to the fixed reference polyline, in meters."""
    starts, ends = path[:-1], path[1:]
    segments = ends - starts
    fraction = np.clip(np.sum((position - starts) * segments, axis=1)
                       / np.maximum(np.sum(segments**2, axis=1), 1e-12), 0, 1)
    nearest = starts + fraction[:, None] * segments
    return float(np.min(np.linalg.norm(nearest - position, axis=1)))


def run_controller_test(client, wheelbase, max_angle, output_path):
    """Track a fixed world path using the same controller as NoMaD."""
    global latest_frame
    vehicle = client.world.player
    apply_vehicle_controls(client, stopped_controls())
    # Settle the spawn before anchoring the reference path.
    for _ in range(10):
        client.sim_world.tick()
    initial_pose = vehicle_pose(vehicle)
    local_reference = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0.5]], dtype=float)
    world_reference = local_to_world(local_reference, initial_pose)
    samples, collisions = [], []
    collision_sensor = client.sim_world.spawn_actor(
        client.sim_world.get_blueprint_library().find("sensor.other.collision"),
        carla.Transform(), attach_to=vehicle)
    collision_sensor.listen(lambda event: collisions.append(event.frame))
    steer = 0.0
    outcome = "timeout"
    print("[ControllerTest] Fixed waypoints: [(1,0),(2,0),(3,0),(4,0.5)] meters; y-positive LEFT")
    try:
        for step in range(int(30 / CONTROL_DT)):
            if not client.game_loop():
                outcome = "window_closed"
                break
            pose = vehicle_pose(vehicle)
            position = np.array(pose[:2])
            remaining = float(np.linalg.norm(position - world_reference[-1]))
            error = path_tracking_error(position, world_reference)
            trajectory = world_to_local(world_reference[1:], pose).tolist()
            controls, target = tracking_controls(trajectory, wheelbase, max_angle,
                                                 steer, CONTROL_DT)
            actual = vehicle.get_control().steer
            samples.append(dict(time=step*CONTROL_DT, pose=list(pose),
                                endpoint_distance=remaining, cross_track_error=error,
                                commanded_steer=controls.steer, actual_steer=actual,
                                speed=controls.speed))
            if collisions:
                outcome = "collision"
                break
            if remaining <= 0.2:
                outcome = "reached"
                break
            if target is None:
                outcome = "no_forward_waypoint"
                break
            steer = controls.steer
            apply_vehicle_controls(client, controls)
            state = camera_state(client)
            if state is not None:
                latest_frame = draw_nomad_paths(state.image.cv_image(),
                    [world_reference[1:]], world_reference[1:], pose)
                cv2.putText(latest_frame, f"Test error={error:.2f}m end={remaining:.2f}m",
                            (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
                client.display.blit(pygame.surfarray.make_surface(
                    cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB).swapaxes(0,1)), (0,0))
                pygame.display.flip()
            if step % 10 == 0:
                print(f"[ControllerTest] t={step*CONTROL_DT:.1f}s error={error:.3f}m "
                      f"end={remaining:.3f}m steer={steer:.3f} actual={actual:.3f}")
            time.sleep(CONTROL_DT)
    except KeyboardInterrupt:
        outcome = "interrupted"
    finally:
        apply_vehicle_controls(client, stopped_controls())
        client.sim_world.tick()
        collision_sensor.stop()
        collision_sensor.destroy()
        maximum_error = max((row["cross_track_error"] for row in samples), default=None)
        passed = (outcome == "reached" and not collisions and maximum_error <= 0.5)
        report = dict(passed=passed, outcome=outcome,
                      endpoint_tolerance_m=0.2, max_cross_track_tolerance_m=0.5,
                      max_cross_track_error_m=maximum_error,
                      wheelbase_m=wheelbase, max_steer_angle_rad=max_angle,
                      reference_local=local_reference.tolist(),
                      reference_world=world_reference.tolist(),
                      collisions=collisions, samples=samples)
        Path(output_path).write_text(json.dumps(report, indent=2))
        print(f"[ControllerTest] {'PASS' if passed else 'FAIL'} outcome={outcome}, "
              f"max_error={maximum_error}; report={Path(output_path).resolve()}")
    return passed


def run_open_loop_test(client, model, device, goal_path, config, output_dir):
    """Capture stationary context and report every local model prediction."""
    global latest_frame
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    apply_vehicle_controls(client, stopped_controls())
    for _ in range(10):
        client.sim_world.tick()
    initial_pose = vehicle_pose(client.world.player)
    transform = nomad_image_transform(config)

    def tensor(bgr):
        return transform(PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)

    goal = cv2.imread(goal_path)
    if goal is None:
        raise FileNotFoundError(goal_path)
    if not cv2.imwrite(str(output / "goal.png"), goal):
        raise RuntimeError("Cannot save goal image")
    history, poses = [], []
    for index in range(config["context_size"] + 1):
        state = None
        deadline = time.monotonic() + 10
        while state is None:
            if not client.game_loop():
                raise RuntimeError("Window closed during open-loop capture")
            apply_vehicle_controls(client, stopped_controls())
            state = camera_state(client)
            if time.monotonic() > deadline:
                raise RuntimeError("Camera did not provide stationary observation")
        frame = state.image.cv_image().copy()
        latest_frame = frame
        if not cv2.imwrite(str(output / f"observation_{index}.png"), frame):
            raise RuntimeError("Cannot save observation")
        history.append(tensor(frame))
        pose = vehicle_pose(client.world.player)
        poses.append(pose)
        yaw_change = abs(math.atan2(math.sin(pose[2]-initial_pose[2]),
                                   math.cos(pose[2]-initial_pose[2])))
        if np.linalg.norm(np.array(pose[:2])-initial_pose[:2]) > 0.02 or yaw_change > math.radians(0.2):
            raise RuntimeError("Vehicle moved during open-loop capture; discard this test")
        time.sleep(CONTROL_DT)
    stats_path = Path(__file__).resolve().parents[2] / "train/vint_train/data/data_config.yaml"
    with stats_path.open() as stream:
        stats = yaml.safe_load(stream)["action_stats"]
    minimum = torch.tensor(stats["min"], device=device)
    maximum = torch.tensor(stats["max"], device=device)
    scheduler = DDPMScheduler(num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    with torch.inference_mode():
        condition = model("vision_encoder", obs_img=torch.cat(history, dim=1),
            goal_img=tensor(goal), input_goal_mask=torch.zeros(1, device=device, dtype=torch.long))
        predicted_distance = float(model("dist_pred_net", obsgoal_cond=condition).flatten()[0].item())
        action = sample_nomad_actions(model, condition, scheduler, config, device)
        deltas = (action + 1) / 2 * (maximum - minimum) + minimum
        waypoints = torch.cumsum(deltas, dim=1)
        if not torch.isfinite(waypoints).all():
            raise ValueError("Non-finite open-loop waypoints")
        # Repeat on identical tensors to check the inference path, not motion.
        repeated = sample_nomad_actions(model, condition, scheduler, config, device)
        repeat_error = float((action-repeated).abs().max().item())
    report = dict(coordinate_frame="NoMaD local: x forward, y left",
        units="normalized waypoint units; local_meters uses deployment scale, not calibrated ground truth",
        context="stationary observations, no motion history",
        fixed_noise_seed=DIFFUSION_SEED, input_goal_mask=0,
        predicted_temporal_distance=predicted_distance,
        predicted_temporal_distance_units="training temporal steps, not CARLA ticks/meters",
        diffusion_output_normalized_deltas=action.cpu().tolist(),
        denormalized_deltas=deltas.cpu().tolist(),
        local_waypoints=waypoints.cpu().tolist(),
        local_waypoints_meters=(waypoints*WAYPOINT_SPACING_METERS).cpu().tolist(),
        deployment_scale_m=WAYPOINT_SPACING_METERS,
        identical_input_repeat_max_abs_error=repeat_error,
        observation_poses=poses, config=config)
    report_path = output / "prediction.json"
    report_path.write_text(json.dumps(report, indent=2))
    for index, path in enumerate(report["local_waypoints"]):
        print(f"[OpenLoop] trajectory {index}: {np.round(path, 4).tolist()}")
    print(f"[OpenLoop] temporal distance={predicted_distance:.3f}, repeat error={repeat_error:.8f}")
    print(f"[OpenLoop] Raw local predictions and input images saved to {output.resolve()}")
    return report


def main(controller_test=False, report_path="controller_test_report.json",
         open_loop=False, open_loop_dir="open_loop_test", goal_image=None,
         path_selection="current"):
    global latest_frame, web_throttle, web_steer, web_capture_triggered

    threading.Thread(target=run_web_server, daemon=True).start()
    print(">>> 網頁伺服器已啟動於 Port 8080！")

    native_client = carla.Client("localhost", 2000)
    native_client.set_timeout(10.0)
    native_client.load_world("Town03")

    client = CarlaClient()
    sim_settings = client.sim_world.get_settings()
    sim_settings.fixed_delta_seconds = CONTROL_DT
    client.sim_world.apply_settings(sim_settings)
    camera = client.world.camera_manager
    camera.transform_index = 0  # Forward-facing, rigid camera.
    camera.set_sensor(0, notify=False, force_respawn=True)

    world = native_client.get_world()
    spawn_points = world.get_map().get_spawn_points()
    vehicles = world.get_actors().filter('vehicle.*')
    if vehicles:
        client.world.player.set_transform(spawn_points[0])  
        print("Vehicle fixed to spawn point 0.")

    wheelbase, max_angle = vehicle_geometry(client.world.player)
    print(f"[Control] wheelbase={wheelbase:.2f}m, max steer={math.degrees(max_angle):.1f}deg")
    if controller_test:
        try:
            return run_controller_test(client, wheelbase, max_angle, report_path)
        finally:
            client.world.destroy()
            pygame.quit()

    rec = UIRecorder()
    drone_state = None

    print("\n=== [PHASE 1] 網頁手動駕駛模式 ===")
    print("請前往網頁，點擊畫面後使用 W/A/S/D 開車，抵達後按 C 鍵拍照。")

    goal_captured = bool(open_loop and goal_image)
    if goal_captured:
        goal_path = str(Path(goal_image).resolve())
        if not Path(goal_path).is_file():
            raise FileNotFoundError(goal_path)
        apply_vehicle_controls(client, stopped_controls())
        camera.surface = None
        print(f"[OpenLoop] Using existing goal image: {goal_path}")

    while not goal_captured:
        client.game_loop()
        drone_state = camera_state(client)
        if drone_state is None:
            continue
        # breakpoint()
        # import cv2
        # img = drone_state.image.cv_image()
        # cv2.imwrite("/home/t254huan/drone_image.png", img)

        latest_frame = drone_state.image.cv_image()

        if web_capture_triggered:
            goal_path = os.path.abspath("goal.png")
            img_to_save = drone_state.image.cv_image()
            if not cv2.imwrite(goal_path, img_to_save):
                raise RuntimeError(f"Could not save {goal_path}")
            goal_location = client.world.player.get_location()
            goal_pose = vehicle_pose(client.world.player)
            apply_vehicle_controls(client, stopped_controls())
            print(f">>> 【成功】已拍下 goal.png 存至 {goal_path}！正在將車輛傳送回原點並啟動NoMaD")
            
            client.world.player.set_transform(spawn_points[0])
            client.world.player.set_target_velocity(carla.Vector3D())
            client.world.player.set_target_angular_velocity(carla.Vector3D())
            camera.surface = None
            camera.frame_packet = None

            goal_captured = True
            break

        manual_controls = DroneControls(
            trajectory=[(0, 0)],
            trajectory_mpc=[(0, 0)],
            speed=web_throttle * 3.0,
            steer=web_steer,
        )
        apply_vehicle_controls(client, manual_controls)
        rec.write(latest_frame)

    # =========================================================
    # 階段二：NoMaD AI 自動導航
    # =========================================================
    print("\n=== [PHASE 2] NoMaD AI 導航中 ===")
    print(f"[NoMaD] path selection: {path_selection}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_default_config("nomad.yaml")
    model = get_model(config)
    model = get_weights(config, model, device)
    model.to(device)
    model.eval()

    if open_loop:
        try:
            return run_open_loop_test(client, model, device, goal_path, config, open_loop_dir)
        finally:
            apply_vehicle_controls(client, stopped_controls())
            client.sim_world.tick()
            client.world.destroy()
            pygame.quit()

    nomad_input_q = Queue(maxsize=1)
    nomad_output_q = Queue(maxsize=1)

    goal_path = os.path.abspath("goal.png")
    
    nomad_process = Process(target=async_nomad, args=(nomad_input_q, nomad_output_q, model, device, goal_path, config))
    nomad_process.daemon = True
    nomad_process.start()

    nomad_controls = DroneControls(trajectory=[(0, 0)], trajectory_mpc=[(0, 0)], speed=0.0, steer=0.0)

    last_prediction = time.monotonic()
    prediction_timeout = 2.0
    request_pending = False
    stop_reason = None
    previous_steer = 0.0
    predicted_trajectory = None
    world_trajectory = None
    world_samples = None
    closest_goal_distance = float("inf")
    image_context = FixedImageContext(config["context_size"] + 1)
    pose_history = {}
    last_control_log = 0.0
    last_log = 0.0
    try:
        while True:
            start_time = time.time()
            client.game_loop()
            drone_state = camera_state(client)
            if drone_state is None:
                continue
            goal_distance = client.world.player.get_location().distance(goal_location)
            outcome, closest_goal_distance = evaluate_goal_progress(
                goal_distance, closest_goal_distance)
            if outcome is not None:
                apply_vehicle_controls(client, stopped_controls())
                if outcome == "reached":
                    print(f"[Evaluation] Reached recorded goal: distance={goal_distance:.3f}m, "
                          f"tolerance={GOAL_RADIUS_METERS:.2f}m; stopping.")
                else:
                    print(f"[Evaluation] Missed recorded goal: closest={closest_goal_distance:.3f}m, "
                          f"current={goal_distance:.3f}m; stopping, arrival NOT achieved.")
                break
            if not nomad_process.is_alive():
                apply_vehicle_controls(client, stopped_controls())
                raise RuntimeError("NoMaD inference process exited")
            
            latest_frame = drone_state.image.cv_image().copy()

            snapshot = client.sim_world.get_snapshot()
            actor_snapshot = snapshot.find(client.world.player.id)
            transform = actor_snapshot.get_transform()
            pose_history[snapshot.frame] = (transform.location.x, transform.location.y,
                                           math.radians(transform.rotation.yaw))
            for old_frame in list(pose_history):
                if old_frame < snapshot.frame - 32:
                    del pose_history[old_frame]
            packet = camera.frame_packet
            if packet is not None and packet[0] in pose_history:
                frame_id, sensor_time, surface = packet
                rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
                image_context.add(sensor_time, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                  pose_history[frame_id])
            # Collect every sensor frame even while inference is busy.
            context_request = image_context.request()
            if not request_pending and context_request is not None:
                context_images, context_times, observation_pose = context_request
                try:
                    nomad_input_q.put_nowait((context_images, context_times,
                                             time.monotonic(), observation_pose))
                    request_pending = True
                except Full:
                    pass
            try:
                result, submitted_at, inference_seconds, observation_pose, samples = nomad_output_q.get_nowait()
                prediction = DroneControls(**result)
                request_pending = False
                # A watchdog measures time since a result arrived, not time
                # spent computing that result. CPU inference can exceed 0.5 s.
                last_prediction = time.monotonic()
                prediction_timeout = max(2.0, 3.0 * inference_seconds)
                if prediction.speed > 0 and samples is not None:
                    world_samples = np.asarray([local_to_world(path, observation_pose)
                                                for path in samples])
                    world_trajectory = select_trajectory(
                        samples, observation_pose, world_trajectory,
                        policy="official_first" if path_selection == "official" else "consensus")
                else:
                    world_trajectory = None
                    world_samples = None
                if time.monotonic() - last_log >= 1.0:
                    print(f"[NoMaD] inference={inference_seconds:.3f}s, "
                          f"observation age={time.monotonic()-submitted_at:.3f}s, "
                          f"candidate endpoints (x-forward,y-left)="
                          f"{None if samples is None else np.round(np.asarray(samples)[:, -1, :], 3).tolist()}, "
                          f"selected world path={None if world_trajectory is None else world_trajectory.tolist()}")
                    last_log = time.monotonic()
            except Empty:
                pass
            predicted_trajectory = (None if world_trajectory is None else
                world_to_local(world_trajectory, vehicle_pose(client.world.player)).tolist())
            target = None
            if predicted_trajectory is None:
                reason = "waiting for image context / valid inference (check worker errors)"
                nomad_controls = stopped_controls()
            elif time.monotonic() - last_prediction > prediction_timeout:
                reason = "inference result watchdog timed out"
                nomad_controls = stopped_controls()
            else:
                nomad_controls, target = tracking_controls(
                    predicted_trajectory, wheelbase, max_angle, previous_steer, CONTROL_DT)
                reason = "trajectory has no valid forward waypoint" if target is None else None
            if reason != stop_reason:
                print(f"[Control] {reason or 'following predicted trajectory'}")
                stop_reason = reason
            previous_steer = nomad_controls.steer
            apply_vehicle_controls(client, nomad_controls)
            if time.monotonic() - last_control_log >= 1.0:
                actual = client.world.player.get_control().steer
                pose = vehicle_pose(client.world.player)
                distance = client.world.player.get_location().distance(goal_location)
                # Ground-truth diagnostics only; never used to select or steer.
                goal_local = world_to_local([goal_pose[:2]], pose)[0]
                goal_yaw_error = math.degrees(math.atan2(
                    math.sin(goal_pose[2] - pose[2]),
                    math.cos(goal_pose[2] - pose[2])))
                print(f"[Control] target={target}, steer={previous_steer:.3f}, "
                      f"actual={actual:.3f}, yaw={math.degrees(pose[2]):.1f}deg, "
                      f"goal_distance={distance:.2f}m, "
                      f"goal_local=({goal_local[0]:.2f}, {goal_local[1]:.2f}), "
                      f"goal_yaw_error={goal_yaw_error:.1f}deg, "
                      f"speed={nomad_controls.speed:.2f}")
                last_control_log = time.monotonic()
            draw_samples = (world_samples if time.monotonic() - last_prediction
                            <= prediction_timeout else None)
            draw_selected = world_trajectory if draw_samples is not None else None
            annotated = draw_nomad_paths(
                drone_state.image.cv_image(), draw_samples, draw_selected,
                vehicle_pose(client.world.player))
            cv2.putText(annotated, f"v={nomad_controls.speed:.2f} steer={previous_steer:.3f}",
                        (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            latest_frame = annotated
            rec.write(annotated)
            client.display.blit(pygame.surfarray.make_surface(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB).swapaxes(0, 1)), (0, 0))
            pygame.display.flip()
            time.sleep(max(0, CONTROL_DT - (time.time() - start_time)))

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        apply_vehicle_controls(client, stopped_controls())
        try:
            # In synchronous mode, commands take effect on the next tick.
            if client.sync:
                client.sim_world.tick()
            else:
                client.sim_world.wait_for_tick()
        finally:
            nomad_process.terminate()
            nomad_process.join(timeout=2)


if __name__ == "__main__":
    from torch.multiprocessing import set_start_method
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--open-loop-test", action="store_true",
                       help="Capture goal manually, reset, and predict with vehicle stopped")
    parser.add_argument("--open-loop-dir", default="open_loop_test")
    parser.add_argument("--goal-image", help="Existing goal image for open-loop testing")
    modes.add_argument("--controller-test", action="store_true",
                        help="Track fixed waypoints without loading NoMaD")
    parser.add_argument("--controller-report", default="controller_test_report.json")
    parser.add_argument("--path-selection", choices=("current", "official"),
                        default="current",
                        help="Path selection policy (default: current; official selects candidate 0)")
    parser.add_argument("--existing-carla", action="store_true",
                        help="Use an already running CARLA server")
    args = parser.parse_args()
    if args.goal_image and not args.open_loop_test:
        parser.error("--goal-image requires --open-loop-test")
    if args.existing_carla:
        main(args.controller_test, args.controller_report,
             args.open_loop_test, args.open_loop_dir, args.goal_image,
             args.path_selection)
    else:
        with start_carla():
            main(args.controller_test, args.controller_report,
             args.open_loop_test, args.open_loop_dir, args.goal_image,
             args.path_selection)