from flask import Flask, Response
import cv2
import carla
import numpy as np
import time
import threading

app = Flask(__name__)

latest_frame = None
camera_sensor = None  # 宣告成全域變數，防止被 Python GC 回收

def camera_callback(image):
    global latest_frame
    array = np.array(image.raw_data)
    array = array.reshape((image.height, image.width, 4))
    latest_frame = array[:, :, :3]  # BGRA 轉 BGR

def init_carla_listener():
    global latest_frame, camera_sensor
    while True:
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(10.0)
            world = client.get_world()
            blueprint_library = world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '480')

            print("Stream server waiting for vehicle in CARLA...")
            while True:
                vehicles = world.get_actors().filter('vehicle.*')
                if len(vehicles) > 0:
                    transform = carla.Transform(carla.Location(x=1.6, z=1.7))
                    # 賦值給全域變數 camera_sensor
                    camera_sensor = world.spawn_actor(camera_bp, transform, attach_to=vehicles[0])
                    camera_sensor.listen(lambda image: camera_callback(image))
                    print("Stream server successfully attached to car camera!")
                    break
                time.sleep(1.0)
            break
        except Exception as e:
            print(f"Connecting to CARLA failed: {e}, retrying in 3 seconds...")
            time.sleep(3.0)

def generate_frames():
    global latest_frame
    while True:
        if latest_frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for CARLA / Vehicle...", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            frame = latest_frame

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=init_carla_listener, daemon=True).start()
    app.run(host='0.0.0.0', port=8080, threaded=True)