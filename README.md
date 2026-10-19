First, running CARLA simulator
```bash
cd  ~/nomad/carla
./CarlaUE4.sh -RenderOffScreen
```
Open another terminal.
```bash
cd  ~/nomad/general-navigation
bash carla_obstacle_test.sh
```
Where,
```bash
--goal-distance 20
# Distance along the road from the starting point to the goal, in meters.

--goal-radius 2
# The robot is considered to have arrived when it is within 2 meters of the goal.

--obstacle-before-goal 10
# Distance along the road from the obstacle center to the goal, in meters.
# The obstacle is placed before the goal.
# Distance from start to obstacle = goal-distance - obstacle-before-goal.
```
