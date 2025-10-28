#!/usr/bin/env python3
import random
import sys
import time
import os

import carla
sys.path.insert(0, r'/home/trung/CaRL')
from CARLA.team_code.eval_agent import EvalAgent

sys.path.insert(0, r'/home/trung/CaRL/CARLA/original_leaderboard/scenario_runner')
sys.path.insert(0,r'/home/trung/CaRL/CARLA/original_leaderboard/leaderboard')
sys.path.insert(0, r'/home/trung/CaRL/CARLA/carla')
sys.path.insert(0,r'/home/trung/CaRL/CARLA/carla/PythonAPI')
sys.path.insert(0, r'/home/trung/CaRL/CARLA/carla/PythonAPI/carla')

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

def main():
    # ---- user-editable paths ----
    CONF_DIR       = "/home/trung/CaRL/CARLA/results/CaRL_PY_01"  # contains config.json + model*.pth
    TOWN          = "Town01"

    # (optional) minimal env toggles the agent reads
    os.environ.setdefault("DEBUG_ENV_AGENT", "0")
    os.environ.setdefault("RECORD", "0")
    os.environ.setdefault("SAMPLE_TYPE", "mean")
    os.environ.setdefault("CPP", "0")
    os.environ.setdefault("HIGH_FREQ_INFERENCE", "0")
    #os.environ.setdefault("CPP_PORT", "5555")
    
    os.environ.setdefault("PYTORCH_KERNEL_CACHE_PATH", os.path.expanduser("~/.cache/torch"))
    # os.environ.setdefault("SAVE_PATH", "/tmp/carl_runs")  # enable if RECORD/DEBUG later

    # ---- connect & sync world (10 Hz like your test) ----
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.load_world(TOWN)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1  # 10 Hz
    world.apply_settings(settings)

    CarlaDataProvider.set_world(world)

    # spectator helper
    spawns = world.get_map().get_spawn_points()
    start_tf = spawns[5]
    def spect_tf(trans):
        return carla.Transform(
            trans.location + carla.Location(z=30),
            carla.Rotation(pitch=-90, yaw=trans.rotation.yaw)
        )
    spectator = world.get_spectator()
    spectator.set_transform(spect_tf(start_tf))

    # spawn ego
    ego_vehicle = CarlaDataProvider.request_new_actor(
        'vehicle.lincoln.mkz_2020', start_tf, rolename='hero', autopilot=False
    )
    world.tick()
    CarlaDataProvider.on_carla_tick()

    # GRP for routing
    grp = CarlaDataProvider.get_global_route_planner()

    # choose destination and build dense global plan
    dest_loc = spawns[50].location

    def draw_route(world: carla.World, route, life_time: float = 20.0):
        """
        Draw a route returned by GlobalRoutePlanner.trace_route.
        route: list of (Waypoint, RoadOption)
        """
        dbg = world.debug
        for i, (wp, ropt) in enumerate(route):
            loc = wp.transform.location + carla.Location(z=0.3)
            # draw a small point
            dbg.draw_point(loc, size=0.1, color=carla.Color(0, 255, 0), life_time=life_time)
            # draw an arrow to next point (except last)
            if i < len(route) - 1:
                nxt = route[i + 1][0].transform.location + carla.Location(z=0.3)
                dbg.draw_line(loc, nxt, thickness=0.05, color=carla.Color(255, 0, 0), life_time=life_time)
            # optional index label
            dbg.draw_string(loc, str(i), draw_shadow=False, color=carla.Color(255, 255, 255), life_time=life_time)

    route = grp.trace_route(start_tf.location, dest_loc)
    draw_route(world, route, life_time=999)
    dense_plan = [(wp.transform, ropt) for (wp, ropt) in route]

    # ---- load & setup agent ----
    agent = EvalAgent('127.0.0.1', '2000')
    agent.setup(CONF_DIR)  # loads config.json, models, etc.

    # provide the dense plan the agent expects (what Leaderboard normally injects)
    agent.dense_global_plan_world_coord = dense_plan

    def spawn_background_vehicles(world: carla.World, num: int = 100, tm_port: int = 8000):
        """
        Spawn `num` background vehicles using CarlaDataProvider, enable autopilot (Traffic Manager),
        and return the list of spawned actors.
        """
        bp_lib = world.get_blueprint_library()
        vehicle_bps = bp_lib.filter("vehicle.*")

        vehicles = []
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        count = 0
        for sp in spawn_points:
            if count >= num:
                break
            bp = random.choice(vehicle_bps)
            model_id = bp.id  # e.g., "vehicle.tesla.model3"
            try:
                # CarlaDataProvider expects a *model string*, not a blueprint
                veh = CarlaDataProvider.request_new_actor(model_id, sp, rolename="traffic")
                if veh is not None:
                    # enable autopilot via Traffic Manager (specify TM port for consistency)
                    veh.set_autopilot(True, tm_port)
                    vehicles.append(veh)
                    count += 1
            except Exception:
                # spawn failed (occupied, invalid pose, etc.) — skip
                continue

        print(f"Spawned {len(vehicles)} background vehicles with autopilot")
        return vehicles
    vehicles = spawn_background_vehicles(world, num=100)

    # ---- main loop ----
    try:
        for i in range(5000):
            CarlaDataProvider.on_carla_tick()
            world.tick()
            time.sleep(0.01)

            control = agent.run_step(None, i)  # timestamp=i is enough for the agent loop
            ego_vehicle.apply_control(control)

            if i % 2 == 0:
                spectator.set_transform(spect_tf(ego_vehicle.get_transform()))
            if i % 10 == 0:
                print(f"Dist to dest: {ego_vehicle.get_location().distance(dest_loc):.2f} m")
                if ego_vehicle.get_location().distance(dest_loc) < 12.0:
                    break
    finally:
        agent.destroy()
        ego_vehicle.destroy()
        for veh in vehicles:
            veh.destroy()

if __name__ == "__main__":
    main()
