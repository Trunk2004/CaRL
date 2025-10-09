'''
Script that statrs n carla servers, n carla leaderboard clients and a PPO training with them.
'''

import subprocess
import time
import sys
import shlex
import psutil
import argparse
import os
import re
import socket


def strtobool(v):
  return str(v).lower() in ('yes', 'y', 'true', 't', '1', 'True')


def next_free_port(port=1024, max_port=65535):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  while port <= max_port:
    try:
      sock.bind(('', port))
      sock.close()
      return port
    except OSError:
      port += 1
  raise IOError('no free ports')


def kill(proc_pid):
  if psutil.pid_exists(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
      try:
        proc.kill()
      except psutil.NoSuchProcess:  # Catch the error caused by the process no longer existing
        pass  # Ignore it
    try:
      process.kill()
    except psutil.NoSuchProcess:  # Catch the error caused by the process no longer existing
      pass  # Ignore it


def kill_all_carla_servers(ports):
  # Need a failsafe way to find and kill all carla servers. We do so by port.
  for proc in psutil.process_iter():
    # check whether the process name matches
    try:
      proc_connections = proc.connections(kind='all')
    except (PermissionError, psutil.AccessDenied, psutil.NoSuchProcess):  # Avoid sudo processes
      proc_connections = None

    if proc_connections is not None:
      for conns in proc_connections:
        if not isinstance(conns.laddr, str):  # Avoid unix paths
          if conns.laddr.port in ports:
            try:
              proc.kill()
            except psutil.NoSuchProcess:  # Catch the error caused by the process no longer existing
              pass  # Ignore it


def cleanup(carla_procs, leaderboard_procs, train_proc, c_ports):
  kill_all_carla_servers(c_ports)

  for carla_proc in carla_procs:
    kill(carla_proc.pid)

  for leaderboard_proc in leaderboard_procs:
    kill(leaderboard_proc.pid)

  if train_proc is not None:
    kill(train_proc.pid)


if __name__ == '__main__':
  try:
    training = True
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--exp_name', type=str, default='PPO_000', help='the name of this experiment')
    parser.add_argument('--git_root',
                        type=str,
                        default=r'/home/jaeger/ordnung/internal/CaRL/CARLA',
                        help='root folder of 2_carla')
    parser.add_argument('--gpu_ids',
                        nargs='+',
                        default=0,
                        type=int,
                        help='GPUs to run the training on. Numer of ids must be equal to --num_envs'
                        'Training runs on the first gpu')
    parser.add_argument('--carla_root',
                        default=r'/home/jaeger/ordnung/internal/carla_9_15',
                        type=str,
                        help='Path to the .sif file containing carla 0.9.15')
    parser.add_argument('--start_port',
                        default=1024,
                        type=int,
                        help='Lowest port to use. Increase the number if you want to run multiple versions of this '
                        'script on the same machine.')
    parser.add_argument('--train_towns',
                        nargs='+',
                        default=(1, 2, 3, 4, 5, 6),
                        type=int,
                        help='Towns the CARLA servers train on. Numer of ids must be equal to --num_envs')
    parser.add_argument('--routes_folder',
                        default=r'roach_preprocessed_routes',
                        type=str,
                        help='Folder in custom_leaderboard/leaderboard/data/ that contains the routes')
    parser.add_argument('--num_envs_per_gpu',
                        default=8,
                        type=int,
                        help='Number of environments per GPU. Only used with dd_ppo.')
    parser.add_argument('--seed', type=int, default=0, help='seed of the experiment')

    parser.add_argument('--num_envs_per_node',
                        default=1,
                        type=int,
                        help='Total number of environments to train with.'
                        'on this machine.')
    parser.add_argument('--num_nodes', default=1, type=int, help='Number of machines to train on.')
    parser.add_argument('--node_id', default=0, type=int, help='Id of the node that this file is running on.')
    parser.add_argument('--rdzv_addr', default='localhost', type=str, help='IP for torchrun to sync gradients over')
    parser.add_argument('--rdzv_port', default=0, type=int, help='port for torchrun to sync gradients over')
    parser.add_argument('--ml_cloud', default=0, type=int, help='Whether the script is run on the ML cloud.')
    parser.add_argument('--use_traj_sync_ppo',
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        nargs='?',
                        const=True,
                        help='if True Run each env in a separate process.')
    parser.add_argument('--train_cpp',
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        nargs='?',
                        const=True,
                        help='whether to train with the c++ training code.')
    parser.add_argument('--PYTORCH_KERNEL_CACHE_PATH',
                        type=str,
                        default='~/.cache',
                        help='path to a cache folder for libtorch (used only in C++)')
    parser.add_argument('--ppo_cpp_install_path',
                        type=str,
                        default='~/ppo.cpp/install',
                        help='path to where the ppo.cpp executable is installed')
    parser.add_argument('--cpp_singularity_file_path',
                        type=str,
                        default='/mnt/bernhard/code/ppo.cpp/tools/ppo_cpp.sif',
                        help='path to the singularity .sif file for c++')
    parser.add_argument('--cpp_system_lib_path_1',
                        type=str,
                        default='/usr/lib/x86_64-linux-gnu',
                        help='path that contains libcudart.so.11.0')
    parser.add_argument('--cpp_system_lib_path_2',
                        type=str,
                        default='/usr/local/cuda/lib64',
                        help='path that contains ?')
    parser.add_argument('--route_repetitions',
                        type=int,
                        default=10,
                        help='How often to repeat training routes. needs to be high enough so they do not run out, '
                        'but low enough to save RAM.')
    parser.add_argument('--debug',
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        nargs='?',
                        const=True,
                        help='exits after each crash when debugging.')
    parser.add_argument('--carla_singularity',
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        nargs='?',
                        const=True,
                        help='whether to run CARLA from a singularity path')
    parser.add_argument('--carla_singularity_path',
                        type=str,
                        default='/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/team_code_roach/custom_carla_container.sif',
                        help='/path/to/custom_carla_container.sif')

    args, unknown = parser.parse_known_args()
    git_root = args.git_root
    raw_logdir = os.path.join(git_root, 'results')
    logdir = os.path.join(raw_logdir, args.exp_name)
    route_root_folder = os.path.join(git_root, fr'custom_leaderboard/leaderboard/data/{args.routes_folder}')
    route_start_id = args.num_envs_per_gpu * args.node_id
    route_end_id = 32  # TODO find suitable solution for multinode. route_start_id + args.num_envs_per_gpu
    id_to_townfile_mapping = {
        1: [
            os.path.join(route_root_folder, f'route_Town01_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        2: [
            os.path.join(route_root_folder, f'route_Town02_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        3: [
            os.path.join(route_root_folder, f'route_Town03_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        4: [
            os.path.join(route_root_folder, f'route_Town04_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        5: [
            os.path.join(route_root_folder, f'route_Town05_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        6: [
            os.path.join(route_root_folder, f'route_Town06_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        7: [
            os.path.join(route_root_folder, f'route_Town07_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        10: [
            os.path.join(route_root_folder, f'route_Town10HD_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        12: [
            os.path.join(route_root_folder, f'route_Town12_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        13: [
            os.path.join(route_root_folder, f'route_Town13_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        15: [
            os.path.join(route_root_folder, f'route_Town15_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
    }
    route_files = []
    for town_id in args.train_towns:
      route_files.append(id_to_townfile_mapping[town_id].pop(0))

    # CARLA has a bug where it spams Error messages to stderr freezing the entire codebase, including restarts
    # To prevent that we redirect the std err output of CARLA servers to null.
    blackhole = open(os.devnull, 'w', encoding='utf-8')  # pylint: disable=locally-disabled, consider-using-with
    server_outs = []
    server_errs = []
    client_outs = []
    client_errs = []
    for i in range(args.num_envs_per_node):
      if args.debug:
        server_outs.append(open(f"{raw_logdir}/logs/server_out_{i:03d}.txt", 'w', encoding='utf-8'))
        server_errs.append(open(f"{raw_logdir}/logs/server_err_{i:03d}.txt", 'w', encoding='utf-8'))
        client_outs.append(open(f"{raw_logdir}/logs/client_out_{i:03d}.txt", 'w', encoding='utf-8'))
        client_errs.append(open(f"{raw_logdir}/logs/client_err_{i:03d}.txt", 'w', encoding='utf-8'))
      else:
        server_outs.append(blackhole)
        server_errs.append(blackhole)
        client_outs.append(blackhole)
        client_errs.append(blackhole)
    client_ports = []
    current_port = args.start_port + 5000 * args.node_id
    skip_next_route = 'False'

    while training:
      if args.debug:
        training = False # Do not restart after a crash when debugging.
      train_process = None
      rl_ports = []
      traffic_manager_ports = []
      sensor_ports = []
      client_ports = []
      carla_primary_ports = []
      if current_port > 60000:
        current_port = args.start_port
      for i in range(args.num_envs_per_node):
        current_port = next_free_port(current_port)
        rl_ports.append(current_port)
        current_port += 3
        current_port = next_free_port(current_port)
        traffic_manager_ports.append(current_port)
        current_port += 3
        current_port = next_free_port(current_port)
        sensor_ports.append(current_port)
        current_port += 3
        current_port = next_free_port(current_port)
        client_ports.append(current_port)
        current_port += 3
        current_port = next_free_port(current_port)
        carla_primary_ports.append(current_port)
        current_port += 3

        if args.num_nodes > 1:
          # Multinode training assume we only run one training job within a node, so we can pick a port
          # Port needs to be consistent across nodes
          tcp_store_port = 7000
        else:
          # Single node training might have this script running multiple times. Find a free local port
          tcp_store_port = next_free_port(7000)

      carla_processes = []
      leaderboard_processes = []

      num_threads_per_server = 2

      if args.ml_cloud:
        for i in range(args.num_envs_per_node):
          print(f'Start server {i}')
          # The -nullrhi option prevents CARLA from using the GPU at all (no rendering will happen).
          # set graphicsadapter to {args.gpu_ids[i]} if actually using the gpu
          if args.carla_singularity:
            carla_processes.append(
                subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
                  f'singularity exec --nv --bind {args.carla_root}:{args.carla_root},{raw_logdir}:{raw_logdir} {args.carla_singularity_path} '
                    f'bash {args.carla_root}/CarlaUE4.sh -carla-rpc-port={client_ports[i]} -nosound -nullrhi '
                    f'-carla-primary-port={carla_primary_ports[i]} -carla-streaming-port={sensor_ports[i]} '
                    f'-RenderOffScreen -graphicsadapter=0 -RPCThreads={num_threads_per_server} -StreamingThreads={num_threads_per_server} -SecondaryThreads={num_threads_per_server} -nothreading',
                    shell=True, stdout=server_outs[i], stderr=server_errs[i]))
          else:
            carla_processes.append(
                subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
                  f'LD_LIBRARY_PATH={os.environ["CONDA_PREFIX"]}/lib:$LD_LIBRARY_PATH '
                    f'bash {args.carla_root}/CarlaUE4.sh -carla-rpc-port={client_ports[i]} -nosound -nullrhi '
                    f'-carla-primary-port={carla_primary_ports[i]} -carla-streaming-port={sensor_ports[i]} '
                    f'-RenderOffScreen -graphicsadapter=0 -RPCThreads={num_threads_per_server} -StreamingThreads={num_threads_per_server} -SecondaryThreads={num_threads_per_server} -nothreading',
                    shell=True, stdout=server_outs[i], stderr=server_errs[i]))
          time.sleep(7)

        for i in range(args.num_envs_per_node):
          print(f'Start client {i}')
          leaderboard_processes.append(
              subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
                f'LD_LIBRARY_PATH={os.environ["CONDA_PREFIX"]}/lib:$LD_LIBRARY_PATH '
                  f'bash start_leaderboard.sh {git_root} {route_files[i]} {logdir} '
                  f'{i} {client_ports[i]} {traffic_manager_ports[i]} {rl_ports[i]} {args.seed} {skip_next_route} '
                  f'{args.route_repetitions}',
                  shell=True, stdout=client_outs[i], stderr=client_errs[i]))
          time.sleep(0.2)
      else:
        for i in range(args.num_envs_per_node):
          print(f'Start server {i}')
          # The -nullrhi option prevents CARLA from using the GPU at all (no rendering will happen).
          # set graphicsadapter to {args.gpu_ids[i]} if actually using the gpu

          if args.carla_singularity:
            carla_processes.append(
                subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
                  f'singularity exec --nv --bind {args.carla_root}:{args.carla_root},{raw_logdir}:{raw_logdir} {args.carla_singularity_path} '
                    f'bash {args.carla_root}/CarlaUE4.sh -carla-rpc-port={client_ports[i]} -nosound -nullrhi '
                    f'-carla-primary-port={carla_primary_ports[i]} -carla-streaming-port={sensor_ports[i]} '
                    f'-RenderOffScreen -graphicsadapter=0 -RPCThreads={num_threads_per_server} -StreamingThreads={num_threads_per_server} -SecondaryThreads={num_threads_per_server} -nothreading',
                    shell=True, stdout=server_outs[i], stderr=server_errs[i]))
          else:
            carla_processes.append(
                subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
                  f'LD_LIBRARY_PATH={os.environ["CONDA_PREFIX"]}/lib:$LD_LIBRARY_PATH '
                    f'bash {args.carla_root}/CarlaUE4.sh -carla-rpc-port={client_ports[i]} -nosound -nullrhi '
                    f'-carla-primary-port={carla_primary_ports[i]} -carla-streaming-port={sensor_ports[i]} '
                    f'-RenderOffScreen -graphicsadapter=0 -RPCThreads={num_threads_per_server} -StreamingThreads={num_threads_per_server} -SecondaryThreads={num_threads_per_server} -nothreading',
                    shell=True, stdout=server_outs[i], stderr=server_errs[i]))
          time.sleep(0.02)
          print(f'Start client {i}')
          leaderboard_processes.append(
              subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
                  f'bash start_leaderboard.sh {git_root} {route_files[i]} {logdir} '
                  f'{i} {client_ports[i]} {traffic_manager_ports[i]} {rl_ports[i]} {args.seed} {skip_next_route} '
                  f'{args.route_repetitions}',
                  shell=True, stdout=client_outs[i], stderr=client_errs[i]))
          time.sleep(0.02)

      skip_next_route = 'False'  # After one route (potentially) was skipped we reset the variable

      cmdline = ' '.join(map(shlex.quote, sys.argv[1:]))
      str_ports = ' '.join(str(x) for x in rl_ports)
      cpp_str_ports = ' '.join('--ports ' + str(x) for x in rl_ports)

      # Find latest model file in case training resumes.
      load_file = None
      largest_step = 0
      if os.path.exists(logdir):
        for file in os.listdir(logdir):
          if file.startswith('model_latest_') and file.endswith('.pth'):
            full_path = os.path.join(logdir, file)
            if os.path.getsize(full_path) > 0:
              numbers_in_string = re.findall(r'\d+', file)
              if len(numbers_in_string) > 0:
                start_step = int(numbers_in_string[0])  # That step was already finished.
                if start_step > largest_step:
                  largest_step = start_step
                  load_file = os.path.join(logdir, file)

      if args.use_traj_sync_ppo:
        num_processes = args.num_envs_per_node
        num_envs_per_proc = 1
        print(f'Num processes : {num_processes}')
      else:
        num_processes = args.num_envs_per_node // args.num_envs_per_gpu
        num_envs_per_proc = args.num_envs_per_gpu

      if args.debug:
        train_out = open(f"{raw_logdir}/logs/train_out.txt", 'w', encoding='utf-8')
        train_err = open(f"{raw_logdir}/logs/train_err.txt", 'w', encoding='utf-8')
      else:
        train_out = sys.stdout
        train_err = sys.stderr

      if args.train_cpp:
        cpp_str_gpu_ids = ' '.join('--gpu_ids ' + str(x) for x in args.gpu_ids)
        num_envs = args.num_envs_per_node * args.num_nodes
        unknown_str = ' '.join(str(x) for x in unknown)
        #   --num_envs_per_proc {num_envs_per_proc} {cmdline}
        train_process = subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
            f'bash start_learner_ac_ppo.sh {git_root} {num_processes} {args.num_nodes} {args.rdzv_addr} '
            f'{args.rdzv_port} {args.PYTORCH_KERNEL_CACHE_PATH} {args.ppo_cpp_install_path} {raw_logdir} '
            f'{args.cpp_singularity_file_path} {args.cpp_system_lib_path_1} {args.cpp_system_lib_path_2} '
            f'{cpp_str_ports} --load_file {load_file} --num_envs {num_envs} --exp_name {args.exp_name} '
            f'--tcp_store_port {tcp_store_port} {cpp_str_gpu_ids} {unknown_str}',
            shell=True, stdout=train_out, stderr=train_err)
      else:
        train_process = subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
            f'bash start_learner_dd_ppo.sh {git_root} {num_processes} {args.num_nodes} {args.rdzv_addr} '
            f'{args.rdzv_port} {cmdline} --ports {str_ports} --logdir {raw_logdir} --load_file {load_file} '
            f'--num_envs_per_proc {num_envs_per_proc} --tcp_store_port {tcp_store_port}',
            shell=True, stdout=train_out, stderr=train_err)

      time.sleep(1)

      all_processes_running = True
      ended_leaderboard = []
      ended_carla = []
      for idx, _ in enumerate(carla_processes):
        ended_leaderboard.append(idx)
        ended_carla.append(idx)
      while all_processes_running:
        time.sleep(30)
        if train_process.poll() is not None:
          all_processes_running = False
          print('Train process ended')
        for idx, carla_process in enumerate(carla_processes):
          if carla_process.poll() is not None:
            all_processes_running = False
            print('Carla server crashed')
            skip_next_route = 'True'
        for idx, leaderboard_process in enumerate(leaderboard_processes):
          if leaderboard_process.poll() is not None:
            all_processes_running = False
            print('Leaderboard process ended')
            skip_next_route = 'True'

      for i in range(360):
        for idx, carla_process in enumerate(carla_processes):
          if carla_process.poll() is not None:
            if idx in ended_carla:
              ended_carla.remove(idx)
              print(f"Server {idx} terminated")
        for idx, leaderboard_process in enumerate(leaderboard_processes):
          if leaderboard_process.poll() is not None:
            if idx in ended_leaderboard:
              ended_leaderboard.remove(idx)
        time.sleep(1)

      for idx in ended_leaderboard:
        print(f"Leaderboard {idx} is hanging and did not terminate")

      print('Process finished:', train_process.returncode)
      if train_process.returncode == 0:
        print('Training finished succesfully')
        training = False

      cleanup(carla_processes, leaderboard_processes, train_process, client_ports)
      time.sleep(10)
      del carla_processes
      del leaderboard_processes
      del train_process

    blackhole.close()
    print('Finished cleanup')

  # Useful for debugging if the script cleans up before it shuts down.
  except KeyboardInterrupt:
    cleanup(carla_processes, leaderboard_processes, train_process, client_ports)
    blackhole.close()
    sys.exit(-1)
