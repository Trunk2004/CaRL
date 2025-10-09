## Change Notes version 1.1 release:

* Improved leaderboard determinism by replacing threaded scenario setup with sackful coroutines. For more information see this [issue](https://github.com/carla-simulator/leaderboard/issues/192).
* Fixed rare cases where Traffic Light was not rendered during training
* Added feature to log routes during training where the agent achieved low performance. The feature is activated by setting the`RECORD=1` and `SAVE_PATH=/some_path` environment variables.
* Reduced number of threads CARLA servers use which improves memory usage
* Fixed a bug where during training occasionally an action repeat was applied.
* Fixed various bugs that could crash the software during training.
* Made custom CARLA version that features bugfixes preventing crashes and reducing CPU requirements without affecting performance. For more information see these issues [1](https://github.com/carla-simulator/carla/issues/9172), [2](https://github.com/carla-simulator/carla/issues/9250)
* Make model inference deterministic during evaluation. This requires the CUBLAS_WORKSPACE_CONFIG=:4096:8 environment variable.
* Trained and released a CaRL v1.1 model. It achieves SOTA performance (73 DS) on longest6 v2.
