# Changelog

<!-- Next changelog -->
## NVIDIA Nemo Run 0.6.0

### Detailed Changelogs:

## Executors

<details><summary>Changelog</summary>

- Added Pre-Launch Commands Support to LeptonExecutor [#312](https://github.com/NVIDIA-NeMo/Run/pull/312)
- Remove breaking torchrun config for single-node runs [#292](https://github.com/NVIDIA-NeMo/Run/pull/292)
- Upgrade skypilot to v0.10.0, introduce network_tier [#297](https://github.com/NVIDIA-NeMo/Run/pull/297)
- Fixes for multi-node execution with torchrun + LocalExecutor [#251](https://github.com/NVIDIA-NeMo/Run/pull/251)
- Add option to specify --container-env for srun [#293](https://github.com/NVIDIA-NeMo/Run/pull/293)
- Fix skypilot archive mount bug [#288](https://github.com/NVIDIA-NeMo/Run/pull/288)
- finetune on dgxcloud with nemo-run and deploy on bedrock example [#286](https://github.com/NVIDIA-NeMo/Run/pull/286)

</details>

## Ray Integration

<details><summary>Changelog</summary>

- Add nsys patch in ray sub template [#318](https://github.com/NVIDIA-NeMo/Run/pull/318)
- Add logs dir to container mount for ray slurm [#287](https://github.com/NVIDIA-NeMo/Run/pull/287)
- Allow customizing folder for SlurmRayRequest [#281](https://github.com/NVIDIA-NeMo/Run/pull/281)

</details>

## CLI & Configuration

<details><summary>Changelog</summary>

</details>

## Experiment & Job Management

<details><summary>Changelog</summary>

- Use thread pool for status, run methods inside experiment + other fixes [#295](https://github.com/NVIDIA-NeMo/Run/pull/295)

</details>

## Packaging & Deployment

<details><summary>Changelog</summary>

- Correctly append tar files for packaging [#317](https://github.com/NVIDIA-NeMo/Run/pull/317)

</details>

## Documentation

<details><summary>Changelog</summary>

- Create CHANGELOG.md [#314](https://github.com/NVIDIA-NeMo/Run/pull/314)
- docs: Fixing doc build issue [#290](https://github.com/NVIDIA-NeMo/Run/pull/290)
- fix docs tutorial links and add intro to guides/index.md [#285](https://github.com/NVIDIA-NeMo/Run/pull/285)
- README [#277](https://github.com/NVIDIA-NeMo/Run/pull/277)

</details>

## CI/CD

<details><summary>Changelog</summary>

- changelog workflow [#315](https://github.com/NVIDIA-NeMo/Run/pull/315)
- Update release.yml [#306](https://github.com/NVIDIA-NeMo/Run/pull/306)
- ci(fix): Use GITHUB_TOKEN for community bot [#302](https://github.com/NVIDIA-NeMo/Run/pull/302)
- ci: Add community-bot [#300](https://github.com/NVIDIA-NeMo/Run/pull/300)

</details>

## Bug Fixes

<details><summary>Changelog</summary>

- [Bugfix] Adding a check for name length [#273](https://github.com/NVIDIA-NeMo/Run/pull/273)
- misc fixes [#280](https://github.com/NVIDIA-NeMo/Run/pull/280)
- adding fix for lowercase and name length k8s requirements [#274](https://github.com/NVIDIA-NeMo/Run/pull/274)

</details>

## Others

<details><summary>Changelog</summary>

- Specify nodes for gpu metrics collection and split data to each rank [#320](https://github.com/NVIDIA-NeMo/Run/pull/320)
- Apply '_enable_goodbye_message' check to both goodbye messages. [#319](https://github.com/NVIDIA-NeMo/Run/pull/319)
- Update refs [#278](https://github.com/NVIDIA-NeMo/Run/pull/278)
- chore: Bump to version 0.6.0rc0.dev0 [#272](https://github.com/NVIDIA-NeMo/Run/pull/272)

</details>

## NVIDIA Nemo Run 0.5.0


- Fix docs warnings [#271](https://github.com/NVIDIA-NeMo/Run/pull/271)
- Fix docs build [#269](https://github.com/NVIDIA-NeMo/Run/pull/269)
- Support overlapped srun commands in Slurm Ray [#263](https://github.com/NVIDIA-NeMo/Run/pull/263)
- Refactor DGXC Lepton data mover: switch to BatchJob with auto cleanup and sleep after every run [#265](https://github.com/NVIDIA-NeMo/Run/pull/265)
- ci: Fix nemo fw template ref after migrating to new org [#256](https://github.com/NVIDIA-NeMo/Run/pull/256)
- Enable Nsys gpu device metrics [#257](https://github.com/NVIDIA-NeMo/Run/pull/257)
- Sync job code in local tunnel for Slurm Ray job [#254](https://github.com/NVIDIA-NeMo/Run/pull/254)
- Change the create dist job function to support creating a single node [#240](https://github.com/NVIDIA-NeMo/Run/pull/240)
- Making job names match Run:ai requirements and making errors more descriptive [#255](https://github.com/NVIDIA-NeMo/Run/pull/255)
- Support for %j in slurm log retrieval [#252](https://github.com/NVIDIA-NeMo/Run/pull/252)
- Add KubeRay tests for Ray APIs [#249](https://github.com/NVIDIA-NeMo/Run/pull/249)
- Upgrade skypilot executor with 0.9.2 [#246](https://github.com/NVIDIA-NeMo/Run/pull/246)  
- Add user scoping for k8s backend and log level support for Ray APIs [#247](https://github.com/NVIDIA-NeMo/Run/pull/247)
- Update to latest Lepton SDK [#248](https://github.com/NVIDIA-NeMo/Run/pull/248)
- Add storage mount options to LeptonExecutor [#237](https://github.com/NVIDIA-NeMo/Run/pull/237)
- Import guard k8s import in Ray Cluster and Job [#245](https://github.com/NVIDIA-NeMo/Run/pull/245)
- Add RayJob and Slurm support for Ray APIs + integration with run.Experiment [#236](https://github.com/NVIDIA-NeMo/Run/pull/236)
- ci: Enforce coverage [#238](https://github.com/NVIDIA-NeMo/Run/pull/238)
- Fix bug with a CLI overwrite [#235](https://github.com/NVIDIA-NeMo/Run/pull/235)
- Add LeptonExecutor support [#224](https://github.com/NVIDIA-NeMo/Run/pull/224)
- Add cancel to docker executor [#233](https://github.com/NVIDIA-NeMo/Run/pull/233)
- Change default log wait timeout to 10s [#232](https://github.com/NVIDIA-NeMo/Run/pull/232)
- Add RayCluster API with Kuberay support [#222](https://github.com/NVIDIA-NeMo/Run/pull/222)
- Add sbatch network arg [#230](https://github.com/NVIDIA-NeMo/Run/pull/230)
- chore: Update package info [#227](https://github.com/NVIDIA-NeMo/Run/pull/227)
- Add support for job groups for local executor [#220](https://github.com/NVIDIA-NeMo/Run/pull/220)
- Roll back get_underlying_types change + introduce extract_constituent [#223](https://github.com/NVIDIA-NeMo/Run/pull/223)
- Fix some bugs for --lazy in CLI [#179](https://github.com/NVIDIA-NeMo/Run/pull/179)
- Adding support for modern type-hints [#221](https://github.com/NVIDIA-NeMo/Run/pull/221)
- Fix bug in CLI with calling a factory-fn inside a list [#214](https://github.com/NVIDIA-NeMo/Run/pull/214)
- Handle more edge cases in --help [#219](https://github.com/NVIDIA-NeMo/Run/pull/219)
- Add autogenerated API reference content to the documentation [#190](https://github.com/NVIDIA-NeMo/Run/pull/190)
- Handle Callable in --help to fix nemo llm export --help error [#217](https://github.com/NVIDIA-NeMo/Run/pull/217)
- Ensure job directory creation for various schedulers [#216](https://github.com/NVIDIA-NeMo/Run/pull/216)
- Adding support for ForwardRef in CLI [#176](https://github.com/NVIDIA-NeMo/Run/pull/176)
- Add additional debug to DGXC data mover [#215](https://github.com/NVIDIA-NeMo/Run/pull/215)
- Handle ctx in entrypoint for experiment [#213](https://github.com/NVIDIA-NeMo/Run/pull/213)
- zozhang/dgxc executor data mover [#206](https://github.com/NVIDIA-NeMo/Run/pull/206)
- Add support for YAML, TOML & JSON [#182](https://github.com/NVIDIA-NeMo/Run/pull/182)
- Add clean mode for experiment to avoid printing any NeMo-Run specific logs [#208](https://github.com/NVIDIA-NeMo/Run/pull/208)
- Fix seed for torchrun [#209](https://github.com/NVIDIA-NeMo/Run/pull/209)
- Support torchrun multi node on local executor [#143](https://github.com/NVIDIA-NeMo/Run/pull/143)
- Add nsys filename param [#205](https://github.com/NVIDIA-NeMo/Run/pull/205)
- Add DGXCloudExecutor docs and update execution guide [#192](https://github.com/NVIDIA-NeMo/Run/pull/192)
- Add --cuda-event-trace=false to nsys command [#180](https://github.com/NVIDIA-NeMo/Run/pull/180)


