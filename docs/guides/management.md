(guides-manage-experiments)=
# Manage Experiments

The central component for managing tasks in NeMo Run is the `Experiment` class. Use it to define, launch, and manage workflows with several tasks. This guide provides an overview of the `Experiment` class, its methods, and how to use it effectively.

(guides-create-experiment)=
## Create an Experiment

Create an experiment by instantiating the `Experiment` class with a descriptive title:

```python
exp = Experiment("My Experiment")
```

When you run the experiment, NeMo Run generates a unique experiment ID that represents a single run.

:::{note}
`Experiment` is a context manager. Use `Experiment.add` and `Experiment.run` after entering the context.
:::

(guides-add-tasks)=
## Add Tasks

Add tasks to an experiment using the `add` method. This method supports the following:

- A single task, which is an instance of either `run.Partial` or `run.Script`, along with its executor.

  ```python
  with exp:
      exp.add(task_1, executor=run.LocalExecutor())
  ```

- A list of tasks, each of which is an instance of either `run.Partial` or `run.Script`, along with a single executor or a list of executors for each task in the group. The group runs tasks in parallel.

  ```python
  with exp:
      exp.add([task_2, task_3], executor=run.DockerExecutor(...))
  ```

Specify a descriptive name for the task using the `name` keyword argument.

The `add` method also accepts a list of plugins, each an instance of `run.Plugin`. Use plugins to change both the task and the executor—useful, for example, to enable a configuration option in the task and set a related environment variable in the executor.

The method returns a unique ID for the task or job. Use this ID to define dependencies between a group of tasks:

```python
with run.Experiment("dag-experiment", log_level="INFO") as exp:
    id1 = exp.add([inline_script, inline_script_sleep], tail_logs=False, name="task-1")
    id2 = exp.add([inline_script, inline_script_sleep], tail_logs=False, name="task-2")
    exp.add(
       [inline_script, inline_script_sleep],
       tail_logs=False,
       name="task-3",
       dependencies=[id1, id2], # task-3 will only run after task-1 and task-2 have completed
   )
```

(guides-launch-experiment)=
## Launch an Experiment

After you add all tasks to an experiment, launch it using the `run` method. This method takes several optional arguments, including `detach`, `sequential`, `tail_logs`, and `direct`:

- `detach`: If `True`, the experiment detaches from the process executing it. Useful when launching on a remote cluster and you want to end the process after scheduling tasks.
- `sequential`: If `True`, run all tasks sequentially. Use when tasks have no dependencies.
- `tail_logs`: If `True`, display logs in real time.
- `direct`: If `True`, execute each task in the same process on your local machine. This mode doesn't support task or job groups.

```python
with exp:
    # Add all tasks
    exp.run(detach=True, sequential=False, tail_logs=True, direct=False)
```

(guides-check-status)=
## Check Experiment Status

Check the status of an experiment using the `status` method:

```python
exp.status()
```

This method displays the status of each task in the experiment. The following sample output comes from the experiment status in [hello_scripts.py](../../../examples/hello-world/hello_scripts.py):

```bash
Experiment Status for experiment_with_scripts_1730761155

Task 0: echo.sh
- Status: SUCCEEDED
- Executor: LocalExecutor
- Job id: echo.sh-zggz3tq0kpljs
- Local Directory: /home/your_user/.nemo_run/experiments/experiment_with_scripts/experiment_with_scripts_1730761155/echo.sh

Task 1: env_echo_
- Status: SUCCEEDED
- Executor: LocalExecutor
- Job id: env_echo_-f3fc3fbj1qjtc
- Local Directory: /home/your_user/.nemo_run/experiments/experiment_with_scripts/experiment_with_scripts_1730761155/env_echo_

Task 2: simple.add.add_object
- Status: RUNNING
- Executor: LocalExecutor
- Job id: simple.add.add_object-s1543tt3f7dcm
- Local Directory: /home/your_user/.nemo_run/experiments/experiment_with_scripts/experiment_with_scripts_1730761155/simple.add.add_object
```

(guides-cancel-task)=
## Cancel a Task

Cancel a task using the `cancel` method:

```python
exp.cancel("task_id")
```

(guides-view-logs)=
## View Logs

View the logs of a task using the `logs` method:

```python
exp.logs("task_id")
```

(guides-review-output)=
## Review Experiment Output

After you run an experiment, NeMo Run displays information on how to inspect and reproduce past experiments. This allows you to check logs, sync artifacts (in the future), cancel running tasks, and rerun an old experiment.

```python
# The experiment was run with the following tasks: ['echo.sh', 'env_echo_', 'simple.add.add_object']
# You can inspect and reconstruct this experiment at a later point in time using:
experiment = run.Experiment.from_id("experiment_with_scripts_1720556256")
experiment.status() # Gets the overall status
experiment.logs("echo.sh") # Gets the log for the provided task
experiment.cancel("echo.sh") # Cancels the provided task if still running
```

```bash
# You can inspect this experiment at a later point in time using the command-line interface as well:
nemorun experiment status experiment_with_scripts_1720556256
nemorun experiment logs experiment_with_scripts_1720556256 0
nemorun experiment cancel experiment_with_scripts_1720556256 0
```

These details are specific to each experiment.

See the [notebook](https://github.com/NVIDIA-NeMo/Run/blob/main/examples/hello-world/hello_experiments.ipynb) for more details and an interactive tutorial.
