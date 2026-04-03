"""
Run ALL insurance-whittaker tests (including new expanded coverage) on Databricks.
"""

import os
import sys
import time
import base64

env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

base = "/home/ralph/repos/insurance-whittaker"


def read_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


src_files = {
    "__init__.py": read_b64(f"{base}/src/insurance_whittaker/__init__.py"),
    "_utils.py": read_b64(f"{base}/src/insurance_whittaker/_utils.py"),
    "_smoother2d.py": read_b64(f"{base}/src/insurance_whittaker/_smoother2d.py"),
    "selection.py": read_b64(f"{base}/src/insurance_whittaker/selection.py"),
    "smoother.py": read_b64(f"{base}/src/insurance_whittaker/smoother.py"),
    "smoother2d.py": read_b64(f"{base}/src/insurance_whittaker/smoother2d.py"),
    "glm.py": read_b64(f"{base}/src/insurance_whittaker/glm.py"),
    "plots.py": read_b64(f"{base}/src/insurance_whittaker/plots.py"),
    "py.typed": read_b64(f"{base}/src/insurance_whittaker/py.typed"),
}

test_dir_path = f"{base}/tests"
test_files = {}
import os as _os
for fname in _os.listdir(test_dir_path):
    if fname.endswith(".py"):
        test_files[fname] = read_b64(f"{test_dir_path}/{fname}")

lines = [
    "import subprocess, sys, os, base64, tempfile",
    "from pathlib import Path",
    "",
    "subprocess.run([sys.executable, '-m', 'pip', 'install',",
    "    'numpy', 'scipy', 'polars', 'pytest', '--quiet'], check=True)",
    "",
    "tmpdir = Path(tempfile.mkdtemp())",
    "pkg_dir = tmpdir / 'insurance_whittaker'",
    "pkg_dir.mkdir()",
    "test_dir = tmpdir / 'iw_tests'",
    "test_dir.mkdir()",
    "(test_dir / '__init__.py').write_text('')",
    "",
]

for fname, b64content in src_files.items():
    lines.append(f"(pkg_dir / '{fname}').write_bytes(base64.b64decode('{b64content}'))")

lines.append("")

for fname, b64content in test_files.items():
    lines.append(f"(test_dir / '{fname}').write_bytes(base64.b64decode('{b64content}'))")

lines.extend([
    "",
    "sys.path.insert(0, str(tmpdir))",
    "",
    "env = dict(os.environ)",
    "env['PYTHONPATH'] = str(tmpdir) + ':' + env.get('PYTHONPATH', '')",
    "result = subprocess.run(",
    "    [sys.executable, '-m', 'pytest', str(test_dir),",
    "     '-v', '--tb=short', '-p', 'no:cacheprovider', '-q'],",
    "    capture_output=True, text=True, env=env,",
    ")",
    "output = (result.stdout or '') + '\\n' + (result.stderr or '')",
    "exit_msg = output[-5000:] + f'\\n\\nEXIT_CODE={result.returncode}'",
    "dbutils.notebook.exit(exit_msg)",
])

notebook_source = "\n".join(lines)
notebook_path = "/Workspace/Shared/insurance-whittaker-expanded-runner"
nb_b64 = base64.b64encode(notebook_source.encode("utf-8")).decode("utf-8")

print(f"Uploading notebook ({len(notebook_source)//1024}KB)...")
w.workspace.import_(
    path=notebook_path,
    content=nb_b64,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
)
print("Uploaded.")

print("Submitting test job...")
run_waiter = w.jobs.submit(
    run_name="insurance-whittaker-expanded-tests",
    tasks=[
        jobs.SubmitTask(
            task_key="run-tests",
            notebook_task=jobs.NotebookTask(notebook_path=notebook_path),
        )
    ],
)

run_id = run_waiter.run_id
print(f"Run ID: {run_id}")

while True:
    run_state = w.jobs.get_run(run_id=run_id)
    life_cycle = str(run_state.state.life_cycle_state)
    print(f"  Status: {life_cycle}")
    if any(s in life_cycle for s in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]):
        break
    time.sleep(20)

result_state = str(run_state.state.result_state)
print(f"\nFinal result: {result_state}")

nb_output = ""
for task in (run_state.tasks or []):
    try:
        output = w.jobs.get_run_output(run_id=task.run_id)
        if output.notebook_output and output.notebook_output.result:
            nb_output = output.notebook_output.result
            print("\n--- Test output ---")
            print(nb_output)
        if output.error:
            print("\n--- Error ---")
            print(output.error)
        if output.error_trace:
            print("\n--- Error trace ---")
            print(output.error_trace[-5000:])
    except Exception as e:
        print(f"Could not get output: {e}")

tests_passed = "SUCCESS" in result_state or "EXIT_CODE=0" in nb_output
if tests_passed:
    print("\nAll tests passed.")
    sys.exit(0)
else:
    print(f"\nTests failed. State: {result_state}")
    sys.exit(1)
