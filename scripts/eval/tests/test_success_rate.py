import importlib.util
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "eval" / "success_rate.py"


def _load_module():
	os.chdir(REPO_ROOT)  # module does sys.path.append(os.getcwd())
	spec = importlib.util.spec_from_file_location("success_rate", MODULE_PATH)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def _make_tree(root: Path, layout: dict):
	# layout: {task_name: [1, 0, ...]} -> one episode dir per entry with a
	# success_/failure_ marker file.
	for task, outcomes in layout.items():
		for i, ok in enumerate(outcomes):
			ep = root / task / f"episode{i}"
			ep.mkdir(parents=True)
			prefix = "success_" if ok else "failure_"
			(ep / f"{prefix}0.txt").write_text("x")


def test_basic_success_rate():
	mod = _load_module()
	with tempfile.TemporaryDirectory() as d:
		root = Path(d) / "exp"
		_make_tree(root, {"open_drawer": [1, 1, 0, 0], "push_buttons": [1, 1, 1, 1]})
		(means, stds), results = mod.get_success_rate(str(root))
		assert abs(means["open_drawer"] - 0.5) < 1e-9
		assert abs(means["push_buttons"] - 1.0) < 1e-9


def test_exclude_suboptimal_and_train_tasks():
	mod = _load_module()
	with tempfile.TemporaryDirectory() as d:
		root = Path(d) / "exp"
		_make_tree(root, {
			"open_drawer": [1, 0],
			"stack_blocks": [1, 1],   # suboptimal -> excluded
			"close_jar": [0, 0],       # train task -> excluded
		})
		(means, _), _ = mod.get_success_rate(
			str(root), exclude_suboptimal_tasks=True, exclude_train_tasks=True)
		assert set(means.keys()) == {"open_drawer"}

		# Each flag should only filter its own task set, not the other's.
		(means, _), _ = mod.get_success_rate(
			str(root), exclude_suboptimal_tasks=True, exclude_train_tasks=False)
		assert set(means.keys()) == {"open_drawer", "close_jar"}

		(means, _), _ = mod.get_success_rate(
			str(root), exclude_suboptimal_tasks=False, exclude_train_tasks=True)
		assert set(means.keys()) == {"open_drawer", "stack_blocks"}


def test_ragged_task_coverage_raises():
	mod = _load_module()
	# two "runs" with different task sets -> aggregation must refuse, not guess
	all_res = {"open_drawer": [1.0, 0.5], "turn_tap": [1.0]}
	try:
		mod.print_results(all_res)
	except ValueError as e:
		assert "different task sets" in str(e)
		return
	raise AssertionError("print_results should refuse ragged task coverage")


if __name__ == "__main__":
	test_basic_success_rate()
	test_exclude_suboptimal_and_train_tasks()
	test_ragged_task_coverage_raises()
	print("OK")
