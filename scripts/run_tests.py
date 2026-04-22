"""
Lightweight test runner – executes every ``test_*`` method in the
``tests/`` package without requiring pytest to be installed.

For environments where ``pip install pytest`` is possible, the
preferred command is still::

    pytest -v

This runner simply provides a pure-Python fallback so that the
bundled zip can ship with a test log even on machines where the full
requirements haven't been installed yet.
"""
from __future__ import annotations

import inspect
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

# Make ``src.*`` importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Minimal pytest shim – enough for the assertions our tests actually use.
# ---------------------------------------------------------------------------
class _Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-6):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, other):
        try:
            return abs(other - self.expected) <= max(
                self.abs, self.rel * abs(self.expected)
            )
        except Exception:
            return False

    def __repr__(self):
        return f"approx({self.expected})"


def _approx(expected, rel=1e-6, abs=1e-6):  # mirrors pytest.approx
    return _Approx(expected, rel=rel, abs=abs)


@contextmanager
def _raises(exc_type):
    try:
        yield
    except exc_type:
        return
    except BaseException as exc:  # wrong type raised
        raise AssertionError(
            f"Expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"Did not raise {exc_type.__name__}")


def _fixture(*dargs, **dkwargs):
    """Decorator shim – fixtures are resolved manually below."""
    def _wrap(fn):
        fn._is_fixture = True
        return fn
    # Allow use as @fixture or @fixture()
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        return _wrap(dargs[0])
    return _wrap


# Build a fake ``pytest`` module so the test files import cleanly.
_pytest_module = ModuleType("pytest")
_pytest_module.approx = _approx
_pytest_module.raises = _raises
_pytest_module.fixture = _fixture
sys.modules["pytest"] = _pytest_module


# ---------------------------------------------------------------------------
# Test collection + execution
# ---------------------------------------------------------------------------
import importlib  # noqa: E402


TEST_FILES = [
    "tests.test_features",
    "tests.test_preprocessing",
    "tests.test_model",
]


def _resolve_fixture(instance, name):
    """If a test method declares a fixture argument, look it up."""
    cls = type(instance)
    if hasattr(cls, name):
        func = getattr(cls, name)
        if getattr(func, "_is_fixture", False):
            return func(instance)
    return None


def _run_method(cls, method_name):
    """Run a single test method; return (status, message)."""
    try:
        instance = cls()
    except Exception:
        return "ERROR", traceback.format_exc()

    method = getattr(instance, method_name)
    sig = inspect.signature(method)
    kwargs = {}
    for pname in list(sig.parameters)[: len(sig.parameters)]:
        if pname == "self":
            continue
        kwargs[pname] = _resolve_fixture(instance, pname)

    try:
        method(**kwargs)
        return "PASS", ""
    except AssertionError as ae:
        return "FAIL", "".join(traceback.format_exception_only(type(ae), ae))
    except Exception:
        return "ERROR", traceback.format_exc()


def _collect_tests(module):
    """Yield (class, test_method_name) pairs from a test module."""
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if not cls.__name__.startswith("Test"):
            continue
        if cls.__module__ != module.__name__:
            continue
        for attr_name, attr in inspect.getmembers(cls):
            if attr_name.startswith("test_") and callable(attr):
                yield cls, attr_name


def main() -> int:
    started = time.time()
    summary = {"pass": 0, "fail": 0, "error": 0, "skipped": 0}
    lines: list[str] = []

    lines.append("=" * 68)
    lines.append("  Phishing URL Detection – Unit Test Run")
    lines.append("=" * 68)
    lines.append("")

    for mod_path in TEST_FILES:
        lines.append(f"[module] {mod_path}")
        try:
            mod = importlib.import_module(mod_path)
        except ImportError as e:
            # Expected for test_model when sklearn isn't installed.
            summary["skipped"] += 1
            lines.append(f"   SKIP   (ImportError: {e})")
            lines.append("")
            continue

        for cls, method_name in _collect_tests(mod):
            status, msg = _run_method(cls, method_name)
            label = f"{cls.__name__}::{method_name}"
            if status == "PASS":
                summary["pass"] += 1
                lines.append(f"   PASS   {label}")
            elif status == "FAIL":
                summary["fail"] += 1
                lines.append(f"   FAIL   {label}")
                for line in msg.splitlines():
                    lines.append(f"          {line}")
            else:
                summary["error"] += 1
                lines.append(f"   ERROR  {label}")
                for line in msg.splitlines():
                    lines.append(f"          {line}")
        lines.append("")

    duration = time.time() - started
    total_run = summary["pass"] + summary["fail"] + summary["error"]
    lines.append("-" * 68)
    lines.append(
        f"Summary: {summary['pass']}/{total_run} passed, "
        f"{summary['fail']} failed, {summary['error']} errors, "
        f"{summary['skipped']} modules skipped (took {duration:.2f}s)"
    )
    lines.append("-" * 68)

    report = "\n".join(lines)
    print(report)

    out_path = PROJECT_ROOT / "results" / "test_results.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\nSaved report to {out_path.relative_to(PROJECT_ROOT)}")

    return 0 if (summary["fail"] == 0 and summary["error"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
