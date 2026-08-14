import subprocess
import tempfile
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Path to run.bat, relative to this file
RUN_BAT = Path(__file__).parent / "java" / "run.bat"


@dataclass
class ConstraintResult:
    constraint: str
    context: str
    satisfied: bool
    severity: str        # OK | ERROR | PARSE_ERROR | RUNTIME_ERROR
    message: Optional[str] = None


@dataclass
class ValidationReport:
    success: bool
    results: list[ConstraintResult] = field(default_factory=list)
    raw_error: Optional[str] = None


def run_ocl_validation(
    ecore_path: str | Path,
    xmi_path:   str | Path,
    ocl_expressions: list[str],
    timeout: int = 60,
) -> ValidationReport:
    """
    Write OCL expressions to a temp file, invoke the Java validator
    via run.bat, parse the JSON result and return a ValidationReport.
    """
    if not RUN_BAT.exists():
        return ValidationReport(
            success=False,
            raw_error=f"run.bat not found at {RUN_BAT}",
        )

    # Write expressions to a temp .ocl file
    # delete=False required on Windows (Java must be able to open it)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".ocl", delete=False, encoding="utf-8"
    )
    stdout = ""
    try:
        tmp.write("\n".join(ocl_expressions))
        tmp.close()

        cmd = [
            "cmd.exe", "/c", str(RUN_BAT),
            "--ecore", str(ecore_path),
            "--xmi",   str(xmi_path),
            "--ocl",   tmp.name,
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            # Keep Windows from opening a new console window
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        # Temporary debug -- remove after fixing
        # print("=== STDOUT ===")
        # print(repr(proc.stdout))
        # print("=== STDERR ===")
        # print(repr(proc.stderr))
        # print("=== EXIT CODE ===")
        # print(proc.returncode)
        
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode != 0:
            return ValidationReport(
                success=False,
                raw_error=(
                    f"Java process exited with code {proc.returncode}.\n"
                    f"STDERR:\n{stderr}"
                ),
            )

        if not stdout:
            return ValidationReport(
                success=False,
                raw_error=f"No output from Java process.\nSTDERR:\n{stderr}",
            )

        data = json.loads(stdout)

        # Top-level fatal error emitted by Java
        if isinstance(data, dict) and "fatal_error" in data:
            return ValidationReport(
                success=False,
                raw_error=data["fatal_error"],
            )

        results = [
            ConstraintResult(
                constraint=r.get("constraint", ""),
                context=   r.get("context",    ""),
                satisfied= r.get("satisfied",  False),
                severity=  r.get("severity",   "UNKNOWN"),
                message=   r.get("message"),
            )
            for r in data
        ]
        return ValidationReport(success=True, results=results)

    except subprocess.TimeoutExpired:
        return ValidationReport(
            success=False,
            raw_error=f"Java process timed out after {timeout}s.",
        )
    except json.JSONDecodeError as exc:
        return ValidationReport(
            success=False,
            raw_error=(
                f"Could not parse Java output as JSON: {exc}\n"
                f"Raw stdout:\n{stdout}"
            ),
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        
def main():
    ecore_path = r".\test_models\family.ecore"
    xmi_path = r".\test_models\generated_model.model"
    raw = input("OCL expressions (separated by ';'): ").strip()
    ocl_expressions = [e.strip() for e in raw.split(";") if e.strip()]

    print(ocl_expressions)
    result = run_ocl_validation(
        ecore_path=Path(ecore_path),
        xmi_path=Path(xmi_path),
        ocl_expressions=ocl_expressions,
    )
    
    if not result.success:
        print("Validation failed:")
        print(result.raw_error)
        return

    if not result.results:
        print("No results returned.")
        return

    for r in result.results:
        print(f"\nConstraint: {r.constraint}")
        print(f"  satisfied: {r.satisfied}")
        print(f"  severity:  {r.severity}")
        print(f"  message:   {r.message}")

if __name__ == "__main__":
    main()