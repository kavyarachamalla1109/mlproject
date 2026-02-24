"""Deployment bundle creation utilities."""

from __future__ import annotations

from pathlib import Path
import shutil


def create_inference_bundle(output_dir: str) -> str:
    """Create a zip bundle with FastAPI app, model, and run instructions."""
    out = Path(output_dir)
    bundle_dir = out / "inference_bundle"

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        Path("main.py"),
        Path("requirements.txt"),
        Path("README.md"),
        Path("artifacts") / "agribot_model.pkl",
    ]

    for file_path in files_to_copy:
        if file_path.exists():
            target = bundle_dir / file_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target)

    templates_src = Path("templates")
    if templates_src.exists():
        shutil.copytree(templates_src, bundle_dir / "templates", dirs_exist_ok=True)

    quickstart = bundle_dir / "QUICKSTART.md"
    quickstart.write_text(
        """# AgriBot Inference Bundle Quickstart

## 1) Install dependencies
```bash
python -m pip install -r requirements.txt
```

## 2) Start API + Web UI
```bash
python main.py
```

The app runs at `http://127.0.0.1:8000`.

## 3) Predict
- Web form: open `/`
- API docs: open `/docs`
- JSON endpoint: `POST /predict-json`

Model path expected by app:
- `artifacts/agribot_model.pkl`
""",
        encoding="utf-8",
    )

    zip_base = out / "agribot_inference_bundle"
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=bundle_dir)
    return zip_path
