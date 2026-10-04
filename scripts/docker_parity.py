"""Docker parity check: container /predict vs in-process predict on the same backend.

    .\\.venv-serve\\Scripts\\python.exe scripts/docker_parity.py --image intent-router:v1 \
        --serve-dir serve_model --out results/serving/docker_parity_v1.json

Starts ONE named container (CPU, ephemeral host port), waits for /health, sends the 20 synthetic
strings below (hand-written; no dataset text), compares label / confidence / ood_score / abstained /
top3 with an in-process Router on the same backend, writes the JSON, and removes only the
container it created. Requires `docker` on PATH and the repo's src/ importable (PYTHONPATH=src).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

# Deterministic, synthetic, logistics-style messages: en / es / fr / de / zh, 4 each.
SYNTHETIC_TEXTS: tuple[str, ...] = (
    "Where is container MSKU1234567 right now?",
    "Please book a dock slot for the 14:00 truck at gate 3 tomorrow.",
    "I want to cancel purchase order 88231 and get a refund.",
    "How long will the port strike delay my pallets from Rotterdam?",
    "Show me last month's average dwell time per yard zone.",
    "Hola, necesito saber en que estado esta mi envio a Valencia.",
    "Quiero reprogramar la cita de descarga para el jueves por la manana.",
    "Cual es la politica de devoluciones para mercancia danada?",
    "El camion lleva tres horas esperando en la puerta, que pasa?",
    "Buenos dias, que tal el dia de hoy?",
    "Bonjour, ou en est ma livraison prevue pour Lyon ce vendredi ?",
    "Pouvez-vous annuler la commande 5521 et rembourser l'acompte ?",
    "Un retard est-il prevu a cause de la greve des camionneurs ?",
    "Je voudrais parler a un conseiller du service client, s'il vous plait.",
    "Wo ist meine Sendung aus Hamburg gerade?",
    "Bitte buchen Sie einen Entladetermin fuer Montag um neun Uhr.",
    "Wie viele Anhaenger stehen aktuell auf dem Hof?",
    "Kannst du die Rechnung als PDF auslesen und die Positionen extrahieren?",
    "我的货物现在到哪里了？",
    "请帮我预约明天上午的卸货时间。",
)

assert len(SYNTHETIC_TEXTS) == 20


def _run(argv: list[str], timeout: int = 600) -> str:
    return subprocess.run(  # noqa: S603 - fixed argv
        argv, capture_output=True, text=True, timeout=timeout, check=True
    ).stdout.strip()


def _post(url: str, text: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - localhost only
        return json.loads(r.read())


def compare(remote: dict[str, Any], local: dict[str, Any]) -> dict[str, Any]:
    """Per-string comparison of one /predict body vs one in-process prediction."""
    top_remote = [t["label"] for t in remote["top3"]]
    top_local = [t["label"] for t in local["top3"]]
    return {
        "label_equal": remote["label"] == local["label"],
        "abstained_equal": remote["abstained"] == local["abstained"],
        "top3_labels_equal": top_remote == top_local,
        "abs_diff_confidence": abs(remote["confidence"] - local["confidence"]),
        "abs_diff_ood_score": abs(remote["ood_score"] - local["ood_score"]),
        "max_abs_diff_top3_prob": max(
            abs(a["prob"] - b["prob"]) for a, b in zip(remote["top3"], local["top3"], strict=True)
        ),
    }


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", default="intent-router:v1")
    ap.add_argument("--serve-dir", type=Path, default=Path("serve_model"))
    ap.add_argument("--backend", default="onnx_fp32")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prob-tol", type=float, default=1e-4)
    ap.add_argument("--ready-timeout", type=float, default=300.0)
    a = ap.parse_args()

    from intent_router.predict import Router  # after argparse so --help works without deps

    name = f"intent-router-parity-{uuid.uuid4().hex[:8]}"
    image_id = _run(["docker", "image", "inspect", a.image, "--format", "{{.Id}}"])
    image_size = int(_run(["docker", "image", "inspect", a.image, "--format", "{{.Size}}"]))
    t0 = time.perf_counter()
    cid = _run(
        ["docker", "run", "-d", "--name", name, "-p", "127.0.0.1::8000",
         "-e", f"ROUTER_BACKEND={a.backend}", a.image]
    )  # fmt: skip
    try:
        port = _run(["docker", "port", name, "8000/tcp"]).splitlines()[0].rsplit(":", 1)[1]
        base = f"http://127.0.0.1:{port}"
        ready_s: float | None = None
        health: dict[str, Any] = {}
        while time.perf_counter() - t0 < a.ready_timeout:
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=3) as r:  # noqa: S310
                    if r.status == 200:
                        health = json.loads(r.read())
                        ready_s = time.perf_counter() - t0
                        break
            except (urllib.error.URLError, OSError, ConnectionError):
                pass
            time.sleep(0.5)
        if ready_s is None:
            logs = _run(["docker", "logs", "--tail", "40", name])
            sys.exit(f"container not ready in {a.ready_timeout}s; logs:\n{logs}")
        router = Router.load(a.serve_dir, a.backend)
        rows = []
        for text in SYNTHETIC_TEXTS:
            remote = _post(f"{base}/predict", text)
            local = router.predict(text)
            rows.append(
                {
                    "text": text,
                    "remote": {k: remote[k] for k in ("label", "confidence", "ood_score",
                                                      "abstained", "top3")},
                    "local": local,
                    **compare(remote, local),
                }
            )  # fmt: skip
        summary = {
            "n": len(rows),
            "all_labels_equal": all(r["label_equal"] for r in rows),
            "all_abstained_equal": all(r["abstained_equal"] for r in rows),
            "all_top3_labels_equal": all(r["top3_labels_equal"] for r in rows),
            "max_abs_diff_confidence": max(r["abs_diff_confidence"] for r in rows),
            "max_abs_diff_ood_score": max(r["abs_diff_ood_score"] for r in rows),
            "max_abs_diff_top3_prob": max(r["max_abs_diff_top3_prob"] for r in rows),
            "prob_tolerance": a.prob_tol,
        }
        summary["prob_within_tolerance"] = summary["max_abs_diff_top3_prob"] <= a.prob_tol
        out = {
            "image": a.image,
            "image_id": image_id,
            "image_size_bytes": image_size,
            "container_id": cid,
            "backend": a.backend,
            "container_start_to_ready_s": ready_s,
            "health": health,
            "in_process_model_fingerprint": router.cfg.model_fingerprint,
            "summary": summary,
            "strings": rows,
        }
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)  # noqa: S603, S607


if __name__ == "__main__":
    main()
