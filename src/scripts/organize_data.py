# scripts/organize_data.py
import argparse, shutil
from pathlib import Path

def copy_dir(src: Path, dst: Path):
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    # copy contenido (merge-safe)
    for p in src.glob("*"):
        target = dst / p.name
        if p.is_dir():
            shutil.copy(str(p), str(target))
        else:
            if target.exists():
                target.unlink()
            shutil.copy(str(p), str(target))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Carpeta OUT del scraper (ej: src/.../data/tmp/out)")
    ap.add_argument("--clean-dir", required=True, help="Destino base para clean (tendrá subcarpetas chunks/ y clean_text/)")
    ap.add_argument("--raw-dir", required=True, help="Destino base para raw (tendrá subcarpeta raw_html/)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    clean_base = Path(args.clean_dir)
    raw_base = Path(args.raw_dir)

    # Fuentes esperadas desde el scraper
    raw_html_src = in_dir / "raw_html"
    clean_text_src = in_dir / "clean_text"
    chunks_src = in_dir / "chunks"

    # Destinos canónicos
    raw_html_dst = raw_base / "raw_html"
    clean_text_dst = clean_base / "clean_text"
    chunks_dst = clean_base / "chunks"

    copy_dir(raw_html_src, raw_html_dst)
    copy_dir(clean_text_src, clean_text_dst)
    copy_dir(chunks_src, chunks_dst)

    # Limpieza de carpeta temporal
    try:
        shutil.rmtree(in_dir)
    except Exception:
        pass

    print("[OK] Datos organizados.")
    print(f"  RAW  -> {raw_html_dst}")
    print(f"  TXT  -> {clean_text_dst}")
    print(f"  JSON -> {chunks_dst}")
