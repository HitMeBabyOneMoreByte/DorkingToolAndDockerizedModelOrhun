import logging, sys, os
from pathlib import Path

DEFAULT_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR       = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_logger(name: str | None = None,
                 level: str | int = DEFAULT_LEVEL,
                 logfile: str | Path | None = LOG_DIR / "app.log"):
    """
    Her modül çağırdığında aynı logger’ı döndürür;  
    hem ekrana hem de dosyaya yazar.
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # Bir kere ayarlandıysa tekrar ekleme
        return logger

    logger.setLevel(level)

    fmt_console = logging.Formatter("[%(levelname)s] %(message)s")
    fmt_file    = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s %(funcName)s:%(lineno)d ― %(message)s"
    )

    # -- Terminal çıkışı
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt_console)
    logger.addHandler(sh)

    # -- Dosya çıkışı
    if logfile:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt_file)
        logger.addHandler(fh)

    # Root logger üzerinden üst kademeye gitmesin
    logger.propagate = False
    return logger
