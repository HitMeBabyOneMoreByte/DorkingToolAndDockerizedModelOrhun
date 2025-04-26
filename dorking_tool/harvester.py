#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

import os, json, uuid
from pathlib import Path
from googleapiclient.discovery import build
from detectors import PDF_Detector, XLS_XLSX_Detector, DOC_DOCX_Detector, FileTypeDetector
from detectors.util import download_file, get_main_domain
from dorking_tool.logger_setup import setup_logger

logger = setup_logger(__name__)

VALID_TYPES = {
    'PDF':    ('PDF',      'pdf'),
    'XLS':    ('XLS_XLSX', 'xls'),
    'XLSX':   ('XLS_XLSX', 'xlsx'),
    'DOC':    ('DOC_DOCX', 'doc'),
    'DOCX':   ('DOC_DOCX', 'docx'),
}

def cse_search_until_k(domain: str, ext: str, k: int, api_key: str, cx: str, max_loops: int = 50):
    service = build("customsearch", "v1", developerKey=api_key)
    query = f"site:{domain} filetype:{ext} 11111111111..99999999999"
    collected, start = [], 1
    for _ in range(max_loops):
        resp = service.cse().list(cx=cx, q=query, num=10, start=start, safe='off').execute()
        items = resp.get("items", [])
        if not items: break
        for it in items:
            link = it.get("link","")
            if link.lower().endswith(f".{ext}") and link not in collected:
                collected.append(link)
                if len(collected)>=k: return collected
        next_page = resp.get("queries", {}).get("nextPage")
        if not next_page: break
        start = next_page[0]["startIndex"]
    return collected

DETECTORS = {
    'PDF':      PDF_Detector.process_pdfs,
    'XLS_XLSX': XLS_XLSX_Detector.process_xls_xlsx,
    'DOC_DOCX': DOC_DOCX_Detector.process_doc_docx,
}

def process_url(url: str, detector_group: str, out_files: Path, out_txt: Path):
    if detector_group != 'PDF':
        if FileTypeDetector.FileTypeDetector([],[]).determine_file_type(url) != detector_group:
            return None

    det = DETECTORS[detector_group]
    res = det([url]) \
          .get(get_main_domain(url), {}) \
          .get(detector_group, {}) \
          .get('POSITIVE', {})
    if res.get('TC_Count',0)==0:
        return None

    entry = {'url':url, 'count':res['TC_Count'], 'image':res.get('image','')}
    local = download_file(url, str(out_files))
    entry['file'] = local

    text_dir = out_txt/Path(local).stem
    text_dir.mkdir(parents=True, exist_ok=True)
    if detector_group=='PDF':
        import pdfplumber
        with pdfplumber.open(local) as pdf:
            for i, page in enumerate(pdf.pages,1):
                (text_dir/f"page_{i:02}.txt").write_text(page.extract_text() or "")
    else:
        raw=""
        try:
            import textract
            raw = textract.process(local).decode("utf-8","ignore")
        except: pass
        for idx in range(0, len(raw), 1000):
            (text_dir/f"part_{idx//1000+1:02}.txt").write_text(raw[idx:idx+1000])

    return entry

def harvest(domain: str, doc_type: str, k: int, outdir: str):
    api_key, cx = os.getenv("GOOGLE_CSE_KEY"), os.getenv("GOOGLE_CSE_CX")
    detector_group, cse_ext = VALID_TYPES[doc_type.upper()]

    run_id = uuid.uuid4().hex
    base = Path(outdir)/run_id
    out_files, out_txt = base/"files", base/"txt"
    base.mkdir(parents=True,exist_ok=True)
    out_files.mkdir(); out_txt.mkdir()

    raw_urls = cse_search_until_k(domain, cse_ext, k, api_key, cx)
    logger.info(f"{len(raw_urls)} ham URL bulundu")

    found, seen = [], set()
    for url in raw_urls:
        if url in seen: continue
        seen.add(url)
        o = process_url(url, detector_group, out_files, out_txt)
        if o:
            found.append(o)
            logger.info(f"[+] {len(found)}/{k} ➜ {url}")
        if len(found)>=k: break

    results_path = base/"results.json"
    with open(results_path,"w",encoding="utf-8") as f:
        json.dump(found, f, ensure_ascii=False, indent=2)

    return run_id, found
