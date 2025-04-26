# app.py
import os, shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse
from harvester import harvest

OUTPUT_BASE = os.getenv("OUTPUT_BASE","/output")

def cleanup_internal():
    # detectors.download_file vb. klasörleri temizle
    if os.path.isdir("downloads"):
        shutil.rmtree("downloads")
    os.makedirs("downloads", exist_ok=True)

class Req(BaseModel):
    domain: str
    type:   str
    k:      int

app = FastAPI()

@app.post("/harvest")
async def do_harvest(r: Req):
    td = r.type.upper()
    if td not in ("PDF","XLS","XLSX","DOC","DOCX"):
        raise HTTPException(400,"Invalid type")
    cleanup_internal()

    try:
        run_id, results = harvest(r.domain, td, r.k, OUTPUT_BASE)
    except Exception as e:
        raise HTTPException(500,str(e))

    return JSONResponse({
        "run_id": run_id,
        "output_dir": f"{OUTPUT_BASE}/{run_id}",
        "results": results
    })
