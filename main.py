import os
import json
import httpx
import pandas as pd
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

app = FastAPI(title="Parallax Autonomous Hybrid Diagnostics Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# 1. CLOUD DATABASE (SUPABASE) CONFIGURATION
# =====================================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://eupgsdxrylzglrtvcurf.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_LYbUHES1W0te5w7RssfllA_Py0qgRCX")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Connected to Supabase Cloud Database.")
except Exception as e:
    supabase = None
    print(f"Supabase connection warning: {e}")

def sync_history_to_cloud(vehicle_id: str, dtcs: list, diagnoses: list, summary: str):
    """Saves scan data to the cloud in the background without slowing down the UI."""
    if not supabase:
        return
    try:
        payload = {
            "vehicle_id": vehicle_id,
            "dtcs": dtcs,
            "top_severity": diagnoses[0]["severity"] if diagnoses else "LOW",
            "diagnoses_json": diagnoses,
            "llm_summary": summary,
            "created_at": datetime.utcnow().isoformat()
        }
        supabase.table("vehicle_history_logs").insert(payload).execute()
        print(f"[Cloud Sync] Logged record for {vehicle_id} to Supabase.")
    except Exception as e:
        print(f"[Cloud Sync Error]: {e}")

# =====================================================================
# 2. LOCAL DATASET INGESTION
# =====================================================================
EXCEL_FILE = "DTC Code Database.xlsx"
KNOWLEDGE_BASE = {}
SEVERITY_WEIGHTS = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}

def safe_float(val, default=0.0) -> float:
    if pd.isna(val):
        return default
    try:
        cleaned = str(val).replace('$', '').replace(',', '').strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return default

def load_database():
    if os.path.exists(EXCEL_FILE):
        try:
            df = pd.read_excel(EXCEL_FILE)
            for _, row in df.iterrows():
                code = str(row.get("dtc_code", "")).strip().upper()
                if not code or code == "NAN":
                    continue

                causes = [
                    str(row[f"root_cause_{i}"]).strip()
                    for i in range(1, 6)
                    if pd.notna(row.get(f"root_cause_{i}"))
                ]
                actions = [
                    str(row[f"action_{i}"]).strip()
                    for i in range(1, 6)
                    if pd.notna(row.get(f"action_{i}"))
                ]

                parts_min = safe_float(row.get("parts_cost_min_USD"), 0.0)
                parts_max = safe_float(row.get("parts_cost_max_USD"), 0.0)
                labor_min = safe_float(row.get("labor_cost_min_USD"), 0.0)
                labor_max = safe_float(row.get("labor_cost_max_USD"), 0.0)

                KNOWLEDGE_BASE[code] = {
                    "code": code,
                    "fault_name": str(row.get("fault_name", "Unknown Fault")),
                    "category": f"{row.get('category_name', 'General')} ({row.get('category_prefix', '')})",
                    "severity": str(row.get("severity_level", "MEDIUM")).strip().upper(),
                    "can_drive": bool(row.get("can_drive", True)),
                    "driver_directive": str(row.get("driver_directive", "Inspect vehicle.")),
                    "parts_cost_min": parts_min,
                    "parts_cost_max": parts_max,
                    "labor_cost_min": labor_min,
                    "labor_cost_max": labor_max,
                    "estimated_cost": f"${parts_min:.0f} – ${parts_max:.0f} (Parts) + ${labor_min:.0f} – ${labor_max:.0f} (Labor)",
                    "suggested_parts": str(row.get("parts_description", "General Inspection/Repair")),
                    "causes": causes,
                    "actions": actions,
                    "source": "Dataset (Excel)"
                }
            print(f"Loaded {len(KNOWLEDGE_BASE)} verified codes from {EXCEL_FILE}")
        except Exception as e:
            print(f"Error loading Excel database: {e}")
    else:
        print(f"Warning: {EXCEL_FILE} not found. Running dynamic mode.")

load_database()

# =====================================================================
# 3. LOCAL IN-MEMORY HISTORY TRACKER
# =====================================================================
VEHICLE_HISTORY = {}

def track_history(vehicle_id: str, dtcs: List[str]) -> dict:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_report = {}
    past_scans = VEHICLE_HISTORY.get(vehicle_id, [])

    for code in dtcs:
        past_occurrences = 0
        last_seen = None
        for scan in reversed(past_scans):
            if code in scan["dtcs"]:
                past_occurrences += 1
                if last_seen is None:
                    last_seen = scan["timestamp"]

        history_report[code] = {
            "seen_before": past_occurrences > 0,
            "occurrences": past_occurrences + 1,
            "last_recorded": last_seen if last_seen else "First scan"
        }

    if vehicle_id not in VEHICLE_HISTORY:
        VEHICLE_HISTORY[vehicle_id] = []
    VEHICLE_HISTORY[vehicle_id].append({"timestamp": now_str, "dtcs": dtcs})
    return history_report

# =====================================================================
# 4. UNIFIED LLM BRIDGE (GROQ CLOUD OR LOCAL OLLAMA)
# =====================================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

async def call_llm(prompt: str, expect_json: bool = False) -> Optional[str]:
    """Helper that queries Groq API if key is present, otherwise falls back to local Ollama."""
    # Option 1: Fast Cloud LLM via Groq (For public deployment)
    if GROQ_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
                body = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 300
                }
                if expect_json:
                    body["response_format"] = {"type": "json_object"}
                
                resp = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Groq API call failed: {e}")

    # Option 2: Local Ollama (For local laptop execution)
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            payload = {
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "keep_alive": -1,
                "options": {"num_predict": 250, "temperature": 0.2}
            }
            if expect_json:
                payload["format"] = "json"

            resp = await client.post("http://localhost:11434/api/generate", json=payload)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"Ollama local call failed: {e}")

    return None

# =====================================================================
# 5. DYNAMIC LLM INFERENCE (FOR NON-DATASET CODES)
# =====================================================================
async def infer_unknown_code_with_llm(code: str, make: str, model: str) -> dict:
    prompt = f"""You are an automotive diagnostic tool. Analyze unknown OBD-II code '{code}' for a {make} {model}.
Return ONLY valid JSON (no markdown formatting):
{{
  "fault_name": "Standard Fault Name",
  "category": "Powertrain / Chassis / Body / Network",
  "severity": "CRITICAL / HIGH / MEDIUM / LOW",
  "can_drive": true,
  "driver_directive": "Short action directive for driver",
  "parts_cost_min": 50.0,
  "parts_cost_max": 200.0,
  "labor_cost_min": 50.0,
  "labor_cost_max": 150.0,
  "suggested_parts": "Component to replace or inspect",
  "causes": ["Probable cause 1", "Probable cause 2"],
  "actions": ["Diagnostic step 1", "Diagnostic step 2"]
}}"""
    raw_response = await call_llm(prompt, expect_json=True)
    if raw_response:
        try:
            data = json.loads(raw_response)
            p_min = float(data.get("parts_cost_min", 50.0))
            p_max = float(data.get("parts_cost_max", 150.0))
            l_min = float(data.get("labor_cost_min", 50.0))
            l_max = float(data.get("labor_cost_max", 100.0))

            return {
                "code": code,
                "fault_name": str(data.get("fault_name", f"Diagnostic Code {code}")),
                "category": str(data.get("category", "General")),
                "severity": str(data.get("severity", "MEDIUM")).upper(),
                "can_drive": bool(data.get("can_drive", True)),
                "driver_directive": str(data.get("driver_directive", "Inspect at authorized workshop.")),
                "parts_cost_min": p_min,
                "parts_cost_max": p_max,
                "labor_cost_min": l_min,
                "labor_cost_max": l_max,
                "estimated_cost": f"${p_min:.0f} – ${p_max:.0f} (Parts) + ${l_min:.0f} – ${l_max:.0f} (Labor)",
                "suggested_parts": str(data.get("suggested_parts", "Component Inspection")),
                "causes": data.get("causes", ["Electrical variance or sensor issue"]),
                "actions": data.get("actions", ["Perform scan tool diagnostic check"]),
                "source": "LLM Inference"
            }
        except Exception as e:
            print(f"Failed to parse LLM JSON for {code}: {e}")

    return {
        "code": code,
        "fault_name": f"Fault Code {code}",
        "category": "Powertrain",
        "severity": "MEDIUM",
        "can_drive": True,
        "driver_directive": "Inspect at authorized workshop.",
        "parts_cost_min": 50.0,
        "parts_cost_max": 150.0,
        "labor_cost_min": 50.0,
        "labor_cost_max": 100.0,
        "estimated_cost": "$50 – $150 (Parts) + $50 – $100 (Labor)",
        "suggested_parts": "Diagnostic Inspection",
        "causes": ["Circuit or sensor irregularity"],
        "actions": ["Verify pending freeze-frame data"],
        "source": "Default Fallback"
    }

# =====================================================================
# 6. USER-FRIENDLY LLM SUMMARY
# =====================================================================
async def generate_user_friendly_summary(vehicle_label: str, symptoms: str, diagnoses: list, total_costs: dict) -> str:
    if not diagnoses:
        return "No diagnostic trouble codes detected."

    fault_summary_lines = [
        f"- {d['fault_name']} ({d['code']}) | Severity: {d['severity']} | Drivable: {d['can_drive']}"
        for d in diagnoses
    ]
    fault_context = "\n".join(fault_summary_lines)

    prompt = f"""You are a friendly, highly skilled automotive expert talking to a non-technical car owner.
Vehicle: {vehicle_label} | Symptoms: {symptoms or 'None reported'}
Diagnosed Faults:
{fault_context}
Total Estimated Cost: ${total_costs['grand_total_min']:.0f} - ${total_costs['grand_total_max']:.0f}

Explain what is going on in simple terms using this exact 3-bullet format:
• What is happening: Explain the core problem in plain English and link it to driver symptoms.
• Is it safe to drive?: Direct verdict and reason.
• What to do next & budget: State the estimated cost range (${total_costs['grand_total_min']:.0f}-${total_costs['grand_total_max']:.0f}) and the exact practical next step."""

    res = await call_llm(prompt, expect_json=False)
    if res:
        return res

    top = diagnoses[0]
    return (
        f"• What is happening: Primary issue detected is {top['fault_name']} ({top['code']}).\n"
        f"• Is it safe to drive?: {'Vehicle may be driven cautiously.' if top.get('can_drive', True) else 'DO NOT DRIVE until inspected.'}\n"
        f"• What to do next & budget: Estimated total cost is ${total_costs['grand_total_min']:.0f} – ${total_costs['grand_total_max']:.0f}."
    )

# =====================================================================
# 7. API ENDPOINTS
# =====================================================================
class DiagnosticRequest(BaseModel):
    make: Optional[str] = "Generic"
    model: Optional[str] = "Vehicle"
    dtcs: List[str]
    symptoms: Optional[str] = ""

@app.get("/health")
def health():
    return {"status": "Parallax Engine is operational."}

@app.post("/diagnose")
async def diagnose(req: DiagnosticRequest, background_tasks: BackgroundTasks):
    seen = set()
    clean_dtcs = [c for d in req.dtcs if (c := d.strip().upper().replace(",", "").replace(".", "")) and not (c in seen or seen.add(c))]

    if not clean_dtcs:
        return {"error": "No DTC codes provided."}

    vehicle_id = f"{req.make} {req.model}".strip()
    history_report = track_history(vehicle_id, clean_dtcs)

    diagnoses = []
    tot_parts_min = 0.0
    tot_parts_max = 0.0
    tot_labor_min = 0.0
    tot_labor_max = 0.0

    for code in clean_dtcs:
        if code in KNOWLEDGE_BASE:
            data = KNOWLEDGE_BASE[code]
        else:
            data = await infer_unknown_code_with_llm(code, req.make, req.model)

        tot_parts_min += data["parts_cost_min"]
        tot_parts_max += data["parts_cost_max"]
        tot_labor_min += data["labor_cost_min"]
        tot_labor_max += data["labor_cost_max"]

        diagnoses.append({
            "code": data["code"],
            "category": data["category"],
            "fault_name": data["fault_name"],
            "severity": data["severity"],
            "can_drive": data["can_drive"],
            "driver_directive": data["driver_directive"],
            "estimated_cost": data["estimated_cost"],
            "suggested_repair": data["suggested_parts"],
            "possible_causes": data["causes"],
            "recommended_actions": data["actions"],
            "source": data.get("source", "Dataset"),
            "history": history_report.get(code, {})
        })

    diagnoses.sort(key=lambda x: SEVERITY_WEIGHTS.get(x["severity"], 0), reverse=True)
    for idx, item in enumerate(diagnoses, 1):
        item["priority_rank"] = idx

    cost_summary = {
        "parts_min": tot_parts_min,
        "parts_max": tot_parts_max,
        "labor_min": tot_labor_min,
        "labor_max": tot_labor_max,
        "grand_total_min": tot_parts_min + tot_labor_min,
        "grand_total_max": tot_parts_max + tot_labor_max
    }

    ai_synthesis = await generate_user_friendly_summary(vehicle_id, req.symptoms, diagnoses, cost_summary)

    # Schedule background persistence to Supabase
    background_tasks.add_task(
        sync_history_to_cloud,
        vehicle_id=vehicle_id,
        dtcs=clean_dtcs,
        diagnoses=diagnoses,
        summary=ai_synthesis
    )

    return {
        "vehicle": {"make": req.make, "model": req.model},
        "symptoms": req.symptoms,
        "cost_summary": cost_summary,
        "diagnoses": diagnoses,
        "combined_summary": {
            "most_critical_code": diagnoses[0]["code"] if diagnoses else None,
            "priority_order": [d["code"] for d in diagnoses],
            "ai_interpretation": ai_synthesis
        }
    }