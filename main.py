import os
import re
import json
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import httpx
from supabase import create_client, Client

app = FastAPI(title="Autonomous Vehicle Error Detection & Diagnostic Core")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# 1. SUPABASE CLOUD DATABASE CONFIGURATION
# =====================================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://eupgsdxrylzglrtvcurf.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_LYbUHES1W0te5w7RssfllA_Py0qgRCX")

supabase: Optional[Client] = None
try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Connected to Supabase Cloud Database.")
except Exception as e:
    print(f"Supabase connection warning: {e}")

# =====================================================================
# 2. LOCAL DATASET INGESTION & FALLBACKS
# =====================================================================
DATASET_PATH = "DTC Code Database.xlsx"
LOCAL_DTC_DB: Dict[str, Dict[str, Any]] = {}

FALLBACK_DTC_DB = {
    "P0300": {
        "fault_name": "Random/Multiple Cylinder Misfire Detected",
        "category": "Powertrain",
        "severity": "High",
        "possible_causes": ["Faulty spark plugs/coils", "Vacuum leak", "Low fuel pressure", "EGR valve stuck open"],
        "recommended_actions": ["Inspect spark plugs & coils", "Perform fuel pressure test", "Check vacuum lines"],
        "estimated_cost": "$150 - $600",
        "can_drive": False,
        "driver_directive": "Drive minimally. Unburnt fuel can overheat and destroy the catalytic converter."
    },
    "P0301": {
        "fault_name": "Cylinder 1 Misfire Detected",
        "category": "Powertrain",
        "severity": "High",
        "possible_causes": ["Faulty Cylinder 1 ignition coil", "Fouled spark plug", "Clogged fuel injector", "Low compression"],
        "recommended_actions": ["Swap ignition coil 1 with 2 to isolate", "Replace spark plugs", "Test injector pulse"],
        "estimated_cost": "$100 - $400",
        "can_drive": False,
        "driver_directive": "Do not drive under heavy load. Inspect cylinder 1 ignition system immediately."
    },
    "P0171": {
        "fault_name": "System Too Lean (Bank 1)",
        "category": "Powertrain / Fuel Metering",
        "severity": "Medium",
        "possible_causes": ["Dirty/faulty MAF sensor", "Intake vacuum leak", "Weak fuel pump", "Clogged fuel filter"],
        "recommended_actions": ["Clean MAF sensor", "Smoke test intake for vacuum leaks", "Check fuel trim values on OBD scanner"],
        "estimated_cost": "$80 - $350",
        "can_drive": True,
        "driver_directive": "Vehicle may be driven gently to repair facility. Avoid aggressive acceleration."
    },
    "P0420": {
        "fault_name": "Catalyst System Efficiency Below Threshold (Bank 1)",
        "category": "Emissions",
        "severity": "Low",
        "possible_causes": ["Degraded catalytic converter", "Faulty downstream O2 sensor", "Exhaust leak before catalyst"],
        "recommended_actions": ["Inspect exhaust for leaks", "Check O2 sensor live waveforms", "Replace catalytic converter if confirmed"],
        "estimated_cost": "$400 - $1800",
        "can_drive": True,
        "driver_directive": "Safe for standard driving, but vehicle will fail emissions/inspection tests."
    },
    "B0001": {
        "fault_name": "Driver Frontal Stage 1 Deployment Control",
        "category": "Body / SRS Airbag",
        "severity": "Critical",
        "possible_causes": ["Faulty clockspring", "Damaged airbag wiring harness", "Defective SRS module"],
        "recommended_actions": ["Scan SRS module with dedicated diagnostic tool", "Inspect clockspring continuity", "Do not probe with multimeter directly"],
        "estimated_cost": "$250 - $900",
        "can_drive": False,
        "driver_directive": "SAFETY HAZARD: Airbag deployment system disabled. Do not transport passengers until serviced."
    },
    "C0035": {
        "fault_name": "Left Front Wheel Speed Sensor Circuit",
        "category": "Chassis / ABS",
        "severity": "Medium",
        "possible_causes": ["Damaged ABS wheel speed sensor", "Broken tone ring", "Wiring harness corroded"],
        "recommended_actions": ["Inspect front left wheel sensor wiring", "Check live wheel speed data on scanner", "Clean sensor head"],
        "estimated_cost": "$120 - $350",
        "can_drive": True,
        "driver_directive": "Vehicle drives normally under dry conditions, but ABS and Traction Control are offline."
    },
    "U0100": {
        "fault_name": "Lost Communication with ECM/PCM 'A'",
        "category": "Network / Communication",
        "severity": "Critical",
        "possible_causes": ["CAN bus wiring short/open", "Blown ECM fuse", "Failing main engine control unit", "Corroded chassis ground"],
        "recommended_actions": ["Inspect ECM power & ground circuits", "Check 120-ohm CAN bus terminating resistance", "Inspect battery voltage under cranking"],
        "estimated_cost": "$200 - $1200",
        "can_drive": False,
        "driver_directive": "Risk of sudden engine shutdown or no-crank. Tow vehicle to an electrical diagnostic specialist."
    }
}

def load_excel_database():
    global LOCAL_DTC_DB
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_excel(DATASET_PATH)
            for _, row in df.iterrows():
                code = str(row.get('DTC', '') or row.get('Code', '')).strip().upper()
                if not code:
                    continue
                LOCAL_DTC_DB[code] = {
                    "fault_name": str(row.get('Description', '') or row.get('Fault Name', 'Generic DTC Fault')),
                    "category": str(row.get('Category', 'Powertrain')),
                    "severity": str(row.get('Severity', 'Medium')),
                    "possible_causes": [c.strip() for c in str(row.get('Possible Causes', '')).split(';') if c.strip()],
                    "recommended_actions": [a.strip() for a in str(row.get('Recommended Actions', '')).split(';') if a.strip()],
                    "estimated_cost": str(row.get('Estimated Cost', '$100 - $400')),
                    "can_drive": bool(row.get('Can Drive', True)),
                    "driver_directive": str(row.get('Driver Directive', 'Inspect vehicle with OBD-II diagnostic equipment.'))
                }
            print(f"Loaded {len(LOCAL_DTC_DB)} codes from {DATASET_PATH}.")
        except Exception as e:
            print(f"Error reading Excel database ({e}). Falling back to internal dictionary.")
            LOCAL_DTC_DB = FALLBACK_DTC_DB
    else:
        print(f"No Excel file found at {DATASET_PATH}. Using internal dictionary.")
        LOCAL_DTC_DB = FALLBACK_DTC_DB

load_excel_database()

# =====================================================================
# 3. SEVERITY & COST PARSING HELPERS
# =====================================================================
SEVERITY_WEIGHTS = {
    "CRITICAL": 100,
    "HIGH": 75,
    "MEDIUM": 50,
    "LOW": 25
}

def parse_cost_range(cost_str: str) -> tuple:
    """Extracts min and max numerical values from a cost string (e.g. '$150 - $400')."""
    if not cost_str:
        return (0.0, 0.0)
    numbers = [float(n.replace(',', '')) for n in re.findall(r'\d+(?:\.\d+)?', cost_str)]
    if len(numbers) == 1:
        return (numbers[0], numbers[0])
    elif len(numbers) >= 2:
        return (numbers[0], numbers[1])
    return (0.0, 0.0)

# =====================================================================
# 4. UNIFIED LLM BRIDGE (GROQ CLOUD OR LOCAL OLLAMA)
# =====================================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

async def call_llm(prompt: str, expect_json: bool = False) -> Optional[str]:
    """Queries Groq API if key is present, otherwise falls back to local Ollama."""
    # Option 1: Cloud LLM via Groq
    if GROQ_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
                body = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": "You are a professional automotive diagnostic master technician."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 400
                }
                if expect_json:
                    body["response_format"] = {"type": "json_object"}
                
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=body
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"].strip()
                else:
                    print(f"Groq API returned error {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"Groq API call exception: {e}")

    # Option 2: Local Ollama (Used when running locally on personal machine)
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
            else:
                print(f"Ollama local call failed with status: {resp.status_code}")
    except Exception as e:
        print(f"Ollama local call exception: {e}")

    return None

# =====================================================================
# 5. INFERENCE & HISTORY LOGIC
# =====================================================================
async def infer_dtc_details(code: str, make: str, model: str) -> Dict[str, Any]:
    prompt = f"""You are an expert master automotive diagnostic technician.
Analyze the Diagnostic Trouble Code (DTC) '{code}' for a '{make} {model}'.

Return ONLY valid JSON matching this exact structure:
{{
  "fault_name": "Standard OBD-II fault name",
  "category": "Powertrain / Chassis / Body / Network",
  "severity": "Critical / High / Medium / Low",
  "possible_causes": ["Cause 1", "Cause 2", "Cause 3"],
  "recommended_actions": ["Action 1", "Action 2", "Action 3"],
  "estimated_cost": "$100 - $350",
  "can_drive": true,
  "driver_directive": "Direct safety guidance for the driver."
}}
"""
    raw_response = await call_llm(prompt, expect_json=True)
    if raw_response:
        try:
            cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_response.strip(), flags=re.MULTILINE)
            return json.loads(cleaned)
        except Exception as e:
            print(f"JSON parse error during fallback inference: {e}")

    # Fallback if both LLMs fail
    return {
        "fault_name": f"Diagnostic Fault Code {code}",
        "category": "General Vehicle Network",
        "severity": "Medium",
        "possible_causes": ["Sensor circuit malfunction", "Wiring connector oxidation", "Module communication timeout"],
        "recommended_actions": ["Connect bi-directional scan tool", "Inspect sensor power/ground circuits"],
        "estimated_cost": "$100 - $350",
        "can_drive": True,
        "driver_directive": "Drive cautiously and book a professional diagnostic scan."
    }

def fetch_and_record_history(make: str, model: str, dtc: str, symptoms: str) -> Dict[str, Any]:
    """Queries and records scan events into Supabase Cloud Database."""
    vehicle_key = f"{make.strip().upper()}::{model.strip().upper()}"
    dtc_clean = dtc.strip().upper()
    
    seen_before = False
    occurrences = 1
    last_logged_at = None

    if supabase:
        try:
            # Query past instances
            response = supabase.table("vehicle_history_logs")\
                .select("id, created_at")\
                .eq("vehicle_key", vehicle_key)\
                .eq("dtc_code", dtc_clean)\
                .execute()
            
            count = len(response.data) if response.data else 0
            if count > 0:
                seen_before = True
                occurrences = count + 1
                last_logged_at = response.data[-1].get("created_at")

            # Insert new telemetry entry
            supabase.table("vehicle_history_logs").insert({
                "vehicle_key": vehicle_key,
                "make": make,
                "model": model,
                "dtc_code": dtc_clean,
                "symptoms": symptoms
            }).execute()
        except Exception as e:
            print(f"Supabase History query error: {e}")

    return {
        "seen_before": seen_before,
        "occurrences": occurrences,
        "last_logged_at": last_logged_at
    }

async def generate_integrated_summary(diagnoses: List[Dict[str, Any]], make: str, model: str, symptoms: str, total_costs: Dict[str, float]) -> str:
    """Generates an integrated mechanical triage synthesis using LLM."""
    if not diagnoses:
        return "No diagnostic trouble codes detected. Telemetry nominal."

    codes_summary = "\n".join([
        f"- Code {d['code']}: {d['fault_name']} | Severity: {d['severity']} | Recurrent: {d.get('history', {}).get('seen_before', False)}"
        for d in diagnoses
    ])

    prompt = f"""You are a master vehicle diagnostic strategist.
Vehicle: {make} {model}
Driver Reported Symptoms: "{symptoms}"
Active DTC Codes:
{codes_summary}

Total Repair Cost Range: ${total_costs['grand_total_min']:.0f} - ${total_costs['grand_total_max']:.0f}

Provide a concise, direct 3-bullet-point briefing for the driver and technician:
• What is happening: (Correlate DTCs with the driver's reported symptoms)
• Is it safe to drive?: (Clear yes/no directive with mechanical reasoning)
• What to do next & budget: (First prioritized inspection step and total expected repair cost bounds)
"""
    ai_summary = await call_llm(prompt, expect_json=False)
    if ai_summary:
        return ai_summary

    top = diagnoses[0]
    return (
        f"• What is happening: Primary issue detected is {top['fault_name']} ({top['code']}).\n"
        f"• Is it safe to drive?: {'Vehicle may be driven cautiously.' if top.get('can_drive', True) else 'DO NOT DRIVE until inspected.'}\n"
        f"• What to do next & budget: Estimated total cost is ${total_costs['grand_total_min']:.0f} – ${total_costs['grand_total_max']:.0f}."
    )

# =====================================================================
# 6. API REQUEST/RESPONSE MODELS & ROUTE
# =====================================================================
class DiagnoseRequest(BaseModel):
    make: str
    model: str
    dtcs: List[str]
    symptoms: Optional[str] = ""

@app.get("/")
def read_root():
    return {"status": "online", "system": "Autonomous Vehicle Error Detection & Diagnostic Core"}

@app.post("/diagnose")
async def run_diagnostics(req: DiagnoseRequest):
    if not req.dtcs:
        raise HTTPException(status_code=400, detail="At least one DTC code is required.")

    processed_diagnoses = []
    total_parts_min, total_parts_max = 0.0, 0.0
    total_labor_min, total_labor_max = 0.0, 0.0

    for raw_code in req.dtcs:
        code = raw_code.strip().upper()
        if not code:
            continue

        # 1. Dataset Lookup or Dynamic LLM fallback
        if code in LOCAL_DTC_DB:
            fault_data = dict(LOCAL_DTC_DB[code])
            source = "Dataset (Excel)"
        else:
            fault_data = await infer_dtc_details(code, req.make, req.model)
            source = "LLM Dynamic Inference"

        # 2. History & Recurrence Check
        history_meta = fetch_and_record_history(req.make, req.model, code, req.symptoms or "")

        # 3. Calculate Cost Accumulation
        cost_min, cost_max = parse_cost_range(fault_data.get("estimated_cost", "$100 - $300"))
        # Assume approx 55% parts, 45% labor split for structured itemization
        p_min, p_max = cost_min * 0.55, cost_max * 0.55
        l_min, l_max = cost_min * 0.45, cost_max * 0.45

        total_parts_min += p_min
        total_parts_max += p_max
        total_labor_min += l_min
        total_labor_max += l_max

        # 4. Severity Rank Weight
        sev_str = str(fault_data.get("severity", "MEDIUM")).upper()
        base_weight = SEVERITY_WEIGHTS.get(sev_str, 50)
        # Bump priority score if fault has recurring history
        if history_meta.get("seen_before"):
            base_weight += 15

        item_result = {
            "code": code,
            "source": source,
            "weight": base_weight,
            "severity": sev_str,
            "fault_name": fault_data.get("fault_name", f"Diagnostic Code {code}"),
            "category": fault_data.get("category", "General"),
            "possible_causes": fault_data.get("possible_causes", []),
            "recommended_actions": fault_data.get("recommended_actions", []),
            "estimated_cost": fault_data.get("estimated_cost", f"${cost_min:.0f} - ${cost_max:.0f}"),
            "can_drive": fault_data.get("can_drive", True),
            "driver_directive": fault_data.get("driver_directive", "Check with diagnostic scan tool."),
            "history": history_meta
        }
        processed_diagnoses.append(item_result)

    # 5. Priority Checker Ranking (Sort descending by calculated weight)
    processed_diagnoses.sort(key=lambda x: x["weight"], reverse=True)
    for idx, item in enumerate(processed_diagnoses, start=1):
        item["priority_rank"] = idx

    cost_summary = {
        "parts_min": round(total_parts_min),
        "parts_max": round(total_parts_max),
        "labor_min": round(total_labor_min),
        "labor_max": round(total_labor_max),
        "grand_total_min": round(total_parts_min + total_labor_min),
        "grand_total_max": round(total_parts_max + total_labor_max)
    }

    # 6. Generate Integrated AI Synthesis
    ai_interpretation = await generate_integrated_summary(
        processed_diagnoses, req.make, req.model, req.symptoms or "", cost_summary
    )

    return {
        "vehicle": {"make": req.make, "model": req.model},
        "reported_symptoms": req.symptoms,
        "total_faults_detected": len(processed_diagnoses),
        "cost_summary": cost_summary,
        "combined_summary": {
            "ai_interpretation": ai_interpretation
        },
        "diagnoses": processed_diagnoses
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
