from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timedelta
import chromadb
import hashlib, json, math, re, uuid, os

app = FastAPI(title="Agentic RAG Schedule Assistant")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "schedule.json"
CHROMA_DIR = str(BASE_DIR / "chroma_db")
TEMPLATE_FILE = BASE_DIR / "templates" / "index.html"
EMBEDDING_DIM = 128

def create_embedding(text: str):
    vector = [0.0] * EMBEDDING_DIM
    for word in re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()):
        digest = hashlib.sha256(word.encode()).digest()
        index = int.from_bytes(digest[:4], "little") % EMBEDDING_DIM
        vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
    magnitude = math.sqrt(sum(x*x for x in vector))
    return [x/magnitude for x in vector] if magnitude else vector

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name="schedule_lightweight",
    metadata={"hnsw:space": "cosine"}
)

def save_schedule(data):
    DATA_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_schedule():
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    events = [
        ("Team Meeting", "meeting", 1, "10:00", "Weekly project progress meeting"),
        ("Python Workshop", "workshop", 2, "14:00", "Hands-on Python and FastAPI workshop"),
        ("Doctor Appointment", "appointment", 4, "09:30", "Regular appointment"),
        ("Submit Project Report", "task", 6, "16:00", "Finish and submit project report"),
        ("Client Meeting", "meeting", 8, "11:00", "Discuss project milestones"),
        ("Design Workshop", "workshop", 10, "15:00", "UI and workflow design session"),
        ("Project Sync", "meeting", 13, "13:00", "Discuss project progress"),
        ("Complete Assignment", "task", 16, "17:00", "Work on pending assignment"),
        ("AI Workshop", "workshop", 20, "11:00", "Practical AI learning session"),
        ("General Appointment", "appointment", 24, "10:30", "Scheduled appointment"),
        ("Final Review Meeting", "meeting", 28, "15:00", "Final project review")
    ]
    result = []
    for title, typ, offset, time, desc in events:
        result.append({
            "id": str(uuid.uuid4()),
            "date": (today + timedelta(days=offset)).strftime("%Y-%m-%d"),
            "time": time, "title": title, "type": typ,
            "description": desc, "duration": 60
        })
    save_schedule(result)
    return result

schedule = load_schedule()

def event_text(e):
    return f"{e['date']} {e['time']} {e['title']} {e['type']} {e.get('description','')}"

def index_event(e):
    collection.upsert(
        ids=[e["id"]],
        documents=[event_text(e)],
        embeddings=[create_embedding(event_text(e))],
        metadatas=[{
            "event_id": e["id"], "date": e["date"],
            "type": e["type"], "title": e["title"],
            "event_json": json.dumps(e)
        }]
    )

def rebuild_index():
    if collection.count():
        collection.delete(where={})
    for e in schedule:
        index_event(e)

try:
    rebuild_index()
except Exception as exc:
    print("Chroma warning:", exc)

# TOOL 1: get_schedule
def get_schedule(query="", date=None, time_from=None, time_to=None):
    if not schedule:
        return []
    text = " ".join(x for x in [query, date, time_from, time_to] if x)
    try:
        result = collection.query(
            query_embeddings=[create_embedding(text or "schedule")],
            n_results=min(10, collection.count())
        )
        ids = result.get("ids", [[]])[0]
        by_id = {e["id"]: e for e in schedule}
        output = [dict(by_id[i]) for i in ids if i in by_id]
    except Exception:
        output = list(schedule)

    if date:
        output = [e for e in output if e["date"] == date]
    return output

# TOOL 2: update_schedule
def update_schedule(action, event_id=None, data=None):
    global schedule
    data = data or {}

    if action == "add":
        event = {
            "id": str(uuid.uuid4()),
            "date": data["date"], "time": data["time"],
            "title": data.get("title", "New Meeting"),
            "type": data.get("type", "meeting"),
            "description": data.get("description", ""),
            "duration": int(data.get("duration", 60))
        }
        schedule.append(event)
        save_schedule(schedule)
        index_event(event)
        return event

    if action == "update":
        for e in schedule:
            if e["id"] == event_id:
                e.update({k:v for k,v in data.items()
                          if k in {"date","time","title","type","description","duration"}})
                save_schedule(schedule)
                index_event(e)
                return e
        return None

    if action == "remove":
        e = next((x for x in schedule if x["id"] == event_id), None)
        if not e:
            return None
        schedule = [x for x in schedule if x["id"] != event_id]
        save_schedule(schedule)
        try: collection.delete(ids=[event_id])
        except Exception: pass
        return e
    return None

def parse_date(text):
    low = text.lower()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if "today" in low: return today.strftime("%Y-%m-%d")
    if "tomorrow" in low: return (today+timedelta(days=1)).strftime("%Y-%m-%d")
    weekdays = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,
                "friday":4,"saturday":5,"sunday":6}
    for name, target in weekdays.items():
        if name in low:
            delta = (target-today.weekday()) % 7 or 7
            return (today+timedelta(days=delta)).strftime("%Y-%m-%d")
    m = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if m:
        try: return datetime(*map(int,m.groups())).strftime("%Y-%m-%d")
        except ValueError: pass
    return None

def parse_time(text):
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text.lower())
    if not m: return None
    h, minute, suffix = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if suffix == "pm" and h < 12: h += 12
    if suffix == "am" and h == 12: h = 0
    return f"{h:02d}:{minute:02d}" if 0 <= h <= 23 and 0 <= minute <= 59 else None

def fmt(e):
    return f"{e['date']} at {e['time']} — {e['title']} ({e['type']})"

def agent(query):
    low = query.lower().strip()
    date = parse_date(query)

    if low.startswith(("add ","create ","schedule ")):
        time = parse_time(query)
        if not date: return {"message":"Please provide a date, for example August 20."}
        if not time: return {"message":"Please provide a time, for example 3 PM."}
        m = re.search(r"(?:add|create|schedule)\s+(?:a|an|the)?\s*(.+?)(?:\s+on\s+|\s+at\s+)", query, re.I)
        title = m.group(1).strip().title() if m else "New Meeting"
        e = update_schedule("add", data={"date":date,"time":time,"title":title,"type":"meeting"})
        return {"message":f"Added: {fmt(e)}","events":[e]}

    if any(x in low for x in ["move ","reschedule ","change "]):
        new_match = re.search(r"\bto\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", low)
        old_match = re.search(r"\bfrom\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", low)
        new_time = parse_time(new_match.group(1)) if new_match else None
        old_time = parse_time(old_match.group(1)) if old_match else None
        candidates = get_schedule(query, date=date)
        if old_time: candidates = [e for e in candidates if e["time"] == old_time]
        if not candidates: return {"message":"I couldn't find the meeting to move."}
        if not new_time: return {"message":"Please provide the new time."}
        e = update_schedule("update", candidates[0]["id"], {"time":new_time})
        return {"message":f"Moved to {new_time}: {e['title']}","events":[e]}

    if low.startswith(("delete ","remove ","cancel ")):
        candidates = get_schedule(query, date=date)
        if not candidates: return {"message":"I couldn't find the event to remove."}
        e = update_schedule("remove", candidates[0]["id"])
        return {"message":f"Removed: {fmt(e)}","events":[]}

    events = get_schedule(query, date=date)
    if "free" in low or "available" in low:
        return {"message":"You appear to be free." if not events else "You have these scheduled events:","events":events}
    return {"message":"No matching schedule entries found." if not events else "Here are the relevant schedule entries:","events":events}

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
def home():
    return TEMPLATE_FILE.read_text(encoding="utf-8")

@app.get("/api/schedule")
def api_schedule():
    return {"events": schedule}

@app.post("/api/chat")
def chat(req: ChatRequest):
    return agent(req.message)

@app.get("/health")
def health():
    return {"status":"ok","events":len(schedule),"vector_database":"ChromaDB"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT","8000")))
