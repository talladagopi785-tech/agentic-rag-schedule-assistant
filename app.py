from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from datetime import datetime, timedelta, date
from typing import Optional, Literal
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import re
import json
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(APP_DIR, "chroma_db")
DATA_FILE = os.path.join(APP_DIR, "schedule.json")

app = FastAPI(title="Agentic RAG Schedule Assistant", version="1.0.0")

# -----------------------------
# Embeddings + ChromaDB
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name="schedule",
    metadata={"hnsw:space": "cosine"}
)

schedule = []


# -----------------------------
# Persistence helpers
# -----------------------------
def save_schedule():
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2, ensure_ascii=False)


def load_schedule():
    global schedule
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                schedule = json.load(f)
            return
        except Exception:
            schedule = []

    # If JSON does not exist, recover schedule from Chroma metadata.
    try:
        data = collection.get(include=["metadatas"])
        recovered = []
        for meta in data.get("metadatas", []):
            if meta and meta.get("event_json"):
                recovered.append(json.loads(meta["event_json"]))
        if recovered:
            schedule = recovered
            save_schedule()
    except Exception:
        pass


# -----------------------------
# Validation / formatting
# -----------------------------
EVENT_TYPES = {"meeting", "workshop", "task", "appointment"}


def valid_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def valid_time(value: str) -> bool:
    try:
        datetime.strptime(value, "%H:%M")
        return True
    except ValueError:
        return False


def normalize_event(event: dict) -> dict:
    event["date"] = str(event["date"])
    event["start_time"] = str(event["start_time"])
    event["end_time"] = str(event["end_time"])
    return event


def event_text(event: dict) -> str:
    return (
        f"Title: {event['title']}. "
        f"Type: {event['type']}. "
        f"Date: {event['date']}. "
        f"Start time: {event['start_time']}. "
        f"End time: {event['end_time']}. "
        f"Description: {event.get('description', '')}."
    )


def index_event(event: dict):
    text = event_text(event)
    collection.upsert(
        ids=[event["id"]],
        documents=[text],
        embeddings=[embedding_model.encode(text, normalize_embeddings=True).tolist()],
        metadatas=[{
            "event_id": event["id"],
            "date": event["date"],
            "type": event["type"],
            "title": event["title"],
            # Chroma metadata supports strings/numbers/bools, so store JSON as string.
            "event_json": json.dumps(event)
        }]
    )


def remove_from_index(event_id: str):
    try:
        collection.delete(ids=[event_id])
    except Exception:
        pass


def rebuild_index():
    if collection.count() > 0:
        return
    for event in schedule:
        index_event(event)


# -----------------------------
# Sample 30-day schedule
# -----------------------------
def create_sample_data():
    if schedule:
        return

    today = datetime.now().date()

    samples = [
        ("Team Meeting", "meeting", 1, "10:00", "11:00", "Weekly project team meeting"),
        ("Python Workshop", "workshop", 3, "14:00", "16:00", "Advanced Python programming workshop"),
        ("Submit Project Report", "task", 5, "09:00", "10:00", "Submit final project report"),
        ("Doctor Appointment", "appointment", 7, "11:30", "12:30", "Regular appointment"),
        ("Client Meeting", "meeting", 9, "15:00", "16:00", "Discuss project requirements"),
        ("AI Workshop", "workshop", 11, "10:00", "13:00", "Introduction to Agentic AI"),
        ("Complete Assignment", "task", 13, "16:00", "17:00", "Complete machine learning assignment"),
        ("Project Review", "meeting", 16, "11:00", "12:00", "Project progress review"),
        ("Team Presentation", "meeting", 20, "13:00", "14:00", "Present project progress"),
        ("Database Workshop", "workshop", 24, "10:00", "12:00", "Database systems workshop"),
        ("Dentist Appointment", "appointment", 27, "15:00", "16:00", "Dental check-up"),
        ("Final Project Task", "task", 29, "09:00", "11:00", "Finish final project work"),
    ]

    for title, event_type, offset, start, end, description in samples:
        event = {
            "id": str(uuid.uuid4()),
            "title": title,
            "type": event_type,
            "date": (today + timedelta(days=offset)).isoformat(),
            "start_time": start,
            "end_time": end,
            "description": description
        }
        schedule.append(event)
        index_event(event)

    save_schedule()


# -----------------------------
# TOOL 1: get_schedule
# -----------------------------
def get_schedule(
    query: str = "",
    date_value: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    event_type: Optional[str] = None,
):
    """
    Retrieves schedule information using exact filters and/or ChromaDB
    semantic retrieval.
    """

    results = list(schedule)

    if date_value:
        results = [e for e in results if e["date"] == date_value]

    if event_type:
        results = [e for e in results if e["type"] == event_type]

    if start_time and end_time:
        # Return events overlapping the requested time window.
        s = datetime.strptime(start_time, "%H:%M").time()
        en = datetime.strptime(end_time, "%H:%M").time()
        filtered = []
        for e in results:
            es = datetime.strptime(e["start_time"], "%H:%M").time()
            ee = datetime.strptime(e["end_time"], "%H:%M").time()
            if es < en and ee > s:
                filtered.append(e)
        results = filtered

    if date_value or event_type or (start_time and end_time):
        return sorted(results, key=lambda x: (x["date"], x["start_time"]))

    if not query:
        return sorted(results, key=lambda x: (x["date"], x["start_time"]))

    if collection.count() == 0:
        return []

    embedding = embedding_model.encode(
        query, normalize_embeddings=True
    ).tolist()

    count = min(10, collection.count())
    response = collection.query(
        query_embeddings=[embedding],
        n_results=count
    )

    ids = response.get("ids", [[]])[0]
    distances = response.get("distances", [[]])[0]

    by_id = {e["id"]: e for e in schedule}
    output = []

    for event_id, distance in zip(ids, distances):
        if event_id in by_id:
            item = dict(by_id[event_id])
            item["_similarity_distance"] = round(float(distance), 4)
            output.append(item)

    return output


# -----------------------------
# TOOL 2: update_schedule
# -----------------------------
def update_schedule(
    action: Literal["add", "update", "remove"],
    title: Optional[str] = None,
    event_id: Optional[str] = None,
    date_value: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    event_type: str = "meeting",
    description: str = "",
):
    """
    Adds, updates, or removes schedule entries.
    """

    if action == "add":
        if not title or not date_value or not start_time or not end_time:
            return {
                "success": False,
                "message": "title, date, start_time and end_time are required"
            }

        if not valid_date(date_value):
            return {"success": False, "message": "Invalid date. Use YYYY-MM-DD."}

        if not valid_time(start_time) or not valid_time(end_time):
            return {"success": False, "message": "Invalid time. Use HH:MM."}

        if event_type not in EVENT_TYPES:
            event_type = "meeting"

        if start_time >= end_time:
            return {"success": False, "message": "End time must be after start time."}

        event = {
            "id": str(uuid.uuid4()),
            "title": title.strip(),
            "type": event_type,
            "date": date_value,
            "start_time": start_time,
            "end_time": end_time,
            "description": description.strip()
        }

        schedule.append(event)
        index_event(event)
        save_schedule()

        return {"success": True, "message": "Event added.", "event": event}

    if action == "update":
        if not event_id:
            return {"success": False, "message": "event_id is required"}

        for event in schedule:
            if event["id"] == event_id:
                if date_value and not valid_date(date_value):
                    return {"success": False, "message": "Invalid date."}
                if start_time and not valid_time(start_time):
                    return {"success": False, "message": "Invalid start time."}
                if end_time and not valid_time(end_time):
                    return {"success": False, "message": "Invalid end time."}

                if title:
                    event["title"] = title.strip()
                if date_value:
                    event["date"] = date_value
                if start_time:
                    event["start_time"] = start_time
                if end_time:
                    event["end_time"] = end_time
                if event_type in EVENT_TYPES:
                    event["type"] = event_type
                if description:
                    event["description"] = description.strip()

                if event["start_time"] >= event["end_time"]:
                    return {"success": False, "message": "End time must be after start time."}

                remove_from_index(event_id)
                index_event(event)
                save_schedule()

                return {"success": True, "message": "Event updated.", "event": event}

        return {"success": False, "message": "Event not found."}

    if action == "remove":
        if not event_id:
            return {"success": False, "message": "event_id is required"}

        for i, event in enumerate(schedule):
            if event["id"] == event_id:
                removed = schedule.pop(i)
                remove_from_index(event_id)
                save_schedule()
                return {"success": True, "message": "Event removed.", "event": removed}

        return {"success": False, "message": "Event not found."}

    return {"success": False, "message": "Unknown action."}


# -----------------------------
# Natural-language parsing
# -----------------------------
def parse_date_from_text(text: str) -> Optional[str]:
    lower = text.lower()
    today = datetime.now().date()

    if "today" in lower:
        return today.isoformat()

    if "tomorrow" in lower:
        return (today + timedelta(days=1)).isoformat()

    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
    }

    for name, target_weekday in weekdays.items():
        if re.search(rf"\b{name}\b", lower):
            delta = (target_weekday - today.weekday()) % 7
            if delta == 0:
                delta = 7
            return (today + timedelta(days=delta)).isoformat()

    # Examples: August 20, August 20 2026
    match = re.search(
        r"\b(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\s+(\d{1,2})"
        r"(?:\s*,?\s*(\d{4}))?\b",
        lower,
        re.I
    )

    if match:
        try:
            month = datetime.strptime(match.group(1), "%B").month
            day = int(match.group(2))
            year = int(match.group(3)) if match.group(3) else today.year
            if not match.group(3) and month < today.month:
                year += 1
            return date(year, month, day).isoformat()
        except ValueError:
            return None

    return None


def parse_time_from_text(text: str) -> Optional[str]:
    match = re.search(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        text,
        re.I
    )
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    period = match.group(3).lower()

    if hour > 12 or minute > 59:
        return None

    if period == "pm" and hour != 12:
        hour += 12
    if period == "am" and hour == 12:
        hour = 0

    return f"{hour:02d}:{minute:02d}"


def parse_time_range(text: str):
    times = re.findall(
        r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
        text,
        re.I
    )
    parsed = [parse_time_from_text(x) for x in times]
    parsed = [x for x in parsed if x]
    if len(parsed) >= 2:
        return parsed[0], parsed[1]
    return None, None


def infer_event_type(text: str) -> str:
    lower = text.lower()
    for kind in EVENT_TYPES:
        if kind in lower:
            return kind
    return "meeting"


def extract_title(text: str, event_type: str) -> str:
    lower = text.lower()
    patterns = [
        r"add\s+(?:a|an)\s+(.+?)\s+on\s+",
        r"create\s+(?:a|an)\s+(.+?)\s+on\s+",
        r"schedule\s+(?:a|an)\s+(.+?)\s+on\s+",
    ]

    for pattern in patterns:
        match = re.search(pattern, lower, re.I)
        if match:
            candidate = match.group(1).strip()
            for kind in EVENT_TYPES:
                candidate = re.sub(rf"\b{kind}\b", "", candidate, flags=re.I)
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if candidate:
                return candidate.title()

    return event_type.title()


# -----------------------------
# Agent
# -----------------------------
def schedule_agent(user_query: str):
    q = user_query.strip()
    lower = q.lower()

    if not q:
        return {"agent_action": "get_schedule", "tool_result": []}

    # ADD
    if re.search(r"\b(add|create|schedule)\b", lower):
        date_value = parse_date_from_text(q)
        start_time = parse_time_from_text(q)
        event_type = infer_event_type(q)
        title = extract_title(q, event_type)

        if not date_value:
            return {
                "agent_action": "update_schedule",
                "tool_result": {
                    "success": False,
                    "message": "Please provide a date, for example: August 20."
                }
            }

        if not start_time:
            return {
                "agent_action": "update_schedule",
                "tool_result": {
                    "success": False,
                    "message": "Please provide a time, for example: 3 PM."
                }
            }

        dt = datetime.strptime(start_time, "%H:%M")
        end_time = (dt + timedelta(hours=1)).strftime("%H:%M")

        result = update_schedule(
            action="add",
            title=title,
            date_value=date_value,
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
            description=q
        )
        return {"agent_action": "update_schedule", "tool_result": result}

    # REMOVE
    if re.search(r"\b(delete|remove|cancel)\b", lower):
        date_value = parse_date_from_text(q)
        candidates = get_schedule(query=q, date_value=date_value)

        if not candidates:
            return {
                "agent_action": "get_schedule",
                "tool_result": [],
                "message": "No matching schedule event was found."
            }

        target = candidates[0]
        result = update_schedule("remove", event_id=target["id"])
        return {"agent_action": "update_schedule", "tool_result": result}

    # UPDATE / MOVE
    if re.search(r"\b(move|reschedule|change|update)\b", lower):
        new_time = parse_time_from_text(q)

        old_match = re.search(
            r"\bfrom\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
            q, re.I
        )
        old_time = parse_time_from_text(old_match.group(1)) if old_match else None

        date_value = parse_date_from_text(q)

        # First use exact date when supplied, otherwise semantic retrieval.
        candidates = get_schedule(
            query=q,
            date_value=date_value
        )

        target = None
        for event in candidates:
            if old_time is None or event["start_time"] == old_time:
                target = event
                break

        if target and new_time:
            dt = datetime.strptime(new_time, "%H:%M")
            duration = (
                datetime.strptime(target["end_time"], "%H:%M")
                - datetime.strptime(target["start_time"], "%H:%M")
            )
            new_end = (
                dt + duration
            ).strftime("%H:%M")

            result = update_schedule(
                action="update",
                event_id=target["id"],
                start_time=new_time,
                end_time=new_end
            )
            return {"agent_action": "update_schedule", "tool_result": result}

        return {
            "agent_action": "get_schedule",
            "tool_result": candidates,
            "message": "I could not identify both the event and the new time."
        }

    # RETRIEVE / FREE-TIME
    date_value = parse_date_from_text(q)
    start_time, end_time = parse_time_range(q)

    event_type = None
    for kind in EVENT_TYPES:
        if kind in lower:
            event_type = kind
            break

    events = get_schedule(
        query=q,
        date_value=date_value,
        start_time=start_time,
        end_time=end_time,
        event_type=event_type
    )

    if "free" in lower:
        return {
            "agent_action": "get_schedule",
            "tool_result": {
                "free": len(events) == 0,
                "events": events
            }
        }

    return {
        "agent_action": "get_schedule",
        "tool_result": events
    }


# -----------------------------
# API models
# -----------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class EventRequest(BaseModel):
    title: str
    event_type: str = "meeting"
    date: str
    start_time: str
    end_time: str
    description: str = ""


# -----------------------------
# API routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    template = os.path.join(APP_DIR, "templates", "index.html")
    with open(template, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/chat")
def chat(request: ChatRequest):
    return {
        "query": request.message,
        "response": schedule_agent(request.message)
    }


@app.get("/schedule")
def all_schedule():
    return {
        "events": sorted(
            schedule,
            key=lambda e: (e["date"], e["start_time"])
        )
    }


@app.post("/schedule")
def add_schedule_event(event: EventRequest):
    result = update_schedule(
        action="add",
        title=event.title,
        event_type=event.event_type,
        date_value=event.date,
        start_time=event.start_time,
        end_time=event.end_time,
        description=event.description
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


# -----------------------------
# Startup
# -----------------------------
load_schedule()
if not schedule:
    create_sample_data()
else:
    rebuild_index()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
