# Redis & Docker — Beginner's Guide for Task 13

Everything you need to understand what Redis and Docker are, why we use them,
and how to operate them day-to-day while building AI backends.

---

## Part 1 — Docker

### What is Docker?

Docker lets you run a program inside a sealed box called a **container**.
The container has everything that program needs (OS libs, config, the binary itself)
and nothing extra. It runs the same way on your laptop, your teammate's laptop, and
a cloud server — no "works on my machine" problems.

```
Without Docker:              With Docker:
Install Redis manually   →   docker run redis
Version conflicts        →   Each container has its own versions
"It works on my machine" →   Identical everywhere
Hard to clean up         →   docker rm = completely gone
```

### Core Concepts

| Term | What it is |
|------|-----------|
| **Image** | A read-only template. Like a recipe. e.g. `redis:7-alpine` |
| **Container** | A running instance of an image. Like a meal made from that recipe |
| **Port mapping** | `-p 6379:6379` — connects your laptop's port 6379 to the container's port 6379 |
| **Volume** | Persistent storage that survives container restarts |
| **`-d` flag** | Runs container in the background (detached mode) |
| **`--name`** | Gives the container a memorable name instead of a random hash |

### The Redis command we ran — explained piece by piece

```bash
docker run -d --name redis-dev -p 6379:6379 redis:7-alpine
```

| Piece | Meaning |
|-------|---------|
| `docker run` | Start a new container |
| `-d` | Run in background (detached) — you get your terminal back |
| `--name redis-dev` | Call it "redis-dev" so we can reference it by name |
| `-p 6379:6379` | Map laptop port 6379 → container port 6379 |
| `redis:7-alpine` | Use the official Redis 7 image, Alpine Linux variant (tiny, ~30MB) |

### Essential Docker Commands

```bash
# See all running containers
docker ps

# See all containers including stopped ones
docker ps -a

# Stop a running container
docker stop redis-dev

# Start a stopped container again
docker start redis-dev

# Remove a container completely
docker rm redis-dev

# See logs from a container
docker logs redis-dev

# Open a terminal inside a running container
docker exec -it redis-dev sh

# Open the Redis CLI inside the container
docker exec -it redis-dev redis-cli
```

### Why Docker for Redis specifically?

- Redis is a C program. Installing it on Windows natively is painful.
- With Docker, one command gives you a clean Redis 7 instance.
- Easy to reset: `docker rm redis-dev` wipes it, `docker run ...` creates fresh.
- In production, your Redis runs in a Docker container (or Kubernetes pod) too —
  so local dev matches production exactly.

---

## Part 2 — Redis

### What is Redis?

Redis is an **in-memory data store** that runs as a separate process.
Your Python app connects to it over a network socket (default: `localhost:6379`).

```
Your FastAPI app                Redis process
┌────────────────┐   TCP        ┌─────────────────────┐
│ redis_client   │─────────────▶│  All your keys/data │
│ .get("key")    │◀─────────────│  stored in RAM       │
│ .set("key","v")│              │  (optionally on disk)│
└────────────────┘              └─────────────────────┘
```

It is NOT a relational database. There are no tables, no SQL.
It stores **key → value** pairs, where the value can be one of several types.

### Redis Data Types (the ones we used)

#### String — the most basic type
```
SET  key value        # store
GET  key              # retrieve
DEL  key              # delete
```
Used in our `cache.py` — the cached JSON response is stored as a String.

#### List — ordered sequence of strings
```
RPUSH key value       # append to right end
LPUSH key value       # prepend to left end  
LRANGE key 0 -1       # get all items (0 = first, -1 = last)
LLEN key              # count items
```
Used in our `conversations.py` — each message is an item in a Redis List.

#### Integer counter (via INCR)
```
INCR key              # atomically increment by 1 (creates at 0 if missing)
INCRBY key 5          # increment by any amount
```
Used in our `rate_limiter.py` — each API key has a counter that Redis increments atomically.

### TTL — Time To Live

Every Redis key can have an expiry. When the TTL hits 0, Redis deletes the key automatically.
This is one of Redis's most useful features.

```bash
# Set a key with 60-second TTL
SET mykey "hello" EX 60

# Or set TTL separately
SET mykey "hello"
EXPIRE mykey 60

# Check how many seconds remain
TTL mykey        # returns seconds remaining, -1 = no expiry, -2 = key gone

# Remove expiry (make key permanent)
PERSIST mykey
```

In our project:
- Cache keys expire after **3600 seconds** (1 hour) — stale answers auto-delete
- Rate limit keys expire after **60 seconds** — window resets automatically
- Conversation keys expire after **86400 seconds** (24 hours) — old chats auto-clean

### Atomic Operations — Why INCR beats a Python counter

"Atomic" means the operation completes entirely or not at all — nothing can interrupt it mid-way.

```
Problem with Python dict (race condition):
  Thread A reads count = 19
  Thread B reads count = 19   ← same read, before A writes
  Thread A writes count = 20  ← under limit, allowed
  Thread B writes count = 20  ← under limit, allowed ← BUG: both got in!
  
  Result: 21st request slipped through.

Solution with Redis INCR (atomic):
  Thread A sends INCR → Redis processes it → returns 20
  Thread B sends INCR → Redis processes it → returns 21
  
  Redis handles one command at a time. Thread B gets 21, hits the limit.
  Result: exactly correct.
```

This is why production rate limiters use Redis INCR, not an in-memory counter.

### Namespaced Keys

When multiple features share one Redis database, prefix every key with its module name
so they don't collide and are easy to inspect:

```
cache:a3f9b2c1...          ← response cache
rate_limit:security@1      ← rate limit counter
conversation:550e8400-...  ← conversation message list
conversation_meta:550e...  ← conversation existence marker
```

In the Redis CLI you can filter by prefix:
```bash
KEYS cache:*              # all cache entries
KEYS rate_limit:*         # all rate limit counters
KEYS conversation:*       # all conversation lists
```

### Serialization — Why JSON

Redis stores everything as bytes/strings. Python objects (dicts, Pydantic models)
can't go in directly. We convert them:

```python
# Writing to Redis
payload = json.dumps(response.model_dump())   # Python dict → JSON string
redis_client.set(key, payload, ex=3600)

# Reading from Redis
raw = redis_client.get(key)                   # JSON string
data = json.loads(raw)                        # JSON string → Python dict
response = ChatResponse(**data)               # dict → Pydantic model
```

### `decode_responses=True` — What it does

Redis communicates over a binary protocol. By default redis-py returns `bytes`:
```python
redis_client.get("key")   # returns  b"hello"  ← bytes, not str
```

With `decode_responses=True`:
```python
redis_client.get("key")   # returns  "hello"   ← str, ready to use
```

We set this once when creating the client in `redis_client.py`. Every command
then returns normal Python strings — no `.decode("utf-8")` needed anywhere.

---

## Part 3 — Redis CLI Cheat Sheet

```bash
# Connect to your running Redis container
docker exec -it redis-dev redis-cli

# ── Keys ──────────────────────────────────────────────────────────
KEYS *                          # list ALL keys (avoid in production — slow)
KEYS cache:*                    # filter by prefix
TYPE cache:abc123               # what type is this key? (string, list, hash...)
TTL  cache:abc123               # seconds until expiry (-1=none, -2=gone)
DEL  cache:abc123               # delete one key
EXISTS cache:abc123             # 1=exists, 0=doesn't exist

# ── Strings ───────────────────────────────────────────────────────
GET  cache:abc123               # read a string value
SET  testkey "hello" EX 30      # write a string with 30s TTL

# ── Lists (conversations) ─────────────────────────────────────────
LRANGE conversation:uuid 0 -1   # get all messages
LLEN   conversation:uuid        # count messages
RPUSH  conversation:uuid '{"role":"user","content":"hi"}'

# ── Counters (rate limiting) ──────────────────────────────────────
INCR rate_limit:security@1      # manually increment
GET  rate_limit:security@1      # read current count

# ── Inspection ────────────────────────────────────────────────────
DBSIZE                          # total number of keys in current DB
INFO memory                     # memory usage stats
INFO stats                      # hit/miss stats, commands/sec
MONITOR                         # live stream of all commands (dev only!)

# ── Danger zone (dev only!) ───────────────────────────────────────
FLUSHDB                         # delete ALL keys in current DB
```

> **MONITOR** is incredibly useful while learning — open a second terminal,
> run `docker exec -it redis-dev redis-cli MONITOR`, then send requests to
> your API and watch every Redis command appear in real time.

---

## Part 4 — How The Three Modules Connect

```
POST /chat request
       │
       ▼
 rate_limiter.py
   INCR rate_limit:{api_key}          ← Redis String (counter)
   EXPIRE rate_limit:{api_key} 60     ← TTL starts on first request
   count > 20? → 429                  ← enforced atomically
       │
       ▼
 cache.py
   GET cache:{sha256(message)}        ← Redis String (JSON)
   hit? → return cached response
   miss? → continue
       │
       ▼
 conversations.py
   EXISTS conversation_meta:{id}      ← Redis String (marker)
   LRANGE conversation:{id} 0 -1      ← Redis List (history)
       │
       ▼
 LLM API call
       │
       ▼
 conversations.py
   RPUSH conversation:{id} message    ← append to Redis List
   EXPIRE conversation:{id} 86400     ← refresh 24h TTL
       │
       ▼
 cache.py
   SET cache:{hash} json EX 3600      ← store in Redis with 1h TTL
       │
       ▼
 Response returned to client
```

---

## Part 5 — Common Mistakes to Avoid

| Mistake | Why it's wrong | Correct approach |
|---------|---------------|------------------|
| `redis.flushdb()` in app code | Wipes all data silently | Only ever in a dev script or test teardown |
| New `Redis()` connection per request | Wastes sockets, slows requests | One shared client with a connection pool |
| Storing Python objects directly | Redis only stores strings/bytes | `json.dumps()` before write, `json.loads()` after read |
| Keys without TTL | Redis RAM fills up forever | Every key gets `EX` or `EXPIRE` |
| No namespace prefix | `user:1` clashes with another module's `1` | Always prefix: `cache:`, `rate_limit:`, `conversation:` |
| Hardcoding `localhost:6379` | Breaks in Docker Compose / production | Read from env vars: `REDIS_HOST`, `REDIS_PORT` |

---

## Part 6 — Quick Start (Run It Now)

```bash
# 1. Start Redis
docker run -d --name redis-dev -p 6379:6379 redis:7-alpine

# 2. Verify Redis is up
docker exec -it redis-dev redis-cli ping
# → PONG

# 3. Go into the task_02 folder
cd 02_Tasks/task_02/smart_backend

# 4. Install dependencies
pip install -r requirements.txt

# 5. Add your Groq key to .env
# Edit .env and replace "your_groq_api_key_here"

# 6. Start the server
uvicorn app.main:app --reload

# 7. In a second terminal — watch Redis live
docker exec -it redis-dev redis-cli MONITOR

# 8. Send a test request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: security@1" \
  -d '{"message": "What is 25 multiplied by 4?"}'

# 9. Send the SAME request again — should come from cache
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: security@1" \
  -d '{"message": "What is 25 multiplied by 4?"}'

# 10. Inspect keys in Redis
docker exec -it redis-dev redis-cli KEYS "*"
docker exec -it redis-dev redis-cli TTL "cache:<hash_from_above>"
```
