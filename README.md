# Looopd Backend

Flask API for emotion detection and semantic similarity matching.

## Endpoints

### GET /health
Returns `{"status": "ok"}`

### POST /detect-emotion
```json
{ "text": "I feel so sad today" }
```
Returns `{"emotion": "sad", "detail": "deeply-sad", "scores": {...}}`

### POST /similar
```json
{
  "text": "I feel really lonely",
  "question_id": "uuid-here",
  "exclude_id": "uuid-of-just-recorded",
  "n": 2
}
```
Returns `{"similar": [...responses], "scores": [0.87, 0.72]}`

### POST /process
Combined endpoint — emotion + similarity in one call.
```json
{
  "text": "transcript here",
  "question_id": "uuid",
  "exclude_id": "uuid"
}
```
Returns `{"emotion": "sad", "detail": "deeply-sad", "similar": [...], "scores": [...]}`

## Deploy on Render.com

1. Push this folder to a GitHub repo
2. Go to render.com → New → Web Service
3. Connect the repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app`
6. Deploy
