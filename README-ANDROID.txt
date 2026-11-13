CLIPFORGE AI — FREE / OPEN-SOURCE BACKEND

What is free here
- No OpenAI key, paid transcription API, or paid LLM API is used.
- faster-whisper runs a local Whisper model for speech-to-text.
- FFmpeg cuts, crops, adds subtitles, and exports MP4 files.
- Highlight selection uses local transcript heuristics; it is not a generative AI editor and cannot guarantee viral clips.
- Captions stay in the original spoken language; automatic translation is not included.

IMPORTANT HOSTING REALITY
The software is free, but compute and storage are not guaranteed to be free. Whisper downloads a model on first run (default: base). CPU-only processing can be slow and needs sufficient RAM/disk. Free web hosts often sleep, limit CPU/RAM, and erase temporary files. Do not expect reliable 143 MB uploads on a tiny free instance.

Run locally (recommended for testing)
1. Install Python 3.10+ and FFmpeg (including ffprobe).
2. Extract this ZIP.
3. In the extracted folder run: pip install -r requirements.txt
4. Run: python app.py
5. Open http://127.0.0.1:8000/api/status and confirm ffmpeg=true and ai=true.
6. For an APK on another device, the backend must be reachable over HTTPS. A localhost server on your phone is not automatically reachable from the internet.

Environment variables
- WHISPER_MODEL=base (options: tiny, base, small; larger models need more RAM/time)
- MAX_UPLOAD_MB=500 (adjust only if the host supports it)
- DATA_DIR=data (use persistent storage if the host offers it)

YouTube URL import is intentionally disabled. Upload videos you own or have permission to edit.

Start command for a suitable Linux host
gunicorn -w 1 -k gthread --threads 2 -b 0.0.0.0:$PORT app:app

For Docker hosting, use the included Dockerfile.
