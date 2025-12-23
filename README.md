# agro-ai-disease-detection
AI-based crop disease detection system for Sri Lankan agriculture
## Run inference (CLI)

Activate venv:
```powershell
.\.venv\Scripts\Activate.ps1

Predict one image:
python src/infer.py "path\to\image.jpg"

Adjust threshold
python src/infer.py "path\to\image.jpg" --threshold 0.7
