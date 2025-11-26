# HDLLoop

## Setup
### Python Setup
Create venv environment
```
python -m venv hdlloopenv
```

Activate venv environment in windows
```
.\hdlloopenv\Scripts\activate 
```

Activate venv environment in mac
```
source ./hdlloopenv/bin/activate
```

Install dependencies for the first time
```
pip install chonkie[code] langfuse python-dotenv
```

Install dependencies using requirements.txt
```
pip install -r requirements.txt
```

### Environment variables
AI_API_KEY

## Launching the application
```
chainlit run app_ui.py -w
```
