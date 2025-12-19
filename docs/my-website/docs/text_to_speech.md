import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /audio/speech

## Overview

| Feature | Supported | Notes |
|---------|-----------|-------|
| Cost Tracking | ✅ | Works with all supported models |
| Logging | ✅ | Works across all integrations |
| End-user Tracking | ✅ | |
| Fallbacks | ✅ | Works between supported models |
| Loadbalancing | ✅ | Works between supported models |
| Guardrails | ✅ | Applies to input text (non-streaming only) |
| Supported Providers | OpenAI, Azure OpenAI, Vertex AI | |

## **Dheera AI Python SDK Usage**
### Quick Start 

```python
from pathlib import Path
from dheera_ai import speech
import os 

os.environ["OPENAI_API_KEY"] = "sk-.."

speech_file_path = Path(__file__).parent / "speech.mp3"
response = speech(
        model="openai/tts-1",
        voice="alloy",
        input="the quick brown fox jumped over the lazy dogs",
    )
response.stream_to_file(speech_file_path)
```

### Async Usage 

```python
from dheera_ai import aspeech
from pathlib import Path
import os, asyncio

os.environ["OPENAI_API_KEY"] = "sk-.."

async def test_async_speech(): 
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = await dheera_ai.aspeech(
            model="openai/tts-1",
            voice="alloy",
            input="the quick brown fox jumped over the lazy dogs",
            api_base=None,
            api_key=None,
            organization=None,
            project=None,
            max_retries=1,
            timeout=600,
            client=None,
            optional_params={},
        )
    response.stream_to_file(speech_file_path)

asyncio.run(test_async_speech())
```

## **Dheera AI Proxy Usage**

Dheera AI provides an openai-compatible `/audio/speech` endpoint for Text-to-speech calls.

```bash
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "alloy"
  }' \
  --output speech.mp3
```

**Setup**

```bash
- model_name: tts
  dheera_ai_params:
    model: openai/tts-1
    api_key: os.environ/OPENAI_API_KEY
```

```bash
dheera_ai --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```
## **Supported Providers**

| Provider    | Link to Usage      |
|-------------|--------------------|
| OpenAI      |   [Usage](#quick-start)                 |
| Azure OpenAI|   [Usage](../docs/providers/azure#azure-text-to-speech-tts)                 |
| Azure AI Speech Service (AVA)|   [Usage](../docs/providers/azure_ai_speech)                 |
| Vertex AI   |   [Usage](../docs/providers/vertex#text-to-speech-apis)                 |
| Gemini      |   [Usage](#gemini-text-to-speech)                 |
| ElevenLabs  |   [Usage](../docs/providers/elevenlabs#text-to-speech-tts)                 |

## `/audio/speech` to `/chat/completions` Bridge

Dheera AI allows you to use `/chat/completions` models to generate speech through the `/audio/speech` endpoint. This is useful for models like Gemini's TTS-enabled models that are only accessible via `/chat/completions`.

### Gemini Text-to-Speech

#### Python SDK Usage

```python showLineNumbers title="Gemini Text-to-Speech SDK Usage"
import dheera_ai
import os

# Set your Gemini API key
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

def test_audio_speech_gemini():
    result = dheera_ai.speech(
        model="gemini/gemini-2.5-flash-preview-tts",
        input="the quick brown fox jumped over the lazy dogs",
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    # Save to file
    from pathlib import Path
    speech_file_path = Path(__file__).parent / "gemini_speech.mp3"
    result.stream_to_file(speech_file_path)
    print(f"Audio saved to {speech_file_path}")

test_audio_speech_gemini()
```

#### Async Usage

```python showLineNumbers title="Gemini Text-to-Speech Async Usage"
import dheera_ai
import asyncio
import os
from pathlib import Path

os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

async def test_async_gemini_speech():
    speech_file_path = Path(__file__).parent / "gemini_speech.mp3"
    response = await dheera_ai.aspeech(
        model="gemini/gemini-2.5-flash-preview-tts",
        input="the quick brown fox jumped over the lazy dogs",
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    response.stream_to_file(speech_file_path)
    print(f"Audio saved to {speech_file_path}")

asyncio.run(test_async_gemini_speech())
```

#### Dheera AI Proxy Usage

**Setup Config:**

```yaml showLineNumbers title="Gemini Proxy Configuration"
model_list:
- model_name: gemini-tts
  dheera_ai_params:
    model: gemini/gemini-2.5-flash-preview-tts
    api_key: os.environ/GEMINI_API_KEY
```

**Start Proxy:**

```bash showLineNumbers title="Start Dheera AI Proxy"
dheera_ai --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

**Make Request:**

```bash showLineNumbers title="Gemini TTS Request"
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-tts",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "alloy"
  }' \
  --output gemini_speech.mp3
```

### Vertex AI Text-to-Speech

#### Python SDK Usage

```python showLineNumbers title="Vertex AI Text-to-Speech SDK Usage"
import dheera_ai
import os
from pathlib import Path

# Set your Google credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/service-account.json"

def test_audio_speech_vertex():
    result = dheera_ai.speech(
        model="vertex_ai/gemini-2.5-flash-preview-tts",
        input="the quick brown fox jumped over the lazy dogs",
    )
    
    # Save to file
    speech_file_path = Path(__file__).parent / "vertex_speech.mp3"
    result.stream_to_file(speech_file_path)
    print(f"Audio saved to {speech_file_path}")

test_audio_speech_vertex()
```

#### Dheera AI Proxy Usage

**Setup Config:**

```yaml showLineNumbers title="Vertex AI Proxy Configuration"
model_list:
- model_name: vertex-tts
  dheera_ai_params:
    model: vertex_ai/gemini-2.5-flash-preview-tts
    vertex_project: your-project-id
    vertex_location: us-central1
```

**Make Request:**

```bash showLineNumbers title="Vertex AI TTS Request"
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vertex-tts",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "en-US-Wavenet-D"
  }' \
  --output vertex_speech.mp3
```

## ✨ Enterprise Dheera AI Proxy - Set Max Request File Size 

Use this when you want to limit the file size for requests sent to `audio/transcriptions`

```yaml
- model_name: whisper
  dheera_ai_params:
    model: whisper-1
    api_key: sk-*******
    max_file_size_mb: 0.00001 # 👈 max file size in MB  (Set this intentionally very small for testing)
  model_info:
    mode: audio_transcription
```

Make a test Request with a valid file
```shell
curl --location 'http://localhost:4000/v1/audio/transcriptions' \
--header 'Authorization: Bearer sk-1234' \
--form 'file=@"/Users/ishaanjaffer/Github/dheera_ai/tests/gettysburg.wav"' \
--form 'model="whisper"'
```


Expect to see the follow response 

```shell
{"error":{"message":"File size is too large. Please check your file size. Passed file size: 0.7392807006835938 MB. Max file size: 0.0001 MB","type":"bad_request","param":"file","code":500}}%  
```