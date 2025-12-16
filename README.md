# Movio – Virtual Companion for visionOS

Movio is a research prototype of a virtual movie companion for **visionOS**.  
It can run as:

- a standalone **visionOS app**, and  
- a backend “brain” for **Unity** via a C-callable bridge.

Movio combines large language models, vision models, speech input/output and mood control into a single assistant.

---

## Key Capabilities

- **Text Chat (GPT-OSS via NVIDIA Integrate)**  
  Natural language conversation about films and theatre, optimized for short, friendly answers.

- **Vision Chat (Qwen-VL)**  
  Multimodal prompts (image + text) for screenshots or stills, routed via Hugging Face or ModelScope.

- **Speech Interface**  
  - Speech-to-Text using Apple’s Speech framework (on-device where required)  
  - Text-to-Speech via `AVSpeechSynthesizer`, with per-mood rate and pitch

- **Mood Steering**  
  Manual tone control (`happy`, `calm`, `sad`, `crying`, `angry`, `anxious`) using lightweight `<style>` blocks and TTS tuning.

- **Optional Web Search (Tavily)**  
  For fresh information (release dates, schedules, etc.), with a short “Fun fact:” line and optional sources.

- **Unity Bridge**  
  C exports such as `movioSetConfig`, `movioAskText`, `movioAskVisionBase64`, `movioStartSTT`, `movioSpeak` so Unity / IL2CPP can drive the assistant.

---

## Architecture Overview

### Model Clients

- `NvidiaChatClient` – OpenAI-compatible client for GPT-OSS (NVIDIA Integrate API)  
- `HFVLMClient` – Vision-language client for Qwen-VL (Hugging Face Router)  
- Optional Qwen3-VL route via ModelScope for models ending with `:modelscope`

### Speech Layer

- `SpeechManager` – wraps `AVSpeechSynthesizer` and the playback audio session  
- `SpeechToTextManager` – manages `AVAudioEngine`, `SFSpeechRecognizer` and permissions

### Tone / Mood

- `MoodTag` enum (`happy`, `calm`, `sad`, `crying`, `angry`, `anxious`)  
- `styleBlock(for:)` – tone instructions passed to the LLM  
- `ttsParams(for:)` – per-mood speech rate and pitch

### Unity Integration

- `MovioUnityBridge` holds shared instances (LLM clients, STT, TTS, config)  
- `@_cdecl` functions expose the bridge to C#/Unity:

  - `movioSetConfig`, `movioSetMood`  
  - `movioAskText`, `movioAskVisionBase64`  
  - `movioStartSTT`, `movioStopSTT`  
  - `movioSpeak`, `movioStopSpeak`

All responses and events are sent back to Unity as simple text tags, e.g.  
`answer:gpt-oss:\n…`, `answer:qwen-vl:\n…`, `stt_partial:…`, `stt_final:…`, `error:…`.

### SwiftUI Frontend

The main UI is implemented in `ContentView` and provides:

- greeting banner and active engine badge  
- mood selector  
- scrollable chat history (user + assistant)  
- input row with:
  - microphone button (push-to-talk)  
  - screenshot picker (`PhotosPicker`)  
  - image URL field  
  - prompt field and Send button  
- utility row:
  - Clear conversation  
  - Repeat last answer  
  - Stop speech and error display

All high-level reasoning, tool use, and tone application are implemented in `ContentView.swift`.

---

## Configuration & API Keys

Movio relies on external services:

- **NVIDIA Integrate** – GPT-OSS models  
- **Hugging Face** – Qwen-VL models  
- **ModelScope** (optional) – Qwen3-VL  
- **Tavily** – web search

In this research version, keys are stored in Swift state properties.  
For any public or shared deployment:

- Replace hard-coded values with placeholders.  
- Load real keys from environment variables, `.xcconfig`, or Keychain.  
- Do not commit live secrets to source control.

---

## Running on visionOS

1. Open the project in Xcode with the **visionOS** SDK installed.  
2. Select a **visionOS simulator** or device target.  
3. Configure your API keys (in code or via a secure mechanism).  
4. Build and run:
   - Text-only prompts use the **GPT-OSS** route.  
   - Prompts plus an image (screenshot or URL) use the **Qwen Vision Agent** route.  
   - The mic button enables push-to-talk; STT auto-fills the prompt and sends it.  
   - “Repeat last answer” replays the most recent TTS output.

---

## Status

Movio is an **experimental research prototype**.  
Models, prompts, and APIs may change as the project evolves, and functionality depends on the availability and performance of third-party services.
