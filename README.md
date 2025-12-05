# Movio – Virtual Companion for visionOS (GPT-OSS + Qwen-VL)

Movio is a virtual companion application for movie and theatre enthusiasts, built for visionOS and designed to be used both:

- as a standalone visionOS app, and  
- as a backend “brain” for a Unity experience (via C-callable bridge functions).

Under the hood it combines:

- GPT-OSS (via NVIDIA Integrate API) for rich text conversations  
- Qwen2.5-VL / Qwen3-VL (via Hugging Face Router / ModelScope) for multimodal (image + text) prompts  
- On-device speech-to-text (Apple Speech framework) and text-to-speech (AVSpeechSynthesizer)  
- Mood steering to change tone (happy / calm / sad / crying / angry / anxious)  
- Optional Tavily web search for fresh information  
- A Unity bridge exposing C functions so Unity / IL2CPP can drive the assistant

This repository contains the Swift side of that system, primarily implemented in `ContentView.swift`.

---

## Architecture Overview

At a high level, the project is composed of:

- Model clients  
  - `NvidiaChatClient` – communicates with GPT-OSS via NVIDIA Integrate API  
  - `HFVLMClient` – communicates with Qwen-VL via Hugging Face Router  
  - A small ModelScope router for Qwen3-VL when the model ends with `:modelscope`
- Speech layer  
  - `SpeechManager` – text-to-speech (TTS)  
  - `SpeechToTextManager` – speech-to-text (STT)
- Tone / mood system  
  - `MoodTag`, `styleBlock(for:)`, `ttsParams(for:)`
- Unity bridge  
  - `MovioUnityBridge` and C-callable wrappers (`movioSetConfig`, `movioAskText`, etc.)
- SwiftUI UI  
  - `ContentView` – main visionOS companion UI

The sections below describe each piece in more detail.

---

## GPT-OSS Text Chat (NVIDIA)

### Data Types

The app uses OpenAI-compatible shapes for GPT-style chat:

```swift
struct ChatMessage: Codable, Identifiable {
    let id = UUID()
    let role: String
    let content: String
}

struct ResponseFormat: Codable { let type: String }

struct ChatRequest: Codable {
    let model: String
    let messages: [ChatMessage]
    let temperature: Double?
    let top_p: Double?
    let max_tokens: Int?
    let stream: Bool?
    let response_format: ResponseFormat?
}

struct ChatChoice: Codable {
    let index: Int?
    let message: ChatMessage?
    let finish_reason: String?
}

struct ChatResponse: Codable {
    let id: String?
    let model: String?
    let choices: [ChatChoice]
}
```

### `NvidiaChatClient`

Located near the top of `ContentView.swift`:

```swift
final class NvidiaChatClient {
    private let endpoint = URL(string: "https://integrate.api.nvidia.com/v1/chat/completions")!

    func send(messages: [ChatMessage],
              apiKey: String,
              model: String = "openai/gpt-oss-20b",
              maxTokens: Int = 1024,
              temperature: Double = 0.3,
              topP: Double = 0.7,
              responseFormat: ResponseFormat? = nil) async throws -> ChatResponse {
        ...
    }
}
```

Responsibilities:

- Build a `ChatRequest` from messages and parameters.  
- Encode to JSON and send a `POST` request.  
- Validate the HTTP status code (must be 2xx).  
- Decode into `ChatResponse`.

The default model is `openai/gpt-oss-20b`, but it can be changed at the call site.

### Usage in UI and Unity

- In `ContentView` (SwiftUI):  
  GPT-OSS is used for standard text-only turns (no image). The text route is selected when there is no screenshot or image URL attached.

- In `MovioUnityBridge.askText`:  
  A short conversation is constructed consisting of:
  - A system message describing Movio’s personality and style.  
  - Two few-shot examples.  
  - The user’s prompt decorated with a mood style block.

  The bridge calls `nvidia.send(...)` and returns a friendly, optionally rewritten answer back to Unity via a callback.

---

## Qwen Vision Chat (Hugging Face Router and ModelScope)

### HF VLM Data Shapes

Vision-language messages use an OpenAI-style schema with `image_url` parts:

```swift
struct HFImageURL: Codable { let url: String }

struct HFContentPart: Codable {
    let type: String           // "text" or "image_url"
    let text: String?
    let image_url: HFImageURL?
}

struct HFMessage: Codable {
    let role: String           // "user", "assistant", "system"
    let content: [HFContentPart]
}

struct HFChatRequest: Codable {
    let messages: [HFMessage]
    let model: String
    let stream: Bool
    let temperature: Double?
    let max_tokens: Int?
    let response_format: ResponseFormat?
}

struct HFChatResponse: Codable {
    struct Choice: Codable {
        struct Message: Codable { let role: String?; let content: String? }
        let index: Int?
        let message: Message?
        let finish_reason: String?
    }
    let choices: [Choice]
}
```

### `HFVLMClient`

This client talks to Hugging Face Router:

```swift
final class HFVLMClient {
    private let endpoint = URL(string: "https://router.huggingface.co/v1/chat/completions")!

    func analyze(imageData: Data,
                 mime: String = "image/jpeg",
                 prompt: String,
                 hfToken: String,
                 model: String = "Qwen/Qwen2.5-VL-72B-Instruct:nebius",
                 maxTokens: Int = 512,
                 temperature: Double = 0.5) async throws -> String { ... }

    func chat(messages: [HFMessage],
              hfToken: String,
              model: String,
              maxTokens: Int = 512,
              temperature: Double = 0.5,
              responseFormat: ResponseFormat? = nil) async throws -> String { ... }
}
```

- `analyze(...)` – convenience method for a single vision prompt (image plus text).  
- `chat(...)` – generic Hugging Face Router `/v1/chat/completions` call for multimodal models such as Qwen-VL.

### ModelScope Router and `vlmChat(...)`

The app optionally routes via ModelScope for Qwen3-VL when the model string ends with `:modelscope`:

- `asOpenAIChatMessages(_:)` – converts `HFMessage` into the `[role, content]` JSON expected by ModelScope.  
- `modelScopeChat(...)` – sends a POST to `https://api-inference.modelscope.cn/v1/chat/completions`.  
- `vlmChat(...)` – chooses between:
  - ModelScope (for `:modelscope` models), or  
  - Hugging Face (`HFVLMClient.chat`) for other models.

These are used internally by the Qwen planning and finalization helpers (see `qwenPlan`, `qwenFinalize`, and the full Qwen agent loop).

---

## Mood and Tone System

The app uses a simple mood tag enum:

```swift
enum MoodTag: String, CaseIterable, Identifiable {
    case happy, calm, sad, crying, angry, anxious
    var id: String { rawValue }
}
```

### `styleBlock(for:)`

Generates a `<style> ToneDirective` block describing how the assistant should sound. Each mood has slightly different instructions (for example, more encouraging if the mood is sad, or more calming if anxious).

### `decorateUserText(_:_:)`

Prepends the style block to the user’s text so the language model treats it as a tone and style guide:

```swift
private func decorateUserText(_ text: String, _ mood: MoodTag) -> String {
    let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !t.isEmpty else { return t }
    return styleBlock(for: mood) + "\n\n" + t
}
```

### `ttsParams(for:)`

Maps a mood to a `(rate, pitch)` pair for TTS so speech matches the selected tone.

---

## Speech-to-Text (STT)

### `SpeechToTextManager`

Encapsulates microphone capture and recognition using Apple’s Speech framework.

Key properties:

- `@Published var isListening` – whether STT is currently active.  
- `@Published var partialText` – live transcript.  
- Callbacks:  
  - `onFinal: ((String) -> Void)?`  
  - `onPartial: ((String) -> Void)?`  
  - `onError: ((String) -> Void)?`

Important methods:

- `ensureAuthorization(_:)`  
  - Requests both speech recognition and microphone permissions.  
  - On visionOS, uses `AVAudioApplication.requestRecordPermission`.  
  - On other platforms, uses `AVAudioSession.sharedInstance().requestRecordPermission`.

- `startListening(autoStopAfterSilence:)`  
  - Sets up an `AVAudioEngine` input tap.  
  - Streams audio to `SFSpeechAudioBufferRecognitionRequest`.  
  - Starts a `SFSpeechRecognitionTask` with partial results enabled.  
  - Uses a timer to auto-stop after a period of silence.

- `stopListening(finalize:)`  
  - Stops recognition and optionally emits the last partial as the final transcript.

Internally, `finish(final:)` cleans up the audio engine, recognition request, task, and timer.

### Mic Toggle in `ContentView`

The `toggleMic()` method integrates STT with the UI:

- If STT is active:
  - Stops STT.  
  - Restores the playback audio session for TTS.

- If STT is inactive:
  - Stops any ongoing TTS to avoid echo.  
  - Requests permissions.  
  - Configures the audio session for `.playAndRecord`.  
  - Sets up STT callbacks:  
    - `onPartial` updates `userInput` live.  
    - `onFinal` fills `userInput`, restores playback, and calls `send()` automatically.  
    - `onError` sets `errorText` and ensures the audio session returns to playback mode.

---

## Text-to-Speech (TTS)

### `SpeechManager`

Simple wrapper around `AVSpeechSynthesizer` and the playback audio session:

```swift
final class SpeechManager: ObservableObject {
    let synth = AVSpeechSynthesizer()
    func configureAudioSession() {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playback, options: [.duckOthers])
            try session.setActive(true)
        } catch {
            print("AudioSession error:", error.localizedDescription)
        }
    }
}
```

### `speakAnswer(_:)` in `ContentView`

Steps:

1. `extractSpeakable(from:)` – trims out sources, additional links, and longer reference sections.  
2. `sanitizeForSpeech(_:)` – removes Markdown formatting, replaces URLs with `(link)`, and collapses excessive whitespace.  
3. Splits into sentences via `splitIntoSentences(_:)`.  
4. Sets voice, rate, and pitch based on the selected mood (`ttsParams(for:)`).  
5. Speaks each sentence with a small delay between them.  
6. Saves `lastSpoken` so the user can tap “Repeat last answer”.

The Unity bridge reuses TTS via `MovioUnityBridge.speak(_:)` and the C export `movioSpeak`.

---

## Tavily Web Search

To provide fresher information when needed, the app integrates Tavily as a web search tool.

### Data Types

```swift
struct TavilyResult: Codable { let title: String?; let url: String?; let content: String? }
struct TavilyResponse: Codable { let answer: String?; let results: [TavilyResult]? }
```

### `tavilySearch(query:)`

- Sends a JSON `POST` to `https://api.tavily.com/search` with:  
  - `api_key`  
  - `query`  
  - `search_depth`  
  - `max_results`  
  - `include_answer`

- Parses the response and formats up to three sources as:

  ```text
  ANSWER:
  <short answer>

  SOURCES:
  - Title
    URL
    snippet
  ```

- Helper functions:  
  - `answerFromSearchResult(_:)` – extracts a natural language answer if present.  
  - `extractSourcesBullets(from:)` – pulls up to three bullet lines in the form `- Title — URL`.

These results can then be provided back into either GPT-OSS or Qwen as additional context for a final answer.

---

## Unity Bridge

The Unity bridge is defined by `MovioUnityBridge` and a set of `@_cdecl` exports.

### `MovioUnityBridge`

This class holds shared components used by Unity:

- `SpeechManager` and `SpeechToTextManager`  
- `NvidiaChatClient` and `HFVLMClient`  
- API keys and model identifiers (`nvidiaKey`, `tavilyKey`, `hfToken`, `hfModel`)  
- Current mood (`mood: MoodTag`)  
- Optional callback `UnityCallback` for sending messages back to Unity:  
  `typealias UnityCallback = @convention(c) (UnsafePointer<CChar>?) -> Void`

Key methods:

- `setConfig(nvidiaKey:tavilyKey:hfToken:hfModel:)`  
- `setMood(_:)`  
- `startSTT()` / `stopSTT(finalize:)` (mirroring the UI mic logic)  
- `speak(_:)` / `stopSpeak()` for TTS  
- `askText(_:)` – GPT-OSS text route  
- `askVision(base64Image:prompt:)` – Qwen-VL vision route

All responses and errors are serialized as simple text messages back to Unity, for example:

- `answer:gpt-oss:
<text>`  
- `answer:qwen-vl:
<text>`  
- `stt_partial:<text>`  
- `stt_final:<text>`  
- `error:<message>`

### C-Callable Exports

These functions expose the bridge to Unity / IL2CPP:

```swift
@_cdecl("movioSetConfig")
public func movioSetConfig(...)

@_cdecl("movioSetMood")
public func movioSetMood(_ moodName: UnsafePointer<CChar>?)

@_cdecl("movioStartSTT")
public func movioStartSTT()

@_cdecl("movioStopSTT")
public func movioStopSTT(_ finalize: Int32)

@_cdecl("movioSpeak")
public func movioSpeak(_ text: UnsafePointer<CChar>?)

@_cdecl("movioStopSpeak")
public func movioStopSpeak()

@_cdecl("movioAskText")
public func movioAskText(_ prompt: UnsafePointer<CChar>?)

@_cdecl("movioAskVisionBase64")
public func movioAskVisionBase64(_ base64Image: UnsafePointer<CChar>?, _ prompt: UnsafePointer<CChar>?)
```

Unity can P/Invoke these functions from C# to control the assistant from within a game or interactive experience.

---

## SwiftUI UI – `ContentView`

### State

`ContentView` manages the main companion UI state:

- API keys and models:  
  - `apiKey` – NVIDIA (GPT-OSS)  
  - `tavilyKey` – Tavily  
  - `hfToken` – Hugging Face  
  - `hfModel` – Qwen-VL model name  
  - `modelScopeToken` – ModelScope token (private)

- Conversation:  
  - `messages: [ChatMessage]` – includes system, user, and assistant messages.  
  - `visibleMessages` – filters out system messages for display.  
  - `userInput` – current text prompt.  
  - `activeEngine` – label for the last engine used (`"GPT-OSS"` or `"QWEN Vision Agent"`).

- Vision:  
  - `selectedItem: PhotosPickerItem?` – selected screenshot from Photos.  
  - `selectedImageData: Data?` – raw data for the selected image.  
  - `imageURLText: String` – URL to a remote image.

- Speech and mood:  
  - `speech: SpeechManager`  
  - `stt: SpeechToTextManager`  
  - `selectedMood: MoodTag`  
  - `lastSpoken: String?`

- Status and focus:  
  - `isSending: Bool` – disables interactions while a request is in flight.  
  - `errorText: String?` – last error message.  
  - `promptFocused`, `urlFocused` – SwiftUI focus states for text fields.

### Layout

Key parts of the UI include:

- Greeting banner and engine badge.  
- Mood selector (`Menu` based on `MoodTag.allCases`).  
- API key and token fields for NVIDIA, Hugging Face, and the HF model string.  
- Scrollable conversation view showing user and assistant messages only.  
- Input row:  
  - Microphone button (push-to-talk, controlled by `toggleMic()`).  
  - Screenshot picker (`PhotosPicker`).  
  - Image URL text field.  
  - Prompt text field.  
  - Send button (uses `canSend` to decide whether it is enabled).

- Utility row:  
  - Clear Conversation (resets the system prompt and conversation).  
  - Repeat last answer.  
  - Stop (cancels TTS).  
  - Error label, if present.

### Logic Helpers

Some notable helpers inside `ContentView`:

- `hasImageInput` – true if there is an attached screenshot or image URL.  
- `canSend` – false while STT is active, a request is in progress, or required keys/tokens are missing.  
- `fetchImageFromURL(...)` – downloads and validates remote images (size, MIME type, etc.).  
- `prepareImageForVLM(...)` – downscales and recompresses image data for Qwen to keep payloads small.  
- `optimizedPayload(for:)` – convenience wrapper for `(Data, mime)` produced by `prepareImageForVLM`.  
- `recentTextContext(...)` – builds a short transcript of recent turns for context sharing between GPT-OSS and Qwen.  
- `applyMovioTone(_:,mood:)` – uses GPT-OSS to rewrite an answer into the Movio tone across routes.  
- `qwenPlan(...)`, `qwenFinalize(...)`, and the Qwen agent loop – implement an image-aware planning and answering flow with optional Tavily search.

---

## Configuration and API Keys

For development, the app stores default keys directly in `@State` properties for convenience.  
For a public repository, real keys should not be committed. Instead:

- Replace default values with placeholders.  
- Load secrets from environment variables, Xcode configuration files, or secure storage such as the Keychain.

Typical keys:

- `apiKey` – NVIDIA Integrate API key (GPT-OSS).  
- `hfToken` – Hugging Face token (vision models).  
- `tavilyKey` – Tavily API key.  
- `modelScopeToken` – ModelScope token (used privately).

---

## Running the App (visionOS)

1. Open the project in Xcode with the visionOS SDK installed.  
2. Select a visionOS Simulator device.  
3. Fill in or paste valid API keys in the UI fields (or configure them in code for local testing).  
4. Run the app:  
   - Type a text prompt and press Send for GPT-OSS text responses.  
   - Attach a screenshot or image URL and press Send for Qwen-VL vision responses.  
   - Use the microphone button for speech input; STT will auto-populate the prompt and auto-send when it detects a pause.  
   - Use the Repeat last answer button to replay the last TTS response.

---

## Notes and Limitations

- This project is intended as a research and demonstration companion application rather than a production system.  
- Long or complex prompts combined with large models may lead to longer latencies or occasional timeouts from third-party APIs.  
- STT uses Apple’s Speech framework and may be limited in simulator environments.  
- TTS and mood steering parameters are tuneable; current values are chosen to represent a casual “movie buddy” style.

---

## License

Add your preferred license text here (for example, MIT or Apache 2.0).
