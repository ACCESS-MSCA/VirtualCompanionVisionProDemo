//
//  ContentView.swift
//  va
//
//  Created by Kyriakos Georgiou on 19/08/2025.
//

import SwiftUI
import AVFoundation
#if os(visionOS)
import AVFAudio
#endif
import Speech   // STT (Speech-to-Text)
import PhotosUI   // for picking a screenshot from Photos
import ImageIO      // for downscaling/compressing screenshots before upload
import UniformTypeIdentifiers   // for UTType.jpeg identifier
import CoreGraphics
import CoreML
internal import Combine
#if canImport(MLX)
import MLX
#endif

// MARK: - OpenAI-compatible shapes
struct ChatMessage: Codable, Identifiable {
    let id = UUID()
    let role: String
    let content: String
    private enum CodingKeys: String, CodingKey { case role, content }
}
struct ResponseFormat: Codable {
    let type: String
}

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

// MARK: - Network Metrics
struct AIRequestMetric {
    let provider: String
    let route: String
    let ttftMs: Int
    let totalMs: Int
    let bytes: Int
    let statusCode: Int?

    var debugLine: String {
        "[AI_METRIC] provider=\(provider) route=\(route) ttft_ms=\(ttftMs) total_ms=\(totalMs) bytes=\(bytes) status=\(statusCode ?? -1)"
    }

    var unityJSON: String {
        let statusValue = statusCode.map(String.init) ?? "null"
        return "{\"provider\":\"\(provider)\",\"route\":\"\(route)\",\"ttft_ms\":\(ttftMs),\"total_ms\":\(totalMs),\"bytes\":\(bytes),\"status_code\":\(statusValue)}"
    }
}

final class AIRequestMetricsCenter {
    static let shared = AIRequestMetricsCenter()
    var onMetric: ((AIRequestMetric) -> Void)?

    private init() {}

    func emit(_ metric: AIRequestMetric) {
        print(metric.debugLine)
        onMetric?(metric)
    }
}

@discardableResult
fileprivate func timedData(for req: URLRequest,
                           provider: String,
                           route: String) async throws -> (Data, URLResponse, AIRequestMetric) {
    let started = Date()
    let (data, resp) = try await URLSession.shared.data(for: req)
    let elapsedMs = Int(Date().timeIntervalSince(started) * 1000)
    let status = (resp as? HTTPURLResponse)?.statusCode
    let metric = AIRequestMetric(provider: provider,
                                 route: route,
                                 ttftMs: elapsedMs,
                                 totalMs: elapsedMs,
                                 bytes: data.count,
                                 statusCode: status)
    AIRequestMetricsCenter.shared.emit(metric)
    return (data, resp, metric)
}

// MARK: - Client
final class NvidiaChatClient {
    private let endpoint = URL(string: "https://integrate.api.nvidia.com/v1/chat/completions")!

    func send(messages: [ChatMessage],
              apiKey: String,
              model: String = "openai/gpt-oss-20b",
              maxTokens: Int = 1024,
              temperature: Double = 0.3,
              topP: Double = 0.7,
              responseFormat: ResponseFormat? = nil) async throws -> ChatResponse {

        var req = URLRequest(url: endpoint)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let body = ChatRequest(model: model,
                               messages: messages,
                               temperature: temperature,
                               top_p: topP,
                               max_tokens: maxTokens,
                               stream: false,
                               response_format: responseFormat)
        req.httpBody = try JSONEncoder().encode(body)

        let (data, resp, _) = try await timedData(for: req, provider: "NVIDIA", route: "/v1/chat/completions")
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let text = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "NVIDIA", code: (resp as? HTTPURLResponse)?.statusCode ?? -1, userInfo: [NSLocalizedDescriptionKey: text])
        }

        let decoded = try JSONDecoder().decode(ChatResponse.self, from: data)
        return decoded
    }
}

// MARK: - HF VLM (Qwen2.5-VL via Hugging Face Router)
// OpenAI-compatible /v1/chat/completions with "image_url" parts (we'll send a data URL).
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

final class HFVLMClient {
    private let endpoint = URL(string: "https://router.huggingface.co/v1/chat/completions")!

    /// Send a prompt + local image (as data URL) to Qwen2.5-VL through HF Router.
    func analyze(imageData: Data,
                 mime: String = "image/jpeg",
                 prompt: String,
                 hfToken: String,
                 model: String = "Qwen/Qwen2.5-VL-72B-Instruct:nebius",
                 maxTokens: Int = 512,
                 temperature: Double = 0.5) async throws -> String {

        // Build a data URL using the provided MIME (e.g., image/jpeg or image/png)
        let dataURL = "data:\(mime);base64," + imageData.base64EncodedString()

        let msg = HFMessage(
            role: "user",
            content: [
                HFContentPart(type: "text", text: prompt, image_url: nil),
                HFContentPart(type: "image_url", text: nil, image_url: HFImageURL(url: dataURL))
            ]
        )

        let body = HFChatRequest(
            messages: [msg],
            model: model,
            stream: false,
            temperature: temperature,
            max_tokens: maxTokens,
            response_format: nil
        )

        var req = URLRequest(url: endpoint)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("Bearer \(hfToken)", forHTTPHeaderField: "Authorization")
        req.httpBody = try JSONEncoder().encode(body)

        let (data, resp, _) = try await timedData(for: req, provider: "HF", route: "/v1/chat/completions:image")
        if let http = resp as? HTTPURLResponse, http.statusCode == 413 {
            throw NSError(domain: "HF", code: 413,
                          userInfo: [NSLocalizedDescriptionKey: "Image too large for server. I reduced size automatically—please retry. If it persists, try a smaller screenshot."])
        }
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let text = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "HF", code: (resp as? HTTPURLResponse)?.statusCode ?? -1,
                          userInfo: [NSLocalizedDescriptionKey: text])
        }

        let decoded = try JSONDecoder().decode(HFChatResponse.self, from: data)
        return decoded.choices.first?.message?.content?.trimmingCharacters(in: .whitespacesAndNewlines)
               ?? (String(data: data, encoding: .utf8) ?? "")
    }
    
    /// Generic OpenAI-compatible chat call for HF Router (supports system messages and response_format).
    func chat(messages: [HFMessage],
              hfToken: String,
              model: String,
              maxTokens: Int = 512,
              temperature: Double = 0.5,
              responseFormat: ResponseFormat? = nil) async throws -> String {
        let body = HFChatRequest(messages: messages,
                                 model: model,
                                 stream: false,
                                 temperature: temperature,
                                 max_tokens: maxTokens,
                                 response_format: responseFormat)
        var req = URLRequest(url: endpoint)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("Bearer \(hfToken)", forHTTPHeaderField: "Authorization")
        req.httpBody = try JSONEncoder().encode(body)

        let (data, resp, _) = try await timedData(for: req, provider: "HF", route: "/v1/chat/completions")
        if let http = resp as? HTTPURLResponse, http.statusCode == 413 {
            throw NSError(domain: "HF", code: 413,
                          userInfo: [NSLocalizedDescriptionKey: "Image too large for server. I reduced size automatically—please retry. If it persists, try a smaller screenshot."])
        }
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let text = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "HF", code: (resp as? HTTPURLResponse)?.statusCode ?? -1,
                          userInfo: [NSLocalizedDescriptionKey: text])
        }
        let decoded = try JSONDecoder().decode(HFChatResponse.self, from: data)
        return decoded.choices.first?.message?.content?.trimmingCharacters(in: .whitespacesAndNewlines)
               ?? (String(data: data, encoding: .utf8) ?? "")
    }
}

// MARK: - Speech
final class SpeechManager: ObservableObject {
    let synth = AVSpeechSynthesizer()
    
    func configureAudioSession() {
        let session = AVAudioSession.sharedInstance()
        do {
            try? session.setActive(false, options: .notifyOthersOnDeactivation) // <- importante
            try session.setCategory(.playAndRecord, mode: .measurement, options: [])
            try session.setActive(true)
           // try session.setCategory(.playback, options: [.duckOthers])
           // try session.setActive(true)
        } catch {
            print("AudioSession error:", error.localizedDescription)
        }
    }
}

#if os(visionOS)
fileprivate func mcRequestMicPermission(_ handler: @escaping (Bool) -> Void) {
    // visionOS: use AVAudioApplication instead of AVAudioSession
    AVAudioApplication.requestRecordPermission(completionHandler: handler)
}
#else
fileprivate func mcRequestMicPermission(_ handler: @escaping (Bool) -> Void) {
    // iOS/macOS/etc: keep using AVAudioSession
    AVAudioSession.sharedInstance().requestRecordPermission(handler)
}
#endif

// MARK: - STT (Speech to Text)
/// Handles microphone capture + on-device Speech framework recognition.
/// Produces partial transcripts, stops automatically after brief silence, and emits a final text callback.
final class SpeechToTextManager: NSObject, ObservableObject {
    @Published var isListening = false
    @Published var partialText: String = ""            // live transcript for UI (optional)
    var onFinal: ((String) -> Void)?                   // callback for the final text
    var onPartial: ((String) -> Void)?                 // callback for partial recognition
    var onError: ((String) -> Void)?                   // callback for errors (permissions, session, recognizer, etc.)

    private var recognizer: SFSpeechRecognizer?
    private let audioEngine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var silenceTimer: Timer?

    override init() {
        super.init()
        // Use the system default locale so we match whatever on-device assets are installed.
        // This avoids failing when a specific locale (for example en-GB) does not have local speech models.
        let rec = SFSpeechRecognizer()
        rec?.delegate = self
        self.recognizer = rec
    }

    /// Ask for both speech recognition and microphone permissions.
    /// Ask for both speech recognition and microphone permissions.
    func ensureAuthorization(_ completion: @escaping (Bool) -> Void) {
        SFSpeechRecognizer.requestAuthorization { status in
            let speechOK = (status == .authorized)

            mcRequestMicPermission { micOK in
                DispatchQueue.main.async {
                    if !(speechOK && micOK) {
                        let msg = !speechOK
                            ? "Speech recognition permission denied."
                            : "Microphone permission denied."
                        self.onError?(msg)
                    }
                    completion(speechOK && micOK)
                }
            }
        }
    }

    /// Begin listening and streaming audio into the recognizer.
    /// Automatically stops after `seconds` of inactivity.
    func startListening(autoStopAfterSilence seconds: TimeInterval = 1.2) throws {
        guard task == nil, !audioEngine.isRunning else { return }
        partialText = ""

        // Ensure recognizer exists and is available for this locale/device
        guard let recognizer else {
            onError?("Speech recognizer not available for this locale.")
            return
        }
        guard recognizer.isAvailable else {
            onError?("Speech recognizer is currently unavailable.")
            return
        }

        // Create the streaming request (report partial results for live UI)
        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
#if os(visionOS) && !targetEnvironment(simulator)
        // On visionOS hardware, Apple requires on-device models; request on-device recognition explicitly.
        req.requiresOnDeviceRecognition = true
#endif
        self.request = req

        // Install a tap on the input node
        let input = audioEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        audioEngine.stop()
        audioEngine.reset()
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.request?.append(buffer)
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            onError?("Audio engine failed to start: \(error.localizedDescription)")
            return
        }
        DispatchQueue.main.async { self.isListening = true }
        print("STT: Listening started")

        // Start recognition task
        task = recognizer.recognitionTask(with: req) { [weak self] result, error in
            guard let self = self else { return }
            if let res = result {
                let text = res.bestTranscription.formattedString
                DispatchQueue.main.async {
                    self.partialText = text
                    self.onPartial?(text)
                }
                self.resetSilenceTimer(seconds)
                if res.isFinal {
                    print("STT: Final transcript: \(text)")
                    self.finish(final: text)
                }
            }
            if let e = error {
                let lower = e.localizedDescription.lowercased()
                // Ignore expected cancellations when we stop listening ourselves
                if lower.contains("canceled") || lower.contains("cancelled") {
                    print("STT: Recognition canceled (expected on stop)")
                } else if lower.contains("failed to access assets") {
                    print("STT: On-device speech assets not available on this device/simulator.")
#if targetEnvironment(simulator)
                    self.onError?("Speech recognition assets are not available in the simulator. Please test on a real device.")
#else
                    self.onError?("Speech recognition assets are not available for this language. Check on-device speech settings.")
#endif
                } else {
                    print("STT: Recognition error:", e.localizedDescription)
                    self.onError?("Recognition error: \(e.localizedDescription)")
                }
                self.finish(final: nil)
            }
        }

        // Kick off silence timer
        resetSilenceTimer(seconds)
    }
    
    func transcribeFile(atPath path: String) {
        // Limpia estado anterior
        task?.cancel()
        task = nil
        request = nil
        partialText = ""

        guard let recognizer else {
            onError?("Speech recognizer not available.")
            return
        }
        guard recognizer.isAvailable else {
            onError?("Speech recognizer is currently unavailable.")
            return
        }

        let url = URL(fileURLWithPath: path)
        if !FileManager.default.fileExists(atPath: url.path) {
            onError?("Audio file not found at path: \(url.path)")
            return
        }

        let req = SFSpeechURLRecognitionRequest(url: url)
        req.shouldReportPartialResults = true

        DispatchQueue.main.async { self.isListening = true }
        print("STT: Transcribing file:", url.lastPathComponent)

        task = recognizer.recognitionTask(with: req) { [weak self] result, error in
            guard let self else { return }

            if let res = result {
                let text = res.bestTranscription.formattedString
                DispatchQueue.main.async {
                    self.partialText = text
                    self.onPartial?(text)
                }
                if res.isFinal {
                    print("STT: Final transcript (file):", text)
                    self.finish(final: text)
                }
            }

            if let e = error {
                let lower = e.localizedDescription.lowercased()
                if lower.contains("canceled") || lower.contains("cancelled") {
                    print("STT: canceled (expected)")
                } else {
                    print("STT file error:", e.localizedDescription)
                    self.onError?("Recognition error: \(e.localizedDescription)")
                }
                self.finish(final: nil)
            }
        }
    }

    /// Stop listening. If `finalize` is true, emit the last partial as final.
    func stopListening(finalize: Bool = true) {
        if finalize,
           !partialText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            finish(final: partialText)
        } else {
            finish(final: nil)
        }
    }

    // MARK: - Internals
    private func finish(final text: String?) {
        print("STT: Finishing (final=\(text != nil))")
        silenceTimer?.invalidate(); silenceTimer = nil
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        request?.endAudio()
        task?.cancel()
        task = nil
        request = nil
        let shouldEmit = isListening
        DispatchQueue.main.async {
            self.isListening = false
            if let t = text, shouldEmit {
                self.onFinal?(t)
            }
        }
    }

    private func resetSilenceTimer(_ seconds: TimeInterval) {
        silenceTimer?.invalidate()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: seconds, repeats: false) { [weak self] _ in
            self?.stopListening(finalize: true)
        }
    }
}

extension SpeechToTextManager: SFSpeechRecognizerDelegate {}

// MARK: - Mood steering (manual selection)
enum MoodTag: String, CaseIterable, Identifiable {
    case happy, calm, sad, crying, angry, anxious
    var id: String { rawValue }
}

enum VoiceGender: String, CaseIterable, Identifiable {
    case male, female, neutral
    var id: String { rawValue }
}

private func styleBlock(for mood: MoodTag) -> String {
    switch mood {
    case .happy:
        return """
        <style>
        ToneDirective: user feels happy — keep it upbeat; add one brief celebratory line after the main answer. Use a friendly, conversational voice with natural contractions. Start with a light opener if it fits (Yeah—, Oh totally—). Keep it brief (2–4 sentences). Do not echo this block.
        </style>
        """.trimmingCharacters(in: .whitespacesAndNewlines)
    case .calm:
        return """
        <style>
        ToneDirective: user feels calm — be friendly and relaxed; plain language; no corporate tone. Use natural contractions. Keep it brief (2–4 sentences). Do not echo this block.
        </style>
        """.trimmingCharacters(in: .whitespacesAndNewlines)
    case .sad:
        return """
        <style>
        ToneDirective: user feels sad — be warm and human; add one brief encouraging line after the main answer. Use a gentle, conversational voice with natural contractions. Keep it brief (2–4 sentences). Do not echo this block.
        </style>
        """.trimmingCharacters(in: .whitespacesAndNewlines)
    case .crying:
        return """
        <style>
        ToneDirective: user is very upset — be gentle and kind; add one soft supportive line after the main answer. Keep wording simple and conversational with natural contractions. Keep it brief (2–4 sentences). Do not echo this block.
        </style>
        """.trimmingCharacters(in: .whitespacesAndNewlines)
    case .angry:
        return """
        <style>
        ToneDirective: user feels frustrated — de-escalate and be practical; add one next step after the main answer. Use a calm, conversational voice with natural contractions. Keep it brief (2–4 sentences). Do not echo this block.
        </style>
        """.trimmingCharacters(in: .whitespacesAndNewlines)
    case .anxious:
        return """
        <style>
        ToneDirective: user feels anxious — be reassuring and clear; add one simple next step after the main answer. Use a friendly, conversational voice with natural contractions. Keep it brief (2–4 sentences). Do not echo this block.
        </style>
        """.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private func decorateUserText(_ text: String, _ mood: MoodTag) -> String {
    let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !t.isEmpty else { return t }
    // Prepend the style so the model attends to it first.
    return styleBlock(for: mood) + "\n\n" + t
}

private func ttsParams(for mood: MoodTag) -> (rate: Float, pitch: Float) {
    switch mood {
    case .happy:   return (AVSpeechUtteranceDefaultSpeechRate * 0.95, 1.10)
    case .calm:    return (AVSpeechUtteranceDefaultSpeechRate * 0.85, 1.05)
    case .sad:     return (AVSpeechUtteranceDefaultSpeechRate * 0.80, 0.98)
    case .crying:  return (AVSpeechUtteranceDefaultSpeechRate * 0.78, 0.95)
    case .angry:   return (AVSpeechUtteranceDefaultSpeechRate * 0.90, 1.00)
    case .anxious: return (AVSpeechUtteranceDefaultSpeechRate * 0.82, 1.02)
    }
}

// MARK: - Helpers used by the Unity bridge (top-level, file scope)
fileprivate func mcSplitIntoSentences(_ text: String) -> [String] {
    var out: [String] = []
    var current = ""
    for ch in text {
        current.append(ch)
        if ".!?".contains(ch) {
            let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { out.append(trimmed) }
            current = ""
        }
    }
    let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
    if !tail.isEmpty { out.append(tail) }
    return out
}

// Generic readability/shape heuristic: returns true when a friendlier rewrite is likely helpful.
fileprivate func shouldFriendlyRewrite(_ text: String) -> Bool {
    let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
    if t.isEmpty { return false }

    // Too long for a snappy buddy tone
    if t.count > 480 { return true }

    // Average words per sentence
    let sentences = mcSplitIntoSentences(t)
    let words = t.split { !$0.isLetter && !$0.isNumber && $0 != "'" }
    let avgWords = Double(words.count) / Double(max(1, sentences.count))
    if avgWords > 22.0 { return true }

    // Lots of newlines => list/reporty
    let newlineCount = t.filter { $0 == "\n" }.count
    if newlineCount >= 3 { return true }

    // URLs or Sources block usually indicate formal/report tone
    if t.range(of: #"https?://"#, options: .regularExpression) != nil { return true }
    if t.range(of: "sources:", options: .caseInsensitive) != nil { return true }

    // Generic formal cues
    let lowers = t.lowercased()
    let formalCues = ["according to", "in conclusion", "in summary", "the following"]
    if formalCues.contains(where: { lowers.contains($0) }) { return true }

    return false
}

// MARK: - Unity Bridge (C-callable wrappers using @_cdecl)
// C# delegate type: void SwiftCallback(const char* message)
public typealias UnityCallback = @convention(c) (UnsafePointer<CChar>?) -> Void
public typealias UnityBytesCallback = @convention(c) (UnsafePointer<UInt8>?, Int32) -> Void

final class MovioUnityBridge: NSObject, AVSpeechSynthesizerDelegate {
    static let shared = MovioUnityBridge()

    // Basic components reused from this file
    let speech = SpeechManager()
    let stt    = SpeechToTextManager()
    let nvidia = NvidiaChatClient()
    let hf     = HFVLMClient()
    
    // Config
    var nvidiaKey: String = ""
    var tavilyKey: String = ""   // reserved; not used in this minimal bridge
    var hfToken: String   = ""
    var hfModel: String   = "Qwen/Qwen2.5-VL-72B-Instruct:nebius"

    // Mood used to decorate prompts and TTS parameters
    var mood: MoodTag = .calm

    // Voice selection used by Unity-facing TTS controls
    var voiceGender: VoiceGender = .neutral
    var voicePitch: Float = 1.0
    var voiceLanguage: String = "en-GB"
    var voiceIdentifier: String? = nil

    // Optional callback back into Unity
    private var unityCallback: UnityCallback?
    private var unityBytesCallback: UnityBytesCallback?
    override init() {
        super.init()
        // STT -> Unity forwarding (dispatch to main to be safe with Unity APIs)
        stt.onPartial = { [weak self] text in self?.sendToUnity("stt_partial:" + text) }
        stt.onFinal   = { [weak self] text in self?.sendToUnity("stt_final:" + text) }
        stt.onError   = { [weak self] msg  in self?.sendToUnity("error:" + msg) }
        // Ensure playback session is ready for TTS
        speech.configureAudioSession()
        speech.synth.delegate = self
        AIRequestMetricsCenter.shared.onMetric = { [weak self] metric in
            self?.sendMetricToUnity(metric)
        }
    }

    // MARK: - Unity callback wiring
    func registerCallback(_ cb: UnityCallback?) { self.unityCallback = cb }
    fileprivate func sendToUnity(_ message: String) {
        guard let cb = unityBytesCallback else { return }

        let utf8 = Array(message.utf8)
        let len = utf8.count

        let ptr = UnsafeMutablePointer<UInt8>.allocate(capacity: len)
        ptr.initialize(from: utf8, count: len)

        DispatchQueue.main.async {
            cb(ptr, Int32(len))
            ptr.deallocate()
        }
    }


    private func sendMetricToUnity(_ metric: AIRequestMetric) {
        sendToUnity("metric:" + metric.unityJSON)
        sendToUnity("metric_ttft:" + metric.provider + ":" + String(metric.ttftMs))
        sendToUnity("metric_total:" + metric.provider + ":" + String(metric.totalMs))
    }
    func sttFromFile(path: String) {
        stt.ensureAuthorization { [weak self] ok in
            guard let self else { return }
            guard ok else {
                self.sendToUnity("error:Speech permission is required.")
                return
            }
            self.stt.onPartial = { [weak self] t in self?.sendToUnity("stt_partial:" + t) }
            self.stt.onFinal   = { [weak self] t in self?.sendToUnity("stt_final:" + t) }
            self.stt.onError   = { [weak self] e in self?.sendToUnity("error:" + e) }

            self.stt.transcribeFile(atPath: path)
        }
    }

    private var ttsPending = 0
        private var ttsBatchId = UUID()
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                           didStart utterance: AVSpeechUtterance) {
        //sendToUnity("tts_started")
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                           didFinish utterance: AVSpeechUtterance) {
        // Solo avisamos cuando ya terminó TODO
        ttsPending = max(0, ttsPending - 1)
                if ttsPending == 0 {
                    sendToUnity("tts_finished")
                }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                           didCancel utterance: AVSpeechUtterance) {
        ttsPending = 0
        sendToUnity("tts_cancelled")
        sendToUnity("tts_finished")
    }
    //private func sendToUnity(_ message: String) {
       // guard let cb = unityCallback else { return }
        // Ensure callbacks land on the main thread for Unity safety
        
        //DispatchQueue.main.async {
        //    message.withCString { cStr in cb(cStr) }
        //}
    //}

    func registerBytesCallback(_ cb: UnityBytesCallback?) { self.unityBytesCallback = cb }

    // MARK: - Public operations
    func setConfig(nvidiaKey: String, tavilyKey: String, hfToken: String, hfModel: String?) {
        self.nvidiaKey = nvidiaKey
        self.tavilyKey = tavilyKey
        self.hfToken   = hfToken
        if let m = hfModel, !m.isEmpty { self.hfModel = m }
    }

    func setMood(_ m: MoodTag) { self.mood = m }

    func setVoiceGender(_ gender: VoiceGender) {
        self.voiceGender = gender
    }

    func setVoicePitch(_ pitch: Float) {
        self.voicePitch = max(0.5, min(pitch, 2.0))
    }

    func setVoiceLanguage(_ language: String) {
        let trimmed = language.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            self.voiceLanguage = trimmed
        }
    }

    func setVoiceIdentifier(_ identifier: String?) {
        let trimmed = identifier?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        self.voiceIdentifier = trimmed.isEmpty ? nil : trimmed
    }

    func availableVoicesSummary() -> String {
        AVSpeechSynthesisVoice.speechVoices()
            .map { voice in
                let name = voice.name
                let lang = voice.language
                let id = voice.identifier
                let quality: String
                switch voice.quality {
                case .default: quality = "default"
                case .enhanced: quality = "enhanced"
                @unknown default: quality = "unknown"
                }
                return "\(name) | \(lang) | \(id) | \(quality)"
            }
            .joined(separator: "\n")
    }

    private func resolveVoice() -> AVSpeechSynthesisVoice? {
        if let identifier = voiceIdentifier,
           let exactVoice = AVSpeechSynthesisVoice(identifier: identifier) {
            return exactVoice
        }

        let preferredLanguage = voiceLanguage.trimmingCharacters(in: .whitespacesAndNewlines)
        let fallbackLanguage = AVSpeechSynthesisVoice.currentLanguageCode()
        let allVoices = AVSpeechSynthesisVoice.speechVoices()

        let languageMatched = allVoices.filter {
            $0.language.caseInsensitiveCompare(preferredLanguage) == .orderedSame
        }
        let regionalMatched = languageMatched.isEmpty
            ? allVoices.filter { $0.language.lowercased().hasPrefix(preferredLanguage.lowercased().prefix(2)) }
            : languageMatched
        let primaryPool = regionalMatched.isEmpty ? allVoices : regionalMatched

        func genderScore(for voice: AVSpeechSynthesisVoice) -> Int {
            let name = voice.name.lowercased()
            switch voiceGender {
            case .male:
                if name.contains("male") { return 3 }
                if name.contains("man") { return 2 }
                return 0
            case .female:
                if name.contains("female") { return 3 }
                if name.contains("woman") { return 2 }
                return 0
            case .neutral:
                return 0
            }
        }

        func qualityScore(for voice: AVSpeechSynthesisVoice) -> Int {
            switch voice.quality {
            case .enhanced: return 1
            default: return 0
            }
        }

        if voiceGender != .neutral,
           let gendered = primaryPool
            .sorted(by: {
                let left = genderScore(for: $0)
                let right = genderScore(for: $1)
                if left != right { return left > right }
                let leftQuality = qualityScore(for: $0)
                let rightQuality = qualityScore(for: $1)
                if leftQuality != rightQuality { return leftQuality > rightQuality }
                return $0.name < $1.name
            })
            .first(where: { genderScore(for: $0) > 0 }) {
            return gendered
        }

        if let preferred = primaryPool.sorted(by: {
            let leftQuality = qualityScore(for: $0)
            let rightQuality = qualityScore(for: $1)
            if leftQuality != rightQuality { return leftQuality > rightQuality }
            return $0.name < $1.name
        }).first {
            return preferred
        }

        if let fallback = AVSpeechSynthesisVoice(language: preferredLanguage) {
            return fallback
        }
        if let current = AVSpeechSynthesisVoice(language: fallbackLanguage) {
            return current
        }
        return AVSpeechSynthesisVoice(language: "en-GB") ?? AVSpeechSynthesisVoice(language: "en-US")
    }

    // STT control
    func startSTT() {
        // Stop TTS to avoid echo
        speech.synth.stopSpeaking(at: .immediate)
        stt.ensureAuthorization { [weak self] ok in
            guard let self else { return }
            guard ok else { self.sendToUnity("error:Microphone/Speech permission is required."); return }
            do {
                try self.stt.startListening(autoStopAfterSilence: 1.6)
            } catch {
                self.sendToUnity("error:Could not start microphone: \(error.localizedDescription)")
            }
        }
    }
    func stopSTT(finalize: Bool) {
        stt.stopListening(finalize: finalize)
        // Return to playback session for TTS
        speech.configureAudioSession()
    }

    // TTS
    func speak(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        // Nueva tanda
        ttsBatchId = UUID()
        let batch = ttsBatchId
        
        // Si estabas hablando, cancela y avisa (opcional pero recomendado)
        if speech.synth.isSpeaking || speech.synth.isPaused {
            speech.synth.stopSpeaking(at: .immediate)
            // no envíes finished aquí; llegará didCancel / didFinish
        }
        
        let sentences = mcSplitIntoSentences(trimmed)
        ttsPending = sentences.count
        sendToUnity("tts_started")
        
        let voice = resolveVoice()
        let (rate, moodPitch) = ttsParams(for: mood)
        let finalPitch = max(0.5, min(voicePitch * moodPitch, 2.0))
        speech.synth.stopSpeaking(at: .immediate)
        for (i, sentence) in mcSplitIntoSentences(trimmed).enumerated() {
            let u = AVSpeechUtterance(string: sentence)
            u.voice = voice
            u.rate  = rate
            u.pitchMultiplier = finalPitch
            u.postUtteranceDelay = (i == 0) ? 0.0 : 0.12
            speech.synth.speak(u)
        }
        // Si por algún motivo la cola quedó vacía (raro), fuerza final
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            guard let self else { return }
            if self.ttsBatchId == batch, self.ttsPending == 0 {
                self.sendToUnity("tts_finished")
            }
        }
    }
    func stopSpeak() { speech.synth.stopSpeaking(at: .immediate) }

    // GPT-OSS text (minimal, no tool/search step here)
    func askText(_ prompt: String) {
        Task {
            do {
                let user = styleBlock(for: mood) + "\n\n" + prompt
                var msgs: [ChatMessage] = [
                    .init(role: "system", content:
                          """
                          You are Movio, a friendly movie-buff friend. Speak casually with natural contractions and short openers when it feels right (Yeah—, Oh totally—). Keep it brief (2–4 sentences), direct, and warm. Use PLAIN TEXT only (no Markdown). If a <style> ToneDirective block is present in the user message, follow it in the final prose.
                          """
                    )
                ]
                // Few-shot style seeds
                msgs.append(.init(role: "user", content: "Best stage-to-film version of Macbeth?"))
                msgs.append(.init(role: "assistant", content: "I’d go with the 2015 one with Michael Fassbender—moody, brutal, and gorgeous. If you want something bolder, the Coen version with Denzel Washington is stark and theatrical. Your call: classic grit or minimalist chill."))
                msgs.append(.init(role: "user", content: "Feel-good musical tonight?"))
                msgs.append(.init(role: "assistant", content: "Go La La Land if you want dreamy jazz and bittersweet smiles. If you want pure sparkle, The Greatest Showman is popcorn joy. Tell me your vibe and I’ll dial it in."))
                // Actual turn
                msgs.append(.init(role: "user", content: user))
                let res = try await nvidia.send(messages: msgs,
                                                apiKey: nvidiaKey,
                                                model: "openai/gpt-oss-20b",
                                                maxTokens: 480,
                                                temperature: 0.65,
                                                topP: 0.95,
                                                responseFormat: nil)
                var answer = (res.choices.first?.message?.content ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                print("REspuesta de nvidia: \(answer)")
                if !answer.isEmpty, shouldFriendlyRewrite(answer) {
                    if let rewritten = try? await friendlyRewrite(answer, mood: self.mood), !rewritten.isEmpty {
                        answer = rewritten
                    }
                }
                if answer.isEmpty { self.sendToUnity("answer:gpt-oss:\nSorry — I couldn’t generate an answer.") }
                else { self.sendToUnity(/*"answer:gpt-oss:" + */answer) }
            } catch {
                self.sendToUnity("error:" + error.localizedDescription)
            }
        }
    }

    // Heuristic: when should we rewrite to a friendlier tone?
    private func shouldFriendlyRewrite(_ text: String) -> Bool {
        let t = text.lowercased()
        if t.contains("sources:") { return true }
        if t.contains("according to") { return true }
        if t.contains("is scheduled") { return true }
        if t.contains("in conclusion") || t.contains("in summary") { return true }
        if t.contains("the film is") || t.contains("the movie is") { return true }
        if t.count > 420 { return true } // too long for a snappy buddy tone
        return false
    }

    // One-shot style pass with GPT‑OSS (fast, ~100–150 tokens)
    private func friendlyRewrite(_ text: String, mood: MoodTag) async throws -> String {
        let sys = """
        Rewrite the user's answer so it sounds like a friendly movie‑buff friend. Keep all facts. Plain text only (no markdown). Use natural contractions and a light opener if it fits (Yeah—, Oh totally—). Limit to 2–4 short sentences. Drop any formal phrasing (no "According to", no report tone). Do not include a Sources section unless explicitly asked. If a <style> ToneDirective block appears, follow it.
        """
        let msgs: [ChatMessage] = [
            .init(role: "system", content: sys),
            .init(role: "user", content: styleBlock(for: mood) + "\n\n" + text)
        ]
        let res = try await nvidia.send(messages: msgs,
                                        apiKey: nvidiaKey,
                                        model: "openai/gpt-oss-20b",
                                        maxTokens: 220,
                                        temperature: 0.7,
                                        topP: 0.95,
                                        responseFormat: nil)
        return (res.choices.first?.message?.content ?? text).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // Qwen vision (expects base64 JPEG/PNG string from Unity)
    func askVision(base64Image: String, prompt: String) {
        Task {
            do {
                guard let data = Data(base64Encoded: base64Image) else {
                    self.sendToUnity("error:Invalid base64 image data"); return
                }
                let dataURL = "data:image/jpeg;base64," + data.base64EncodedString() // assume JPEG; works for PNG too
                let system = "Write the final answer as Movio. Start conversational and human — natural contractions and a short opener if it fits. Keep it brief (2–4 sentences), plain text only (no markdown). Then add exactly one line: Fun fact: …"
                let msgs: [HFMessage] = [
                    HFMessage(role: "system", content: [HFContentPart(type: "text", text: system, image_url: nil)]),
                    HFMessage(role: "user", content: [
                        HFContentPart(type: "text", text: styleBlock(for: mood), image_url: nil),
                        HFContentPart(type: "text", text: prompt, image_url: nil),
                        HFContentPart(type: "image_url", text: nil, image_url: HFImageURL(url: dataURL))
                    ])
                ]
                let text = try await hf.chat(messages: msgs,
                                             hfToken: hfToken,
                                             model: hfModel,
                                             maxTokens: 512,
                                             temperature: 0.5,
                                             responseFormat: nil)
                let answer = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if answer.isEmpty { self.sendToUnity("answer:qwen-vl:\nSorry — I couldn’t generate an answer.") }
                else { self.sendToUnity("answer:qwen-vl:\n" + answer) }
            } catch {
                self.sendToUnity("error:" + error.localizedDescription)
            }
        }
    }
}

// MARK: - C-callable exports for Unity (IL2CPP)

@_cdecl("registerUnityCallback")
public func registerUnityCallback(_ cb: UnityCallback?) {
    MovioUnityBridge.shared.registerCallback(cb)
}

@_cdecl("movioSetConfig")
public func movioSetConfig(_ nvidiaKey: UnsafePointer<CChar>?,
                           _ tavilyKey: UnsafePointer<CChar>?,
                           _ hfToken: UnsafePointer<CChar>?,
                           _ hfModel: UnsafePointer<CChar>?) {
    let n = nvidiaKey.flatMap { String(cString: $0) } ?? ""
    let t = tavilyKey.flatMap { String(cString: $0) } ?? ""
    let h = hfToken.flatMap  { String(cString: $0) } ?? ""
    let m = hfModel.flatMap  { String(cString: $0) }
    MovioUnityBridge.shared.setConfig(nvidiaKey: n, tavilyKey: t, hfToken: h, hfModel: m)
}


@_cdecl("movioSetMood")
public func movioSetMood(_ moodName: UnsafePointer<CChar>?) {
    let name = (moodName.flatMap { String(cString: $0) } ?? "calm").lowercased()
    let map: [String: MoodTag] = ["happy": .happy, "calm": .calm, "sad": .sad, "crying": .crying, "angry": .angry, "anxious": .anxious]
    MovioUnityBridge.shared.setMood(map[name] ?? .calm)
}

@_cdecl("movioSetVoiceGender")
public func movioSetVoiceGender(_ genderName: UnsafePointer<CChar>?) {
    let name = (genderName.flatMap { String(cString: $0) } ?? "neutral").lowercased()
    let map: [String: VoiceGender] = ["male": .male, "female": .female, "neutral": .neutral]
    MovioUnityBridge.shared.setVoiceGender(map[name] ?? .neutral)
}

@_cdecl("movioSetVoicePitch")
public func movioSetVoicePitch(_ pitch: Float) {
    MovioUnityBridge.shared.setVoicePitch(pitch)
}

@_cdecl("movioSetVoiceLanguage")
public func movioSetVoiceLanguage(_ language: UnsafePointer<CChar>?) {
    let value = language.flatMap { String(cString: $0) } ?? ""
    MovioUnityBridge.shared.setVoiceLanguage(value)
}

@_cdecl("movioSetVoiceIdentifier")
public func movioSetVoiceIdentifier(_ identifier: UnsafePointer<CChar>?) {
    let value = identifier.flatMap { String(cString: $0) }
    MovioUnityBridge.shared.setVoiceIdentifier(value)
}

@_cdecl("movioListVoices")
public func movioListVoices() {
    let summary = MovioUnityBridge.shared.availableVoicesSummary()
    MovioUnityBridge.shared.sendToUnity("voices:" + summary)
}

@_cdecl("movioStartSTT")
public func movioStartSTT() { MovioUnityBridge.shared.startSTT() }

@_cdecl("movioStopSTT")
public func movioStopSTT(_ finalize: Int32) { MovioUnityBridge.shared.stopSTT(finalize: finalize != 0) }

@_cdecl("movioSpeak")
public func movioSpeak(_ text: UnsafePointer<CChar>?) {
    let s = text.flatMap { String(cString: $0) } ?? ""
    MovioUnityBridge.shared.speak(s)
}

@_cdecl("movioStopSpeak")
public func movioStopSpeak() { MovioUnityBridge.shared.stopSpeak() }

@_cdecl("movioAskText")
public func movioAskText(_ prompt: UnsafePointer<CChar>?) {
    let p = prompt.flatMap { String(cString: $0) } ?? ""
    MovioUnityBridge.shared.askText(p)
}

@_cdecl("movioAskVisionBase64")
public func movioAskVisionBase64(_ base64Image: UnsafePointer<CChar>?, _ prompt: UnsafePointer<CChar>?) {
    let b64 = base64Image.flatMap { String(cString: $0) } ?? ""
    let p   = prompt.flatMap      { String(cString: $0) } ?? ""
    MovioUnityBridge.shared.askVision(base64Image: b64, prompt: p)
}

@_cdecl("registerUnityBytesCallback")
public func registerUnityBytesCallback(_ cb: UnityBytesCallback?) {
    MovioUnityBridge.shared.registerBytesCallback(cb)
}

@_cdecl("movioSTTFromFile")
public func movioSTTFromFile(_ path: UnsafePointer<CChar>?) {
    let p = path.flatMap { String(cString: $0) } ?? ""
    MovioUnityBridge.shared.sttFromFile(path: p)
}


// MARK: - UI
struct ContentView: View {
    private var hasImageInput: Bool {
        return selectedImageData != nil || !imageURLText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var canSend: Bool {
        if isSending || stt.isListening { return false }
        if hasImageInput {
            // Qwen-VL hosted path only
            // Hosted path: require token by provider (HF vs ModelScope). ModelScope token is stored privately.
            let needsModelScope = hfModel.lowercased().hasSuffix(":modelscope")
            let tokenOK = needsModelScope
                ? !modelScopeToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                : !hfToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            return tokenOK && !hfModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        } else {
            // GPT-OSS text route
            return !apiKey.isEmpty &&
                   !userInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }
    
    @State private var apiKey: String = "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"          // Paste your key for now (we’ll move to Keychain next step)
    @State private var tavilyKey: String = "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"
    @State private var userInput: String = ""
    @State private var messages: [ChatMessage] = [
        .init(role: "system", content:
              """
              You are Movio, an assistant for movie & theatre lovers—and for people currently watching a film or show. Speak like a friendly movie-buff friend: casual, warm, and concise. Use natural contractions and short openers when it fits (Yeah—, Oh totally—). Keep answers brief (2–4 sentences) unless the user asks for depth. Tone will be provided separately each turn.
              If a <style> ToneDirective block appears in the user message, you must follow it in the final prose; ignore it for any JSON/tool planning.

              You have access to a web search tool named "search_web" (backed by Tavily).

              STYLE:
              - Use PLAIN TEXT only — no Markdown formatting (no **bold**, italics, code, or backticks) in the answer body.
              - Write names and titles without special formatting.
              - Start conversational, then if applicable add exactly one line: "Fun fact: …" and optionally up to 3 short Sources lines.

              PLANNING STEP (first response only):
              - If the user asks for fresh facts, times, lineups, box office, what's-on-now, etc., do not answer directly.
              - Output exactly one JSON object on a single line with no extra text:
                {"action":"search_web","query":"..."}
              - If you already have enough info to answer confidently, output:
                {"action":"final","answer":"..."}

              ANSWERING STEP (after you see a message that starts with "search_web result:"):
              - Write the final answer in natural language as Movio (no JSON).
              - Follow STYLE above (plain text only, conversational tone).
              - Include exactly one short "Fun fact:" line after the main answer and before Sources.
              """
        )
    ]
    @State private var isSending = false
    @State private var errorText: String?
    @StateObject private var speech = SpeechManager()
    @State private var lastSpoken: String? = nil
    @StateObject private var stt = SpeechToTextManager()   // on-device STT
    // Manual mood selection for research/tests
    @State private var selectedMood: MoodTag = .calm
    @State private var selectedVoiceGender: VoiceGender = .neutral
    @FocusState private var promptFocused: Bool
    @FocusState private var urlFocused: Bool

    // Hugging Face VLM (Qwen2.5-VL) token and screenshot selection
    @State private var hfToken: String = "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"
    // Provider-qualified HF model for Qwen VL (e.g., ":nebius" suffix)
    @State private var hfModel: String = "Qwen/Qwen2.5-VL-72B-Instruct:nebius"
    private let modelScopeToken: String = "ms-ad35a3c3-03db-4ad2-a9ab-cdae539799b4"
    @State private var selectedItem: PhotosPickerItem? = nil
    @State private var selectedImageData: Data? = nil
    @State private var imageURLText: String = ""
    // Shows which engine handled the last turn
    @State private var activeEngine: String? = nil   // "GPT-OSS" or "QWEN Vision Agent"

    private let client = NvidiaChatClient()
    private let hfClient = HFVLMClient()

    // Only show user/assistant bubbles in the UI
    private var visibleMessages: [ChatMessage] {
        messages.filter { $0.role != "system" }
    }

    struct AgentAction: Codable {
        let action: String
        let query: String?
        let answer: String?
    }
    struct VisionAction: Codable {
        let action: String
        let query: String?
        let answer: String?
        let question: String?
    }
    private func decodeVisionAction(from text: String) -> VisionAction? {
        if let d = text.data(using: .utf8),
           let a = try? JSONDecoder().decode(VisionAction.self, from: d) { return a }
        if let r = text.range(of: #"\{[\s\S]*\}"#, options: .regularExpression) {
            if let d2 = String(text[r]).data(using: .utf8),
               let a2 = try? JSONDecoder().decode(VisionAction.self, from: d2) { return a2 }
        }
        return nil
    }
    struct TavilyResult: Codable { let title: String?; let url: String?; let content: String? }
    struct TavilyResponse: Codable { let answer: String?; let results: [TavilyResult]? }

    // Clean special tokens (e.g., <|return|>) and control chars that can confuse parsing
    private func cleanModelText(_ s: String) -> String {
        var t = s
        // Strip tokens like <|return|>, <|tool|>, etc.
        t = t.replacingOccurrences(of: #"<\|[^>]+?\|>"#, with: "", options: .regularExpression)
        // Remove ASCII control chars
        t = t.replacingOccurrences(of: #"[\u0000-\u001F]"#, with: "", options: .regularExpression)
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func decodeAction(from text: String) -> AgentAction? {
        // 1) Try strict JSON first
        if let d = text.data(using: .utf8),
           let a = try? JSONDecoder().decode(AgentAction.self, from: d) {
            return a
        }
        // 2) Fallback: extract first {...} using a regex and decode that
        if let range = text.range(of: #"\{[\s\S]*\}"#, options: .regularExpression) {
            let json = String(text[range])
            if json.replacingOccurrences(of: " ", with: "").replacingOccurrences(of: "\n", with: "") == "{}" { return nil }
            if let d2 = json.data(using: .utf8),
               let a2 = try? JSONDecoder().decode(AgentAction.self, from: d2) {
                return a2
            }
        }
        return nil
    }
    
    // MARK: - ModelScope (OpenAI-compatible) router for Qwen3-VL

    private func asOpenAIChatMessages(_ msgs: [HFMessage]) -> [[String: Any]] {
        msgs.map { m in
            let parts: [[String: Any]] = m.content.compactMap { p in
                switch p.type {
                case "text":      return ["type":"text", "text": p.text ?? ""]
                case "image_url": return ["type":"image_url", "image_url": ["url": p.image_url?.url ?? ""]]
                default:          return nil
                }
            }
            return ["role": m.role, "content": parts]
        }
    }

    private func modelScopeChat(messages: [HFMessage],
                                token: String,
                                model: String,
                                maxTokens: Int,
                                temperature: Double,
                                responseFormat: ResponseFormat?) async throws -> String {
        var req = URLRequest(url: URL(string: "https://api-inference.modelscope.cn/v1/chat/completions")!)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        let body: [String: Any] = [
            "model": model,
            "messages": asOpenAIChatMessages(messages),
            "temperature": temperature,
            "max_tokens": maxTokens
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, resp, _) = try await timedData(for: req, provider: "ModelScope", route: "/v1/chat/completions")
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw NSError(domain: "ModelScope",
                          code: (resp as? HTTPURLResponse)?.statusCode ?? -1,
                          userInfo: [NSLocalizedDescriptionKey: String(data: data, encoding: .utf8) ?? "HTTP error"])
        }
        struct Choice: Decodable { struct Msg: Decodable { let content: String? }; let message: Msg? }
        struct R: Decodable { let choices: [Choice] }
        let r = try JSONDecoder().decode(R.self, from: data)
        return r.choices.first?.message?.content ?? ""
    }

    private func vlmChat(messages: [HFMessage],
                         model: String,
                         maxTokens: Int,
                         temperature: Double,
                         responseFormat: ResponseFormat?) async throws -> String {
        // Convention: suffix ":modelscope" selects ModelScope provider (Qwen3-VL)
        let parts = model.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: true)
        let bare  = String(parts.first ?? Substring(model))
        let tag   = parts.count > 1 ? String(parts.last!) : ""
        if tag.lowercased() == "modelscope" {
            return try await modelScopeChat(messages: messages,
                                            token: modelScopeToken,   // <-- use the private constant
                                            model: bare,
                                            maxTokens: maxTokens,
                                            temperature: temperature,
                                            responseFormat: responseFormat)
        } else {
            // Default to your Hugging Face path (Qwen2.5-VL etc.)
            return try await hfClient.chat(messages: messages,
                                           hfToken: hfToken,
                                           model: model,
                                           maxTokens: maxTokens,
                                           temperature: temperature,
                                           responseFormat: responseFormat)
        }
    }
    
    // MARK: - Direct answer helpers (GPT‑OSS text path)
    @MainActor

    // Build a minimal, planning‑free context to coax a direct natural‑language answer
    private func buildDirectAnswerMessages(latestUser: String) -> [ChatMessage] {
        var msgs: [ChatMessage] = [
            .init(role: "system", content:
                """
                You are Movio, an assistant for movie & theatre lovers.
                Use PLAIN TEXT only (no Markdown). Keep answers concise and accurate.
                Resolve pronouns like “he/she/they” to the most recent named person or title mentioned in CONTEXT.
                If still unclear, ask a one-line clarification instead of returning nothing.
                Format (when applicable):
                1) Main answer (1–2 sentences).
                2) Fun fact: <one short line>.
                3) Sources (0–3 simple dash bullets like "- Title — URL").
                """
            )
        ]
        let ctx = recentTextContext(maxChars: 1200, maxMessages: 8)
        if !ctx.isEmpty {
            msgs.append(.init(role: "system", content: "CONTEXT (prior turns):\n\(ctx)"))
        }
        msgs.append(.init(role: "user", content: decorateUserText(latestUser, selectedMood)))
        return msgs
    }

    // Infer the most recent proper-name entity (e.g., "Hugh Jackman") from recent messages
    private func inferFocusEntity() -> String? {
        for m in messages.reversed() where m.role != "system" {
            if let r = m.content.range(of: #"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"#, options: .regularExpression) {
                return String(m.content[r])
            }
        }
        return nil
    }

    private func tavilySearch(query: String) async throws -> String {
        guard let url = URL(string: "https://api.tavily.com/search") else { return "" }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            "api_key": tavilyKey,
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
            "include_answer": true
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, resp, _) = try await timedData(for: req, provider: "Tavily", route: "/search")
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            return String(data: data, encoding: .utf8) ?? ""
        }
        let decoded = try? JSONDecoder().decode(TavilyResponse.self, from: data)
        let items = (decoded?.results ?? []).prefix(3)
        let lines = items.compactMap { r in
            let t = r.title ?? "Untitled"
            let u = r.url ?? ""
            let c = (r.content ?? "").prefix(280)
            return "- \(t)\n  \(u)\n  \(c)"
        }
        if let ans = decoded?.answer, !ans.isEmpty {
            return "ANSWER:\n\(ans)\n\nSOURCES:\n" + lines.joined(separator: "\n")
        }
        return lines.joined(separator: "\n")
    }

    var body: some View {
        VStack(spacing: 16) {
            // Greeting banner
            Text("Hello, I’m Movio — have any questions I can answer?")
                .font(.footnote)
                .foregroundStyle(.secondary)
            if let engine = activeEngine {
                HStack {
                    Spacer()
                    Label(engine, systemImage: engine == "QWEN Vision Agent" ? "camera.viewfinder" : "bolt.horizontal.circle")
                        .font(.caption2)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(
                            Capsule().fill((engine == "QWEN Vision Agent") ? Color.blue.opacity(0.12) : Color.green.opacity(0.12))
                        )
                        .overlay(
                            Capsule().stroke((engine == "QWEN Vision Agent") ? Color.blue.opacity(0.6) : Color.green.opacity(0.6), lineWidth: 0.8)
                        )
                        .foregroundStyle((engine == "QWEN Vision Agent") ? .blue : .green)
                }
            }
            
            // Mood selector (lightweight UI for research)
            // Mood + voice selector (lightweight UI for research)
            HStack(spacing: 12) {
                Text("Mood:")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Menu {
                    ForEach(MoodTag.allCases) { m in
                        Button(m.rawValue.capitalized) { selectedMood = m }
                    }
                } label: {
                    Label(selectedMood.rawValue.capitalized, systemImage: "face.smiling")
                        .font(.caption)
                }

                Text("Voice:")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Menu {
                    ForEach(VoiceGender.allCases) { g in
                        Button(g.rawValue.capitalized) {
                            selectedVoiceGender = g
                            MovioUnityBridge.shared.setVoiceGender(g)
                        }
                    }
                } label: {
                    Label(selectedVoiceGender.rawValue.capitalized, systemImage: "speaker.wave.2")
                        .font(.caption)
                }

                Spacer()
            }
            

            

            // Conversation view (auto-scroll; hide system messages)
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(visibleMessages) { msg in
                            HStack(alignment: .top) {
                                Text(msg.role.capitalized + ":")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .frame(width: 80, alignment: .trailing)
                                Text(msg.content)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .id(msg.id)
                            .padding(8)
                            .background(msg.role == "assistant" ? Color.gray.opacity(0.15) : .clear)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                        }
                    }
                    .padding(.vertical, 4)
                }
                .onAppear {
                    if let last = visibleMessages.last {
                        DispatchQueue.main.async {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
                .onChange(of: messages.count) { _, _ in
                    if let last = visibleMessages.last {
                        withAnimation {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }

            // Input + buttons (now with mic for push-to-talk)
            HStack {
                // Mic: tap to start/stop listening
                Button {
                    Task { await toggleMic() }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: stt.isListening ? "mic.fill" : "mic")
                        Text(stt.isListening ? "Stop" : "Speak")
                    }
                }
                .buttonStyle(.bordered)
                .disabled(isSending)
                .accessibilityLabel(stt.isListening ? "Stop listening" : "Start listening")

                // Screenshot picker (PhotosPicker)
                PhotosPicker(selection: $selectedItem, matching: .images) {
                    Label(selectedImageData == nil ? "Screenshot" : "Screenshot ✓", systemImage: "photo")
                }
                .onChange(of: selectedItem) { _, newItem in
                    Task {
                        if let data = try? await newItem?.loadTransferable(type: Data.self) {
                            selectedImageData = data
                        } else {
                            selectedImageData = nil
                        }
                    }
                }

                // Optional: Image URL input (alternate to screenshot)
                TextField("Image URL (optional)", text: $imageURLText)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled(true)
                    .keyboardType(.URL)
                    .frame(minWidth: 180)
                    .padding()
                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(.secondary))
                    .focused($urlFocused)

                // Live text (we also show partial transcript while listening)
                TextField("Type your prompt…", text: $userInput, axis: .vertical)
                    .lineLimit(1...4)
                    .padding()
                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(.secondary))
                    .focused($promptFocused)
                // Send typed prompt (auto-routes to vision or text as needed)
                Button {
                    Task { await send() }
                } label: {
                    if isSending { ProgressView() } else { Text("Send") }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canSend)
            }

         

            // Listening badge (shows only while mic is active)
            if stt.isListening {
                Label("Listening… (auto-stops on pause)", systemImage: "waveform")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            HStack {
                Button("Clear Conversation") {
                    messages = [ .init(role: "system", content:
                        """
                        You are Movio, an assistant for movie & theatre lovers—and for people currently watching a film or show. Speak like a friendly movie-buff friend: casual, warm, and concise. Use natural contractions and short openers when it fits (Yeah—, Oh totally—). Keep answers brief (2–4 sentences) unless the user asks for depth. Tone will be provided separately each turn.
                        If a <style> ToneDirective block appears in the user message, you must follow it in the final prose; ignore it for any JSON/tool planning.

                        You have access to a web search tool named "search_web" (backed by Tavily).

                        STYLE:
                        - Use PLAIN TEXT only — no Markdown formatting (no **bold**, italics, code, or backticks) in the answer body.
                        - Write names and titles without special formatting.
                        - Start conversational, then if applicable add exactly one line: "Fun fact: …" and optionally up to 3 short Sources lines.

                        PLANNING STEP (first response only):
                        - If the user asks for fresh facts, times, lineups, box office, what's-on-now, etc., do not answer directly.
                        - Output exactly one JSON object on a single line with no extra text:
                          {"action":"search_web","query":"..."}
                        - If you already have enough info to answer confidently, output:
                          {"action":"final","answer":"..."}

                        ANSWERING STEP (after you see a message that starts with "search_web result:"):
                        - Write the final answer in natural language as Movio (no JSON).
                        - Follow STYLE above (plain text only, conversational tone).
                        - Include exactly one short "Fun fact:" line after the main answer and before Sources.
                        """
                    ) ]
                    lastSpoken = nil
                    imageURLText = ""
                }
                .buttonStyle(.bordered)

                Button("Repeat last answer") {
                    if let last = lastSpoken {
                        speech.synth.stopSpeaking(at: .immediate)
                        speakAnswer(last)
                    }
                }
                .buttonStyle(.bordered)
                .disabled(lastSpoken == nil)

                Button("Stop") {
                    speech.synth.stopSpeaking(at: .immediate)
                }
                .buttonStyle(.bordered)
                .disabled(!speech.synth.isSpeaking)

                Spacer()

                if let e = errorText {
                    Text(e).foregroundStyle(.red).font(.footnote)
                }
            }
        }
        .padding(24)
        .onAppear {
                // Ensure playback session is configured for TTS at launch
                speech.configureAudioSession()
                MovioUnityBridge.shared.setVoiceGender(selectedVoiceGender)
                print("NOTE: Requested GEMMA3-14B not found; using GEMMA3-12B via featherless-ai; Qwen-32B via fireworks-ai.")
            }
            .onChange(of: selectedVoiceGender) { _, newValue in
                MovioUnityBridge.shared.setVoiceGender(newValue)
            }
    }

    // Fetch a remote image and return (rawData, mime). We accept only https and image/* responses.
    private func fetchImageFromURL(_ urlString: String,
                                   maxDownloadBytes: Int = 8_000_000,
                                   timeout: TimeInterval = 15) async throws -> (Data, String) {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed), ["https"].contains(url.scheme?.lowercased() ?? "") else {
            throw NSError(domain: "Movio", code: 400, userInfo: [NSLocalizedDescriptionKey: "Only https image URLs are supported."])
        }
        var req = URLRequest(url: url, timeoutInterval: timeout)
        req.setValue("Mozilla/5.0 (Movio-Swift)", forHTTPHeaderField: "User-Agent")
        req.setValue("image/avif,image/webp,image/apng,image/*,*/*;q=0.8", forHTTPHeaderField: "Accept")

        let (data, resp, metric) = try await timedData(for: req, provider: "RemoteImage", route: "GET image")
        print("URL fetch ms:", metric.totalMs, "bytes:", data.count)

        let len = resp.expectedContentLength
        if len > 0 && len > (maxDownloadBytes * 2) {
            throw NSError(domain: "Movio", code: 413, userInfo: [NSLocalizedDescriptionKey: "Image too large (Content-Length)." ])
        }
        if data.count == 0 {
            throw NSError(domain: "Movio", code: 404, userInfo: [NSLocalizedDescriptionKey: "Empty response from URL."])
        }

        var mime = (resp.mimeType ?? "").lowercased()
        if !mime.hasPrefix("image/") {
            // Sniff via CGImageSource
            if let src = CGImageSourceCreateWithData(data as CFData, nil),
               let uti = CGImageSourceGetType(src) {
                if let ut = UTType(uti as String) {
                    if ut.conforms(to: .jpeg) { mime = "image/jpeg" }
                    else if ut.conforms(to: .png) { mime = "image/png" }
                    else if ut.conforms(to: .gif) { mime = "image/gif" }
                    else { mime = "image/jpeg" }
                } else {
                    // Fallback when the identifier cannot be mapped to UTType
                    mime = "image/jpeg"
                }
            } else {
                throw NSError(domain: "Movio", code: 415, userInfo: [NSLocalizedDescriptionKey: "URL did not return an image."])
            }
        }
        return (data, mime)
    }

    // Convenience: build a data: URL string from (data, mime)
    private func makeDataURL(data: Data, mime: String) -> String {
        return "data:\(mime);base64," + data.base64EncodedString()
    }

    // DRY: append assistant message and speak it
    private func showAndSpeak(_ text: String) {
        messages.append(.init(role: "assistant", content: text))
        speakAnswer(text)
    }

    private func resolveContentViewVoice() -> AVSpeechSynthesisVoice? {
        let preferredLanguage = MovioUnityBridge.shared.voiceLanguage.trimmingCharacters(in: .whitespacesAndNewlines)
        let allVoices = AVSpeechSynthesisVoice.speechVoices()

        if let identifier = MovioUnityBridge.shared.voiceIdentifier,
        let exactVoice = AVSpeechSynthesisVoice(identifier: identifier) {
            return exactVoice
        }

        let languageMatched = allVoices.filter {
            $0.language.caseInsensitiveCompare(preferredLanguage) == .orderedSame
        }
        let regionalMatched = languageMatched.isEmpty
            ? allVoices.filter { $0.language.lowercased().hasPrefix(preferredLanguage.lowercased().prefix(2)) }
            : languageMatched
        let primaryPool = regionalMatched.isEmpty ? allVoices : regionalMatched

        func genderScore(for voice: AVSpeechSynthesisVoice) -> Int {
            let name = voice.name.lowercased()
            switch selectedVoiceGender {
            case .male:
                if name.contains("male") { return 3 }
                if name.contains("man") { return 2 }
                return 0
            case .female:
                if name.contains("female") { return 3 }
                if name.contains("woman") { return 2 }
                return 0
            case .neutral:
                return 0
            }
        }

        func qualityScore(for voice: AVSpeechSynthesisVoice) -> Int {
            switch voice.quality {
            case .enhanced: return 1
            default: return 0
            }
        }

        if selectedVoiceGender != .neutral,
        let gendered = primaryPool
            .sorted(by: {
                let left = genderScore(for: $0)
                let right = genderScore(for: $1)
                if left != right { return left > right }
                let leftQuality = qualityScore(for: $0)
                let rightQuality = qualityScore(for: $1)
                if leftQuality != rightQuality { return leftQuality > rightQuality }
                return $0.name < $1.name
            })
            .first(where: { genderScore(for: $0) > 0 }) {
            return gendered
        }

        if let preferred = primaryPool.sorted(by: {
            let leftQuality = qualityScore(for: $0)
            let rightQuality = qualityScore(for: $1)
            if leftQuality != rightQuality { return leftQuality > rightQuality }
            return $0.name < $1.name
        }).first {
            return preferred
        }

        return AVSpeechSynthesisVoice(language: preferredLanguage)
            ?? AVSpeechSynthesisVoice(language: AVSpeechSynthesisVoice.currentLanguageCode())
            ?? AVSpeechSynthesisVoice(language: "en-GB")
            ?? AVSpeechSynthesisVoice(language: "en-US")
    }

    // DRY: compress/resize an attached screenshot to an optimized payload
    private func optimizedPayload(for data: Data) -> (Data, String) {
        let p = prepareImageForVLM(data, maxPixel: 1280, targetMaxBytes: 3_000_000, quality: 0.72)
        return (p?.0 ?? data, p?.1 ?? "image/jpeg")
    }

    // DRY: ask the model for a short continuation if the text looks incomplete
    private func continueIfIncomplete(_ partial: String,
                                      context: [ChatMessage],
                                      apiKey: String) async throws -> String {
        var text = partial
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let needsMore = trimmed.isEmpty || trimmed.hasSuffix("…") || !".!?".contains(trimmed.last ?? " ")
        if needsMore {
            let ctx = context + [
                .init(role: "assistant", content: text),
                .init(role: "user", content: "Continue from where you left off. Finish without repeating.")
            ]
            let cont = try await client.send(messages: ctx, apiKey: apiKey)
            let extra = cleanModelText(cont.choices.first?.message?.content ?? "")
            if !extra.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { text += "\n" + extra }
        }
        return text
    }
    
    // Apply Movio buddy tone across all moods and routes (LLM + VLM)
    private func applyMovioTone(_ text: String, mood: MoodTag) async -> String {
        var out = text
        // Pass 1: friendly rewrite if the output looks stiff (generic heuristic)
        if shouldFriendlyRewrite(out) {
            if let rewrite = try? await client.send(messages: [
                .init(role: "system", content: "Rewrite as a friendly movie-buff friend. Keep facts. 2–4 short sentences. Plain text only. No Sources unless asked."),
                .init(role: "user", content: styleBlock(for: mood) + "\n\n" + out)
            ], apiKey: apiKey), let msg = rewrite.choices.first?.message {
                let r = msg.content.trimmingCharacters(in: .whitespacesAndNewlines)
                if !r.isEmpty { out = r }
            }
        }
        // Pass 2: always enforce the selected mood tone (even Calm)
        if let rewriteTone = try? await client.send(messages: [
            .init(role: "system", content: "Rewrite to match this tone guide exactly. Keep meaning and facts. Plain text only. 2–4 short sentences. No Sources unless asked."),
            .init(role: "user", content: styleBlock(for: mood) + "\n\n" + out)
        ], apiKey: apiKey), let msg2 = rewriteTone.choices.first?.message {
            let toned = msg2.content.trimmingCharacters(in: .whitespacesAndNewlines)
            if !toned.isEmpty { out = toned }
        }
        return out
    }

    // DRY: final-answer style for composing from web results
    private let SHOW_SOURCES_BY_DEFAULT = false
    private let movioFinalStyle =
    """
    Now write the final answer in natural language as Movio — friendly and concise. Use PLAIN TEXT only (no Markdown). Start conversational with natural contractions; keep it to 2–4 sentences. If helpful, add exactly one line starting with "Fun fact:". Only include Sources if the user asked for links; otherwise omit them. Do not output JSON.
    """
    
    private func isBlank(_ s: String) -> Bool {
        s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func looksLikeJSON(_ s: String) -> Bool {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.hasPrefix("{") || t.hasPrefix("[")
    }

    // Try to decode {"action":"final","answer": "..."} from any JSON-ish string.
    private func extractAnswerIfJSON(_ s: String) -> String? {
        if let act = decodeAction(from: s),
           act.action == "final",
           let a = act.answer,
           !a.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return a
        }
        return nil
    }

    private func answerFromSearchResult(_ result: String) -> String? {
        // If Tavily provided an ANSWER section, prefer that.
        if let mark = result.range(of: "ANSWER:", options: .caseInsensitive) {
            let tail = result[mark.upperBound...]
            let end = tail.range(of: "\nSOURCES:", options: .caseInsensitive)?.lowerBound ?? tail.endIndex
            let extracted = String(tail[..<end]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !extracted.isEmpty { return extracted }
        }
        // Fallback: pick the first non-empty, non-bullet, non-URL line.
        let lines = result.split(separator: "\n").map(String.init)
        if let first = lines.first(where: { line in
            let l = line.trimmingCharacters(in: .whitespaces)
            guard !l.isEmpty else { return false }
            if l.hasPrefix("-") || l.hasPrefix("•") { return false }
            if l.lowercased().hasPrefix("source") { return false }
            if l.lowercased().hasPrefix("sources") { return false }
            if l.hasPrefix("http") { return false }
            return true
        }) {
            return first
        }
        return nil
    }
    
    // Extract up to 3 "- Title — URL" bullets from the Tavily result text.
    private func extractSourcesBullets(from result: String) -> [String] {
        var bullets: [String] = []
        do {
            let regex = try NSRegularExpression(pattern: #"(?m)^- (.+)\n\s+(https?://\S+)"#)
            let ns = result as NSString
            let matches = regex.matches(in: result, range: NSRange(location: 0, length: ns.length))
            for m in matches.prefix(3) {
                if m.numberOfRanges >= 3 {
                    let title = ns.substring(with: m.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                    let url = ns.substring(with: m.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines)
                    bullets.append("- \(title) — \(url)")
                }
            }
        } catch {
            print("Sources regex error:", error.localizedDescription)
        }
        return bullets
    }

    private func extractSpeakable(from full: String) -> String {
        var text = full
        // Normalize line endings
        text = text.replacingOccurrences(of: "\r\n", with: "\n")
                   .replacingOccurrences(of: "\r", with: "\n")

        // If a Fun fact line exists, keep through that line and drop the rest
        if let fun = text.range(of: "(?im)^\\n?\\s*fun\\s*fact\\s*:\\s*.*$", options: .regularExpression) {
            let afterFun = text[fun.upperBound...]
            if let nextBreak = afterFun.firstIndex(of: "\n") {
                return String(text[..<nextBreak]).trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                return text.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }

        // Otherwise cut before common source/reference sections or first naked URL line
        let patterns = [
            #"(?im)\n\s*\*\*sources\*\*"#,
            #"(?im)\n\s*sources\s*:"#,
            #"(?im)\n\s*sources\b"#,
            #"(?im)\n\s*references\b"#,
            #"(?im)\n\s*citations\b"#,
            #"(?im)\n\s*further\s*reading\b"#,
            #"(?im)\n\s*links\b"#
        ]
        var cut: String.Index? = nil
        for p in patterns {
            if let r = text.range(of: p, options: .regularExpression) {
                cut = (cut == nil || r.lowerBound < cut!) ? r.lowerBound : cut
            }
        }
        if let urlRange = text.range(of: #"(?m)^[ \t]*https?://\S+.*$"#, options: .regularExpression) {
            cut = (cut == nil || urlRange.lowerBound < cut!) ? urlRange.lowerBound : cut
        }
        let head = cut.map { String(text[..<$0]) } ?? text
        return head.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func sanitizeForSpeech(_ text: String) -> String {
        var s = text
        // Strip common markdown emphasis/backticks
        s = s.replacingOccurrences(of: #"(\*\*|__)(.*?)\1"#, with: "$2", options: .regularExpression)
        s = s.replacingOccurrences(of: #"`([^`]+)`"#, with: "$1", options: .regularExpression)
        // Replace URLs with a neutral token
        s = s.replacingOccurrences(of: #"https?://\S+"#, with: "(link)", options: .regularExpression)
        // Remove bullet markers at line starts
        s = s.replacingOccurrences(of: #"(?m)^\s*[-•]\s*"#, with: "", options: .regularExpression)
        // Collapse excessive whitespace
        s = s.replacingOccurrences(of: #"\s{2,}"#, with: " ", options: .regularExpression)
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func splitIntoSentences(_ text: String) -> [String] {
        var out: [String] = []
        var current = ""
        for ch in text {
            current.append(ch)
            if ".!?".contains(ch) {
                let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { out.append(trimmed) }
                current = ""
            }
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { out.append(tail) }
        return out
    }
    
    private func speakAnswer(_ text: String) {
        // 1) Trim sources, sanitize, remember for Repeat
        let speakable = sanitizeForSpeech(extractSpeakable(from: text))
        guard !speakable.isEmpty else { return }
        lastSpoken = speakable

        // 2) Configure voice: prefer British English; fall back to device's current language
        let voice = resolveContentViewVoice()
        let (rate, pitch) = ttsParams(for: selectedMood)

        // 3) Interrupt anything currently speaking
        speech.synth.stopSpeaking(at: .immediate)

        // 4) Speak sentence by sentence with a tiny pause
        let sentences = splitIntoSentences(speakable)
        for (idx, sentence) in sentences.enumerated() {
            let u = AVSpeechUtterance(string: sentence)
            u.voice = voice
            u.rate = rate
            u.pitchMultiplier = max(0.5, min(pitch, 2.0))
            u.postUtteranceDelay = (idx == sentences.count - 1) ? 0.0 : 0.12
            speech.synth.speak(u)
        }
    }

    /// Toggle push-to-talk mic:
    /// - Start: stop TTS, request permissions, switch audio session to playAndRecord, start STT.
    /// - Stop: stop STT and restore playback session. On final text, auto-send.
    @MainActor
    private func toggleMic() async {
        // If we are currently listening, stop and finalize.
        if stt.isListening {
            stt.stopListening(finalize: true)
            // Restore playback session for TTS
            speech.configureAudioSession()
            return
        }

        // Resign focus from text fields to avoid RTI focus churn
        promptFocused = false
        urlFocused = false

        // Interrupt any ongoing speech to avoid echo
        speech.synth.stopSpeaking(at: .immediate)

        // Ask for authorization
        stt.ensureAuthorization { granted in
            guard granted else {
                self.errorText = "Microphone/Speech permission is required."
                return
            }
            // Switch audio session for recording
            let session = AVAudioSession.sharedInstance()
            do {
                try session.setCategory(.playAndRecord, options: [.duckOthers, .allowBluetoothA2DP])
                try session.setMode(.measurement) // better for speech recognition capture
                try session.setActive(true)
#if targetEnvironment(simulator)
                print("STT: Running in Simulator — speech recognition/microphone may be limited.")
#endif
            } catch {
                self.errorText = "Audio session error: \(error.localizedDescription)"
                return
            }

            // On final transcript: populate field, stop STT, restore playback, auto-send.
            self.stt.onFinal = { finalText in
                DispatchQueue.main.async {
                    self.userInput = finalText
                    // Return to playback for TTS
                    self.speech.configureAudioSession()
                    // Auto-send if we have something
                    if !self.userInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        Task { await self.send() }
                    }
                }
            }
            self.stt.onPartial = { live in
                DispatchQueue.main.async {
                    self.userInput = live
                }
            }
            self.stt.onError = { message in
                DispatchQueue.main.async {
                    self.errorText = message
                    if self.stt.isListening {
                        self.stt.stopListening(finalize: false)
                    }
                    // Return to playback session for TTS
                    self.speech.configureAudioSession()
                }
            }

            do {
                try self.stt.startListening(autoStopAfterSilence: 1.8)
            } catch {
                self.errorText = "Could not start microphone: \(error.localizedDescription)"
                // Ensure we return to playback mode
                self.speech.configureAudioSession()
            }
        }
    }

    /// Downscale and compress image data for VLM upload.
    /// Returns (optimizedData, mime). We always output JPEG to keep payloads small.
    private func prepareImageForVLM(_ data: Data,
                                    maxPixel: CGFloat = 1280,
                                    targetMaxBytes: Int = 3_000_000,
                                    quality: CGFloat = 0.7) -> (Data, String)? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        // Iteratively reduce the longest side until under size budget or floor.
        var pixel: CGFloat = maxPixel
        var lastGood: Data? = nil

        while pixel >= 640 {
            let opts: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: Int(pixel),
                kCGImageSourceCreateThumbnailWithTransform: true
            ]
            guard let thumb = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) else { break }

            let out = NSMutableData()
            guard let dest = CGImageDestinationCreateWithData(out, UTType.jpeg.identifier as CFString, 1, nil) else { break }
            let destOpts: [CFString: Any] = [ kCGImageDestinationLossyCompressionQuality: quality ]
            CGImageDestinationAddImage(dest, thumb, destOpts as CFDictionary)
            guard CGImageDestinationFinalize(dest) else { break }

            let result = out as Data
            if result.count <= targetMaxBytes {
                return (result, "image/jpeg")
            } else {
                lastGood = result
                pixel *= 0.75  // reduce and try again
            }
        }
        if let last = lastGood { return (last, "image/jpeg") }
        return nil
    }

    /// Build a short plain-text transcript of recent turns (user/assistant),
    /// so Qwen can share context with GPT-OSS across turns. Images are summarized by text anyway.
    private func recentTextContext(maxChars: Int = 2000, maxMessages: Int = 12) -> String {
        let convo = messages.filter { $0.role != "system" }.suffix(maxMessages)
        var out = ""
        for m in convo {
            // Strip Sources/URLs so the salient names/titles survive for coreference
            let cleaned = extractSpeakable(from: m.content)
            let line = "\(m.role.capitalized): \(cleaned)\n"
            if out.count + line.count > maxChars { break }
            out += line
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Ask Qwen to plan next step from image+text. Returns parsed action.
    private func qwenPlan(imageData: Data, mime: String, userText: String) async throws -> VisionAction? {
        let planSystem = """
        You are Movio-Vision. Use the IMAGE and the user's text.
        Return exactly ONE JSON object on a single line with no extra text.
        Allowed actions:
        - {"action":"search_web","query":"..."}
        - {"action":"final","answer":"..."}
        - {"action":"ask_user","question":"..."}
        Rules: plain JSON only; keep queries short and specific.
        """
        // Include recent text-only context so Qwen can maintain continuity
        let ctxText = recentTextContext()
        var msgs: [HFMessage] = [HFMessage(role: "system", content: [HFContentPart(type: "text", text: planSystem, image_url: nil)])]
        if !ctxText.isEmpty {
            msgs.append(HFMessage(role: "system", content: [HFContentPart(type: "text", text: "CONTEXT (prior turns):\n\(ctxText)", image_url: nil)]))
        }
        let dataURL = "data:\(mime);base64," + imageData.base64EncodedString()
        let usr = HFMessage(role: "user", content: [
            HFContentPart(type: "text", text: userText, image_url: nil),
            HFContentPart(type: "image_url", text: nil, image_url: HFImageURL(url: dataURL))
        ])
        let text = try await vlmChat(messages: msgs + [usr],
                                     model: hfModel,
                                     maxTokens: 192,
                                     temperature: 0.2,
                                     responseFormat: ResponseFormat(type: "json_object"))
        return decodeVisionAction(from: text)
    }

    /// Ask Qwen to write the final Movio answer (plain text), optionally using a web result.
    private func qwenFinalize(imageData: Data, mime: String, userText: String, searchBlock: String?) async throws -> String {
        let finalSystem = """
        Write the final answer as Movio. Start conversational and human — natural contractions, short opener if it fits (Yeah—, Oh totally—). Keep it brief (2–4 sentences) unless the user asked for depth. Use plain text only (no markdown). Then add exactly one line: "Fun fact: …". If a web result is provided, base facts on it. Optionally include up to 3 short Sources lines like "- Title — URL".
        """
        var msgs: [HFMessage] = [
            HFMessage(role: "system", content: [HFContentPart(type: "text", text: finalSystem, image_url: nil)])
        ]
        let ctxText = recentTextContext()
        if !ctxText.isEmpty {
            msgs.append(HFMessage(role: "system", content: [HFContentPart(type: "text", text: "CONTEXT (prior turns):\n\(ctxText)", image_url: nil)]))
        }
        if let block = searchBlock {
            msgs.append(HFMessage(role: "system", content: [HFContentPart(type: "text", text: "search_web result:\n\(block)", image_url: nil)]))
        }
        let dataURL = "data:\(mime);base64," + imageData.base64EncodedString()
        // Order: style block first, then user text, then image
        msgs.append(HFMessage(role: "user", content: [
            HFContentPart(type: "text", text: styleBlock(for: selectedMood), image_url: nil),
            HFContentPart(type: "text", text: userText, image_url: nil),
            HFContentPart(type: "image_url", text: nil, image_url: HFImageURL(url: dataURL))
        ]))
        // DIAGNOSTIC
        if let user = msgs.last {
            let joined = user.content.compactMap { $0.text }.joined(separator: " | ")
            print("QWEN DECORATED USER ORDER:", joined.prefix(240))
        }
        let text = try await vlmChat(messages: msgs,
                                     model: hfModel,
                                     maxTokens: 192,
                                     temperature: 0.2,
                                     responseFormat: ResponseFormat(type: "json_object"))
        return text
    }

    /// Full Qwen agentic loop: plan -> (optional search) -> finalize.
    @MainActor
    private func runQwenAgent(imageData: Data, mime: String, userText: String) async {
        do {
            guard let action = try await qwenPlan(imageData: imageData, mime: mime, userText: userText) else {
                errorText = "Could not parse Qwen plan."
                return
            }
            switch action.action {
            case "search_web":
                let q = action.query?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                let result = try await tavilySearch(query: q)
                let final = try await qwenFinalize(imageData: imageData, mime: mime, userText: userText, searchBlock: result)
                let toned = await applyMovioTone(final, mood: selectedMood)
                showAndSpeak(toned)
            case "final":
                // Even if the plan returned an answer, re-compose with tone-aware finalize so the style block is applied.
                let final = try await qwenFinalize(imageData: imageData, mime: mime, userText: userText, searchBlock: nil)
                let toned = await applyMovioTone(final, mood: selectedMood)
                showAndSpeak(toned)
            case "ask_user":
                let q = action.question ?? "Could you clarify what you want to know about this image?"
                showAndSpeak(q)
            default:
                let fallback = "Sorry — I couldn't decide next steps."
                showAndSpeak(fallback)
            }
        } catch {
            errorText = error.localizedDescription
        }
    }



    // MARK: - Actions
    @MainActor
    private func send() async {
        errorText = nil
        let trimmed = userInput.trimmingCharacters(in: .whitespacesAndNewlines)

        // Print tone diagnostics for research
        print("Mood:", selectedMood.rawValue)

        // If a screenshot is attached, use the QWEN Vision Agent path
        if let imageData = selectedImageData {
            let promptToUse = trimmed.isEmpty ? "Describe this image in one or two sentences, focusing on film/theatre-relevant details." : trimmed

            // Show the user's message in the chat bubble for consistency
            messages.append(.init(role: "user", content: promptToUse))
            userInput = ""

            // Prepare image (resize/compress)
            let (payloadData, payloadMime) = optimizedPayload(for: imageData)
            print("Vision upload bytes:", payloadData.count)

            // Badge + sending state
            activeEngine = "QWEN Vision Agent"
            isSending = true
            defer { isSending = false }

          
            await runQwenAgent(imageData: payloadData, mime: payloadMime, userText: promptToUse)
            
            // Auto-clear screenshot selection after a vision turn
            await MainActor.run {
                selectedImageData = nil
                selectedItem = nil
            }
            return
        }

        // If an Image URL is provided (and no screenshot), use the same QWEN Vision Agent path
        if selectedImageData == nil, !imageURLText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let promptToUse = trimmed.isEmpty ? "Describe this image in one or two sentences, focusing on film/theatre-relevant details." : trimmed
            // Show the user's message for consistency
            messages.append(.init(role: "user", content: promptToUse))
            userInput = ""

            // Fetch & normalize the image
            isSending = true
            activeEngine = "QWEN Vision Agent"
            defer { isSending = false }
            do {
                let (raw, _) = try await fetchImageFromURL(imageURLText)
                let (payloadData, payloadMime) = optimizedPayload(for: raw)
                print("Vision payload bytes (from URL):", payloadData.count)
                activeEngine = "QWEN Vision Agent"
                
                await runQwenAgent(imageData: payloadData, mime: payloadMime, userText: promptToUse)
                
                await MainActor.run {
                    // Clear URL after a vision turn
                    imageURLText = ""
                }
            } catch {
                let msg = error.localizedDescription
                self.errorText = msg
                print("Image URL error:", msg)
                // Keep the message list consistent with an assistant error bubble
                messages.append(.init(role: "assistant", content: "Sorry — couldn’t load that image URL: \(msg)"))
            }
            return
        }

        // Otherwise, use the GPT‑OSS text agent path
        guard !trimmed.isEmpty else { return }
        activeEngine = "GPT-OSS"
        userInput = ""
        messages.append(.init(role: "user", content: trimmed))

        if messages.count > 20 {
            if let i = messages.firstIndex(where: { $0.role != "system" }) {
                messages.remove(at: i) // drop oldest non-system turn; keep system prompt
            }
        } // keep system, trim oldest

        isSending = true
        defer { isSending = false }

        do {
            // --- PLAN: ask for a JSON action; two attempts; never surface raw JSON to user ---
            let first = try await client.send(messages: messages, apiKey: apiKey,
                                              responseFormat: ResponseFormat(type: "json_object"))
            var planText = first.choices.first?.message?.content ?? ""
            print("FIRST RAW MODEL OUTPUT:", planText)
            planText = cleanModelText(planText)

            var action = decodeAction(from: planText)
            if action == nil {
                print("PLAN PARSE FAIL: forcing strict JSON on retry")
                let forced = messages + [
                    .init(role: "system", content: "PLANNING STEP: Return ONLY a single JSON object exactly as specified. No explanations, no extra text.")
                ]
                let retry = try await client.send(messages: forced, apiKey: apiKey,
                                                  responseFormat: ResponseFormat(type: "json_object"))
                let retryText = retry.choices.first?.message?.content ?? ""
                print("RETRY PLAN OUTPUT:", retryText)
                planText = cleanModelText(retryText)
                action = decodeAction(from: planText)
            }

            if let action = action {
                if action.action == "search_web", let q = action.query, !q.isEmpty {
                // Execute tool locally
                let result = try await tavilySearch(query: q)
                print("TAVILY RESULT (truncated):", String(result.prefix(400)))
                // Append tool output
                var compose = messages
                compose.append(.init(role: "system", content: "search_web result:\n\(result)"))
                compose.append(.init(role: "system", content: movioFinalStyle))
                // Add the styled user prompt only for the final prose generation
                compose.append(.init(role: "user", content: decorateUserText(trimmed, selectedMood)))
                // DIAGNOSTIC: print the decorated tail we send
                if let lastUser = compose.last(where: { $0.role == "user" }) {
                    print("FINAL DECORATED USER (GPT‑OSS):", lastUser.content.prefix(240))
                }
                let second = try await client.send(messages: compose, apiKey: apiKey)
                    var finalText = second.choices.first?.message?.content ?? ""
                    // If the model still sent JSON, normalize it:
                    if let maybe = decodeAction(from: finalText) {
                        if maybe.action == "final", let a = maybe.answer, !a.isEmpty {
                            finalText = a
                        } else if maybe.action == "search_web" {
                            // The model planned again instead of answering; fall back to tool's answer.
                            if let ans = answerFromSearchResult(result) { finalText = ans }
                        }
                    }
                    // If cut off, request a short continuation once
                    // Ask for a short continuation if needed
                    finalText = try await continueIfIncomplete(finalText, context: compose, apiKey: apiKey)
                    var displayFinal = finalText
                    // Prefer decoded 'answer' if JSON-ish:
                    if let a = extractAnswerIfJSON(finalText) { displayFinal = a }
                    // If still JSON-ish or blank, try to derive from tool output directly:
                    if looksLikeJSON(displayFinal) || isBlank(displayFinal) {
                        if let ans = answerFromSearchResult(result) { displayFinal = ans }
                    }
                    // Append sources only if explicitly requested by the user or feature flag enabled
                    let userAskedForLinks = trimmed.range(of: "source", options: .caseInsensitive) != nil ||
                                            trimmed.range(of: "link", options: .caseInsensitive) != nil ||
                                            trimmed.range(of: "where", options: .caseInsensitive) != nil
                    if (SHOW_SOURCES_BY_DEFAULT || userAskedForLinks),
                       displayFinal.range(of: "\nSources", options: .caseInsensitive) == nil {
                        let bullets = extractSourcesBullets(from: result)
                        if !bullets.isEmpty {
                            displayFinal += "\n\nSources\n" + bullets.joined(separator: "\n")
                        }
                    }
                    if isBlank(displayFinal) { displayFinal = "Sorry — I couldn’t generate an answer from the search results." }
                    displayFinal = await applyMovioTone(displayFinal, mood: selectedMood)
                    showAndSpeak(displayFinal)
                } else if action.action == "final" {
                    // Compose a toned final answer using the decorated user text and recent context
                    let directMsgs = buildDirectAnswerMessages(latestUser: trimmed)
                    let direct = try await client.send(messages: directMsgs, apiKey: apiKey)
                    var directText = cleanModelText(direct.choices.first?.message?.content ?? "")
                    directText = try await continueIfIncomplete(directText, context: directMsgs, apiKey: apiKey)
                    if let a = extractAnswerIfJSON(directText) { directText = a }
                    if isBlank(directText) { directText = "Sorry — I couldn’t generate an answer. Please try again." }
                    directText = await applyMovioTone(directText, mood: selectedMood)
                    showAndSpeak(directText)
                } else {
                    // Fallback: request a direct answer (reduced context)
                    print("FALLBACK: plan undecodable, requesting direct answer (reduced context)")
                    let directMsgs = buildDirectAnswerMessages(latestUser: trimmed)
                    let direct = try await client.send(messages: directMsgs, apiKey: apiKey)
                    var directText = cleanModelText(direct.choices.first?.message?.content ?? "")
                    directText = try await continueIfIncomplete(directText, context: directMsgs, apiKey: apiKey)
                    var cleanedDirect = directText
                    if let a = extractAnswerIfJSON(directText) { cleanedDirect = a }
                    if isBlank(cleanedDirect), let entity = inferFocusEntity() {
                        let q = "\(entity) \(trimmed)"
                        let result = try await tavilySearch(query: q)
                        if var ans = answerFromSearchResult(result) {
                            let bullets = extractSourcesBullets(from: result)
                            if !bullets.isEmpty { ans += "\n\nSources\n" + bullets.joined(separator: "\n") }
                            showAndSpeak(ans)
                            return
                        }
                    }
                    if isBlank(cleanedDirect) { cleanedDirect = "Sorry — I couldn’t generate an answer. Please try again." }
                    cleanedDirect = await applyMovioTone(cleanedDirect, mood: selectedMood)
                    showAndSpeak(cleanedDirect)
                }
            } else {
                // Not a tool-directed message; request direct answer (reduced context)
                print("NO ACTION PARSED: requesting direct answer (reduced context)")
                let directMsgs = buildDirectAnswerMessages(latestUser: trimmed)
                let direct = try await client.send(messages: directMsgs, apiKey: apiKey)
                var directText = cleanModelText(direct.choices.first?.message?.content ?? "")
                directText = try await continueIfIncomplete(directText, context: directMsgs, apiKey: apiKey)
                var cleanedFinal = directText
                if let a = extractAnswerIfJSON(directText) { cleanedFinal = a }
                if isBlank(cleanedFinal), let entity = inferFocusEntity() {
                    let q = "\(entity) \(trimmed)"
                    let result = try await tavilySearch(query: q)
                    if var ans = answerFromSearchResult(result) {
                        let bullets = extractSourcesBullets(from: result)
                        if !bullets.isEmpty { ans += "\n\nSources\n" + bullets.joined(separator: "\n") }
                        showAndSpeak(ans)
                        return
                    }
                }
                if isBlank(cleanedFinal) { cleanedFinal = "Sorry — I couldn’t generate an answer. Please try again." }
                cleanedFinal = await applyMovioTone(cleanedFinal, mood: selectedMood)
                showAndSpeak(cleanedFinal)
            }
        } catch {
            errorText = error.localizedDescription
        }
    }
}
        
