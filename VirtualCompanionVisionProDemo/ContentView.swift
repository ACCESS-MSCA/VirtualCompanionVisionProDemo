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
    let message: ChatResponseMessage?
    let finish_reason: String?
}
struct ChatResponse: Codable {
    let id: String?
    let model: String?
    let choices: [ChatChoice]
}

struct ChatResponseContentPart: Codable {
    let type: String?
    let text: String?
}

struct ChatResponseMessage: Codable {
    let role: String?
    let content: String?

    private enum CodingKeys: String, CodingKey {
        case role
        case content
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try? container.decode(String.self, forKey: .role)

        if let plainContent = try? container.decode(String.self, forKey: .content) {
            content = plainContent
            return
        }

        if let parts = try? container.decode([ChatResponseContentPart].self, forKey: .content) {
            let joined = parts
                .compactMap { $0.text?.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
                .joined(separator: " ")
            content = joined.isEmpty ? nil : joined
            return
        }

        content = nil
    }
}

struct UnityChatMessageDefinition: Codable {
    let role: String
    let content: String
}

struct UnityChatMessagesEnvelope: Codable {
    let messages: [UnityChatMessageDefinition]
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

    private func makeRequest(messages: [ChatMessage],
                             apiKey: String,
                             model: String,
                             maxTokens: Int,
                             temperature: Double,
                             topP: Double,
                             responseFormat: ResponseFormat?) throws -> URLRequest {
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
        return req
    }

    func send(messages: [ChatMessage],
              apiKey: String,
              model: String = "openai/gpt-oss-20b",
              maxTokens: Int = 1024,
              temperature: Double = 0.3,
              topP: Double = 0.7,
              responseFormat: ResponseFormat? = nil) async throws -> ChatResponse {
        let req = try makeRequest(messages: messages,
                                  apiKey: apiKey,
                                  model: model,
                                  maxTokens: maxTokens,
                                  temperature: temperature,
                                  topP: topP,
                                  responseFormat: responseFormat)

        let (data, resp, _) = try await timedData(for: req, provider: "NVIDIA", route: "/v1/chat/completions")
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let text = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "NVIDIA", code: (resp as? HTTPURLResponse)?.statusCode ?? -1, userInfo: [NSLocalizedDescriptionKey: text])
        }

        do {
            return try JSONDecoder().decode(ChatResponse.self, from: data)
        } catch {
            let bodyPreview = String(data: data, encoding: .utf8)?
                .replacingOccurrences(of: "\n", with: " ")
                .prefix(320) ?? ""
            throw NSError(
                domain: "NVIDIA",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "Chat decode failed: \(error.localizedDescription). body=\(bodyPreview)"])
        }
    }

    func prewarm(apiKey: String,
                 model: String = "openai/gpt-oss-20b") async throws {
        let req = try makeRequest(
            messages: [
                .init(role: "system", content: "Reply with exactly: OK"),
                .init(role: "user", content: "OK")
            ],
            apiKey: apiKey,
            model: model,
            maxTokens: 8,
            temperature: 0.0,
            topP: 1.0,
            responseFormat: nil
        )

        let started = Date()
        let (data, resp) = try await URLSession.shared.data(for: req)
        let elapsedMs = Int(Date().timeIntervalSince(started) * 1000)
        let statusCode = (resp as? HTTPURLResponse)?.statusCode ?? -1

        print("[AI_PREWARM] provider=NVIDIA route=/v1/chat/completions elapsed_ms=\(elapsedMs) bytes=\(data.count) status=\(statusCode)")

        guard (200..<300).contains(statusCode) else {
            let text = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "NVIDIA", code: statusCode, userInfo: [NSLocalizedDescriptionKey: text])
        }
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
    private enum SessionProfile {
        case playback
        case conversation
    }

    private(set) var synth = AVSpeechSynthesizer()
    private var currentProfile: SessionProfile?

    private func applySessionProfile(_ profile: SessionProfile,
                                     category: AVAudioSession.Category,
                                     mode: AVAudioSession.Mode,
                                     options: AVAudioSession.CategoryOptions) {
        let session = AVAudioSession.sharedInstance()
        do {
            // Reassert category+mode on every transition point. On visionOS, app/audio
            // interruptions can invalidate the active session even if our logical
            // profile did not change.
            try session.setCategory(category, mode: mode, options: options)
            try session.setActive(true)
            currentProfile = profile
        } catch {
            currentProfile = nil
            print("AudioSession error (\(profile)):", error.localizedDescription)
        }
    }
    
    func configurePlaybackSession() {
        applySessionProfile(.playback,
                            category: .playback,
                            mode: .default,
                            options: [.duckOthers])
    }

    func configureRecordingSession() {
        configureConversationSession()
    }

    func configureConversationSession() {
        applySessionProfile(.conversation,
                            category: .playAndRecord,
                            mode: .measurement,
                            options: [.duckOthers, .allowBluetoothHFP, .allowBluetoothA2DP])
    }

    func configureAudioSession() {
        configurePlaybackSession()
    }

    func resetSynth(delegate: AVSpeechSynthesizerDelegate?) {
        let newSynth = AVSpeechSynthesizer()
        newSynth.delegate = delegate
        synth = newSynth
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
    private enum SessionState: String {
        case idle
        case waitingForFirstSpeech
        case capturingSpeech
        case finishing
    }

    private struct AudioDiagnostics {
        var sessionStartedAt: Date?
        var formatDescription: String = "unknown"
        var bufferCount: Int = 0
        var speechLikeBufferCount: Int = 0
        var firstBufferMs: Int?
        var firstSpeechLikeBufferMs: Int?
        var lastBufferMs: Int?
        var maxObservedLevelDb: Float = -160.0
        var lastRecognizerErrorDomain: String?
        var lastRecognizerErrorCode: Int?
        var lastRecognizerErrorDescription: String?
    }

    @Published var isListening = false
    @Published var partialText: String = ""            // live transcript for UI (optional)
    var onFinal: ((String) -> Void)?                   // callback for the final text
    var onPartial: ((String) -> Void)?                 // callback for partial recognition
    var onError: ((String) -> Void)?                   // callback for errors (permissions, session, recognizer, etc.)

    private var recognizer: SFSpeechRecognizer?
    private let audioEngine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var state: SessionState = .idle
    private var currentSessionID: UUID?
    private var initialSpeechTimer: Timer?
    private var endOfSpeechTimer: Timer?
    private var initialSpeechDeadline: Date?
    private var audioDiagnostics = AudioDiagnostics()
    private var hasAuthorizedSpeechRecognition = false
    private var hasAuthorizedMicrophone = false

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
        if hasAuthorizedSpeechRecognition && hasAuthorizedMicrophone {
            DispatchQueue.main.async {
                completion(true)
            }
            return
        }

        let requestMicrophonePermissionIfNeeded = {
            if self.hasAuthorizedMicrophone {
                DispatchQueue.main.async {
                    completion(true)
                }
                return
            }

            mcRequestMicPermission { micOK in
                if micOK {
                    self.hasAuthorizedMicrophone = true
                }

                DispatchQueue.main.async {
                    if !micOK {
                        self.onError?("Microphone permission denied.")
                    }
                    completion(micOK)
                }
            }
        }

        if hasAuthorizedSpeechRecognition {
            requestMicrophonePermissionIfNeeded()
            return
        }

        SFSpeechRecognizer.requestAuthorization { status in
            let speechOK = (status == .authorized)
            if speechOK {
                self.hasAuthorizedSpeechRecognition = true
                requestMicrophonePermissionIfNeeded()
                return
            }

            DispatchQueue.main.async {
                self.onError?("Speech recognition permission denied.")
                completion(false)
            }
        }
    }

    /// Begin listening and streaming audio into the recognizer.
    /// Automatically stops after `seconds` of inactivity.
    func startListening(autoStopAfterSilence seconds: TimeInterval = 1.2,
                        initialSpeechTimeout: TimeInterval = 5.0) throws {
        guard currentSessionID == nil, !audioEngine.isRunning else { return }
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
        self.request = makeStreamingRequest()
        let sessionID = UUID()
        currentSessionID = sessionID
        initialSpeechDeadline = Date().addingTimeInterval(initialSpeechTimeout)
        state = .waitingForFirstSpeech

        // Install a tap on the input node
        let input = audioEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        resetDiagnostics(format: format)
        audioEngine.stop()
        audioEngine.reset()
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.recordInputBuffer(buffer)
            self?.request?.append(buffer)
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            cleanupForNewSession()
            onError?("Audio engine failed to start: \(error.localizedDescription)")
            return
        }
        DispatchQueue.main.async { self.isListening = true }
        print("STT: Listening started")
        print("STT: Session format \(audioDiagnostics.formatDescription)")
        startRecognitionTask(using: recognizer, sessionID: sessionID, autoStopAfterSilence: seconds)
        startInitialSpeechTimer(for: sessionID)
    }
    
    func transcribeFile(atPath path: String) {
        cleanupForNewSession()
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
        let sessionID = UUID()
        currentSessionID = sessionID
        initialSpeechDeadline = nil
        state = .capturingSpeech
        resetDiagnostics(format: nil)

        DispatchQueue.main.async { self.isListening = true }
        print("STT: Transcribing file:", url.lastPathComponent)

        task = recognizer.recognitionTask(with: req) { [weak self] result, error in
            guard let self else { return }
            guard self.currentSessionID == sessionID else { return }

            if let res = result {
                let text = res.bestTranscription.formattedString
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    DispatchQueue.main.async {
                        self.partialText = text
                        self.onPartial?(text)
                    }
                }
                if res.isFinal {
                    print("STT: Final transcript (file):", text)
                    self.finish(sessionID: sessionID, final: trimmed.isEmpty ? nil : text)
                }
            }

            if let e = error {
                self.recordRecognizerError(e)
                let lower = e.localizedDescription.lowercased()
                if lower.contains("canceled") || lower.contains("cancelled") {
                    print("STT: canceled (expected)")
                } else {
                    print("STT file error:", e.localizedDescription)
                    self.fail(sessionID: sessionID, message: "Recognition error: \(e.localizedDescription)")
                }
            }
        }
    }

    /// Stop listening. If `finalize` is true, emit the last partial as final.
    func stopListening(finalize: Bool = true) {
        guard let sessionID = currentSessionID else { return }
        if finalize,
           let finalText = usablePartialText {
            finish(sessionID: sessionID, final: finalText)
        } else {
            finish(sessionID: sessionID, final: nil)
        }
    }

    // MARK: - Internals
    private var usablePartialText: String? {
        let trimmed = partialText.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : partialText
    }

    private func cleanupForNewSession() {
        currentSessionID = nil
        state = .idle
        initialSpeechDeadline = nil
        invalidateTimers()
        request?.endAudio()
        task?.cancel()
        task = nil
        request = nil
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        audioEngine.reset()
    }

    private func startRecognitionTask(using recognizer: SFSpeechRecognizer,
                                      sessionID: UUID,
                                      autoStopAfterSilence seconds: TimeInterval) {
        guard currentSessionID == sessionID, let request else { return }

        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            guard self.currentSessionID == sessionID else { return }

            if let res = result {
                let text = res.bestTranscription.formattedString
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)

                if !trimmed.isEmpty {
                    self.markSpeechDetected()
                    DispatchQueue.main.async {
                        self.partialText = text
                        self.onPartial?(text)
                    }
                    self.resetEndOfSpeechTimer(seconds, sessionID: sessionID)
                }

                if res.isFinal {
                    let finalText = trimmed.isEmpty ? self.usablePartialText : text
                    print("STT: Final transcript: \(finalText ?? "<empty>")")
                    self.finish(sessionID: sessionID, final: finalText)
                }
            }

            if let e = error {
                guard self.currentSessionID == sessionID else { return }
                self.recordRecognizerError(e)

                if self.shouldIgnoreExpectedCancellation(e) {
                    print("STT: Recognition canceled (expected on stop)")
                    return
                }

                if self.shouldRetryForInitialNoSpeech(e, sessionID: sessionID) {
                    print("STT: Retrying after initial no-speech signal")
                    self.request?.endAudio()
                    self.task = nil
                    self.request = self.makeStreamingRequest()
                    self.startRecognitionTask(using: recognizer, sessionID: sessionID, autoStopAfterSilence: seconds)
                    return
                }

                if self.shouldIgnorePostFinalError() {
                    print("STT: Ignoring post-final recognition error:", e.localizedDescription)
                    return
                }

                if self.isSpeechServiceFailure(e) {
                    let diagnostic = self.simulatorDiagnosticSummary(reason: "speech_service_failure")
                    print("STT: Speech service failure:", diagnostic)
#if targetEnvironment(simulator)
                    self.fail(sessionID: sessionID, message: "Recognition error: Simulator speech service failure. \(diagnostic)")
#else
                    self.fail(sessionID: sessionID, message: "Recognition error: Speech service failure. \(diagnostic)")
#endif
                    return
                }

                if e.localizedDescription.lowercased().contains("failed to access assets") {
                    print("STT: On-device speech assets not available on this device/simulator.")
#if targetEnvironment(simulator)
                    self.fail(sessionID: sessionID, message: "Speech recognition assets are not available in the simulator. Please test on a real device.")
#else
                    self.fail(sessionID: sessionID, message: "Speech recognition assets are not available for this language. Check on-device speech settings.")
#endif
                } else {
                    print("STT: Recognition error:", e.localizedDescription)
                    self.fail(sessionID: sessionID, message: "Recognition error: \(e.localizedDescription)")
                }
            }
        }
    }

    private func makeStreamingRequest() -> SFSpeechAudioBufferRecognitionRequest {
        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
#if os(visionOS)
        req.requiresOnDeviceRecognition = true
#endif
        return req
    }

    private func markSpeechDetected() {
        guard state == .waitingForFirstSpeech else { return }
        state = .capturingSpeech
        initialSpeechTimer?.invalidate()
        initialSpeechTimer = nil
        initialSpeechDeadline = nil
    }

    private func finish(sessionID: UUID, final text: String?) {
        guard currentSessionID == sessionID else { return }
        state = .finishing
        currentSessionID = nil
        let trimmedFinal = text?.trimmingCharacters(in: .whitespacesAndNewlines)
        let finalText = (trimmedFinal?.isEmpty == false) ? text : nil
        print("STT: Finishing (final=\(finalText != nil))")
        invalidateTimers()
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        audioEngine.reset()
        request?.endAudio()
        task?.cancel()
        task = nil
        request = nil
        DispatchQueue.main.async {
            self.isListening = false
            self.state = .idle
            self.initialSpeechDeadline = nil
            self.audioDiagnostics = AudioDiagnostics()
            if let t = finalText {
                self.onFinal?(t)
            }
        }
    }

    private func fail(sessionID: UUID, message: String) {
        guard currentSessionID == sessionID else { return }
        state = .finishing
        currentSessionID = nil
        print("STT: Failing session with message:", message)
        invalidateTimers()
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        audioEngine.reset()
        request?.endAudio()
        task?.cancel()
        task = nil
        request = nil
        DispatchQueue.main.async {
            self.isListening = false
            self.state = .idle
            self.initialSpeechDeadline = nil
            self.audioDiagnostics = AudioDiagnostics()
            self.onError?(message)
        }
    }

    private func invalidateTimers() {
        initialSpeechTimer?.invalidate()
        initialSpeechTimer = nil
        endOfSpeechTimer?.invalidate()
        endOfSpeechTimer = nil
    }

    private func startInitialSpeechTimer(for sessionID: UUID) {
        initialSpeechTimer?.invalidate()
        guard let deadline = initialSpeechDeadline else { return }
        let remaining = deadline.timeIntervalSinceNow
        guard remaining > 0 else {
            fail(sessionID: sessionID, message: "Recognition error: No speech detected")
            return
        }

        initialSpeechTimer = Timer.scheduledTimer(withTimeInterval: remaining, repeats: false) { [weak self] _ in
            guard let self else { return }
            guard self.currentSessionID == sessionID else { return }
            guard self.state == .waitingForFirstSpeech else { return }
            let diagnostic = self.simulatorDiagnosticSummary(reason: "initial_timeout")
            print("STT: Initial speech timeout:", diagnostic)
            self.fail(sessionID: sessionID, message: self.buildNoSpeechDetectedMessage(diagnostic: diagnostic))
        }
    }

    private func resetEndOfSpeechTimer(_ seconds: TimeInterval, sessionID: UUID) {
        endOfSpeechTimer?.invalidate()
        endOfSpeechTimer = Timer.scheduledTimer(withTimeInterval: seconds, repeats: false) { [weak self] _ in
            guard let self else { return }
            guard self.currentSessionID == sessionID else { return }
            if let finalText = self.usablePartialText {
                self.finish(sessionID: sessionID, final: finalText)
            } else {
                self.finish(sessionID: sessionID, final: nil)
            }
        }
    }

    private func shouldIgnoreExpectedCancellation(_ error: Error) -> Bool {
        let lower = error.localizedDescription.lowercased()
        return lower.contains("canceled") || lower.contains("cancelled")
    }

    private func shouldRetryForInitialNoSpeech(_ error: Error, sessionID: UUID) -> Bool {
        guard currentSessionID == sessionID else { return false }
        guard state == .waitingForFirstSpeech else { return false }
        guard usablePartialText == nil else { return false }
        guard let deadline = initialSpeechDeadline, deadline.timeIntervalSinceNow > 0 else { return false }

        let lower = error.localizedDescription.lowercased()
        return lower.contains("no speech detected")
    }

    private func shouldIgnorePostFinalError() -> Bool {
        state == .finishing || currentSessionID == nil
    }

    private func resetDiagnostics(format: AVAudioFormat?) {
        audioDiagnostics = AudioDiagnostics()
        audioDiagnostics.sessionStartedAt = Date()
        audioDiagnostics.formatDescription = describeAudioFormat(format)
    }

    private func describeAudioFormat(_ format: AVAudioFormat?) -> String {
        guard let format else { return "file_request" }

        let commonFormat: String
        switch format.commonFormat {
        case .pcmFormatFloat32: commonFormat = "float32"
        case .pcmFormatFloat64: commonFormat = "float64"
        case .pcmFormatInt16: commonFormat = "int16"
        case .pcmFormatInt32: commonFormat = "int32"
        case .otherFormat: commonFormat = "other"
        @unknown default: commonFormat = "unknown"
        }

        return "sr=\(Int(format.sampleRate)) ch=\(format.channelCount) \(commonFormat) interleaved=\(format.isInterleaved)"
    }

    private func recordInputBuffer(_ buffer: AVAudioPCMBuffer) {
        audioDiagnostics.bufferCount += 1

        let elapsedMs = elapsedMillisecondsSinceSessionStart()
        if audioDiagnostics.firstBufferMs == nil {
            audioDiagnostics.firstBufferMs = elapsedMs
            print("STT: First audio buffer received at \(elapsedMs)ms")
        }
        audioDiagnostics.lastBufferMs = elapsedMs

        guard let peakDb = estimatePeakLevelDb(buffer) else { return }
        audioDiagnostics.maxObservedLevelDb = max(audioDiagnostics.maxObservedLevelDb, peakDb)

        if peakDb > -45.0 {
            audioDiagnostics.speechLikeBufferCount += 1
            if audioDiagnostics.firstSpeechLikeBufferMs == nil {
                audioDiagnostics.firstSpeechLikeBufferMs = elapsedMs
                print("STT: First speech-like audio detected at \(elapsedMs)ms (peak_db=\(formatDb(peakDb)))")
            }
        }
    }

    private func estimatePeakLevelDb(_ buffer: AVAudioPCMBuffer) -> Float? {
        let frameLength = Int(buffer.frameLength)
        guard frameLength > 0 else { return nil }

        if let channelData = buffer.floatChannelData {
            let channelCount = Int(buffer.format.channelCount)
            var peak: Float = 0
            for channelIndex in 0..<channelCount {
                let samples = UnsafeBufferPointer(start: channelData[channelIndex], count: frameLength)
                for sample in samples {
                    let magnitude = abs(sample)
                    if magnitude > peak {
                        peak = magnitude
                    }
                }
            }

            if peak <= 0 { return -160.0 }
            return max(-160.0, 20.0 * log10f(peak))
        }

        return nil
    }

    private func elapsedMillisecondsSinceSessionStart() -> Int {
        guard let startedAt = audioDiagnostics.sessionStartedAt else { return 0 }
        return max(0, Int(Date().timeIntervalSince(startedAt) * 1000.0))
    }

    private func recordRecognizerError(_ error: Error) {
        let nsError = error as NSError
        audioDiagnostics.lastRecognizerErrorDomain = nsError.domain
        audioDiagnostics.lastRecognizerErrorCode = nsError.code
        audioDiagnostics.lastRecognizerErrorDescription = nsError.localizedDescription
        print("STT: Recognizer error domain=\(nsError.domain) code=\(nsError.code) desc=\(nsError.localizedDescription)")
    }

    private func isSpeechServiceFailure(_ error: Error) -> Bool {
        let nsError = error as NSError
        if nsError.domain == "kAFAssistantErrorDomain" && nsError.code == 1101 {
            return true
        }

        let lower = nsError.localizedDescription.lowercased()
        return lower.contains("localspeechrecognition") || lower.contains("assistant")
    }

    private func simulatorDiagnosticSummary(reason: String) -> String {
        var parts: [String] = ["[sim-stt]", "reason=\(reason)", "buffers=\(audioDiagnostics.bufferCount)"]
        parts.append("speech_like_buffers=\(audioDiagnostics.speechLikeBufferCount)")
        parts.append("format=\(audioDiagnostics.formatDescription)")

        if let firstBufferMs = audioDiagnostics.firstBufferMs {
            parts.append("first_buffer_ms=\(firstBufferMs)")
        } else {
            parts.append("first_buffer_ms=none")
        }

        if let firstSpeechLikeBufferMs = audioDiagnostics.firstSpeechLikeBufferMs {
            parts.append("first_speech_like_ms=\(firstSpeechLikeBufferMs)")
        } else {
            parts.append("first_speech_like_ms=none")
        }

        if let lastBufferMs = audioDiagnostics.lastBufferMs {
            parts.append("last_buffer_ms=\(lastBufferMs)")
        }

        parts.append("max_db=\(formatDb(audioDiagnostics.maxObservedLevelDb))")

        if let domain = audioDiagnostics.lastRecognizerErrorDomain {
            parts.append("last_error_domain=\(domain)")
        }
        if let code = audioDiagnostics.lastRecognizerErrorCode {
            parts.append("last_error_code=\(code)")
        }
        if let description = audioDiagnostics.lastRecognizerErrorDescription?.replacingOccurrences(of: " ", with: "_") {
            parts.append("last_error=\(description)")
        }

        return parts.joined(separator: " ")
    }

    private func buildNoSpeechDetectedMessage(diagnostic: String) -> String {
#if targetEnvironment(simulator)
        if audioDiagnostics.bufferCount == 0 {
            return "Recognition error: No speech detected. Simulator delivered no input audio buffers. \(diagnostic)"
        }
        if audioDiagnostics.speechLikeBufferCount == 0 {
            return "Recognition error: No speech detected. Audio buffers arrived but stayed below the speech threshold. \(diagnostic)"
        }
        if let domain = audioDiagnostics.lastRecognizerErrorDomain,
           domain == "kAFAssistantErrorDomain",
           audioDiagnostics.lastRecognizerErrorCode == 1101 {
            return "Recognition error: No speech detected. Audio reached the simulator, but the local Speech service failed. \(diagnostic)"
        }
        return "Recognition error: No speech detected. Audio reached the simulator but no transcript was produced. \(diagnostic)"
#else
        return "Recognition error: No speech detected"
#endif
    }

    private func formatDb(_ value: Float) -> String {
        String(format: "%.1f", value)
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
    private enum TTSPrimeReason {
        case startup
        case voiceSelection
    }

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
    var voiceGender: VoiceGender = .neutral
    var voicePitch: Float = 1.0
    var voiceLanguage: String = ""
    var voiceIdentifier: String? = nil

    // Optional callback back into Unity
    private var unityCallback: UnityCallback?
    private var unityBytesCallback: UnityBytesCallback?
    private var activeResponseTask: Task<Void, Never>?
    private var activeResponseRequestID = UUID()
    private var cachedTTSVoice: AVSpeechSynthesisVoice?
    private var isPrewarmingTTS = false
    private var hasPrewarmedTTS = false
    private var primedTTSVoiceKey: String?
    private var activeTTSPrimeReason: TTSPrimeReason?
    private var activeTTSPrimeVoiceKey: String?
    private var pendingSpeakAfterPrime: String?
    private var shouldEmitUserFacingTTSStarted = false
    private var isPrewarmingChat = false
    private var hasPrewarmedChat = false
    private var isAwaitingNativeTTSStop = false
    private var hasSentAIReady = false
    private let traceOrigin = ProcessInfo.processInfo.systemUptime
    override init() {
        super.init()
        // STT -> Unity forwarding (dispatch to main to be safe with Unity APIs)
        stt.onPartial = { [weak self] text in self?.sendToUnity("stt_partial:" + text) }
        stt.onFinal   = { [weak self] text in
            self?.speech.configureConversationSession()
            self?.sendToUnity("stt_final:" + text)
        }
        stt.onError   = { [weak self] msg  in
            self?.speech.configureConversationSession()
            self?.sendToUnity("error:" + msg)
        }
        // Use a single conversation session inside the Unity bridge to
        // avoid expensive profile switches between TTS and live microphone.
        speech.configureConversationSession()
        speech.synth.delegate = self
        cachedTTSVoice = resolveTTSVoice()
        AIRequestMetricsCenter.shared.onMetric = { [weak self] metric in
            self?.sendMetricToUnity(metric)
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) { [weak self] in
            self?.prewarmResponsePipelineIfNeeded()
        }
        trace("bridge_init")
    }

    // MARK: - Unity callback wiring
    func registerCallback(_ cb: UnityCallback?) { self.unityCallback = cb }
    private func sendToUnity(_ message: String) {
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

    private func trace(_ event: String) {
        let elapsedMs = Int((ProcessInfo.processInfo.systemUptime - traceOrigin) * 1000.0)
        print("[AI_TRACE_SWIFT] t=\(elapsedMs)ms \(event)")
    }

    private func secretTag(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "empty" }
        let prefixLen = min(6, trimmed.count)
        let prefix = String(trimmed.prefix(prefixLen))
        return "\(prefix)...(len=\(trimmed.count))"
    }

    private func describeVoice(_ voice: AVSpeechSynthesisVoice?) -> String {
        guard let voice else { return "nil" }

        let gender: String
        switch voice.gender {
        case .male: gender = "male"
        case .female: gender = "female"
        case .unspecified: gender = "unspecified"
        @unknown default: gender = "unknown"
        }

        let quality: String
        switch voice.quality {
        case .enhanced: quality = "enhanced"
        case .premium: quality = "premium"
        case .default: quality = "default"
        @unknown default: quality = "unknown"
        }

        return "name=\(voice.name) lang=\(voice.language) gender=\(gender) quality=\(quality) id=\(voice.identifier)"
    }

    private func voiceKey(for voice: AVSpeechSynthesisVoice?) -> String {
        guard let voice else { return "nil" }
        return voice.identifier.lowercased()
    }

    private func refreshSynth(reason: String) {
        speech.resetSynth(delegate: self)
        primedTTSVoiceKey = nil
        shouldEmitUserFacingTTSStarted = false
        trace("tts_synth_reset reason=\(reason)")
    }

    private func resolveAndCacheTTSVoice() -> AVSpeechSynthesisVoice? {
        let voice = resolveTTSVoice()
        cachedTTSVoice = voice
        return voice
    }

    private func handleVoiceSelectionChanged(reason: String) {
        let voice = resolveAndCacheTTSVoice()
        let key = voiceKey(for: voice)
        primedTTSVoiceKey = nil
        pendingSpeakAfterPrime = nil
        ttsPending = 0

        if isPrewarmingTTS {
            speech.synth.stopSpeaking(at: .immediate)
            isPrewarmingTTS = false
            activeTTSPrimeReason = nil
            activeTTSPrimeVoiceKey = nil
        }

        if speech.synth.isSpeaking || speech.synth.isPaused {
            speech.synth.stopSpeaking(at: .immediate)
        }

        refreshSynth(reason: reason)
        maybePrimeVoiceIfIdle(voice: voice, key: key, reason: .voiceSelection)
    }

    private func maybePrimeVoiceIfIdle(voice: AVSpeechSynthesisVoice?,
                                       key: String,
                                       reason: TTSPrimeReason) {
        guard let voice else { return }
        guard !isPrewarmingTTS else { return }
        guard !speech.synth.isSpeaking, !speech.synth.isPaused else { return }
        guard primedTTSVoiceKey != key else { return }
        startVoicePrime(voice: voice, key: key, reason: reason)
    }

    private func startVoicePrime(voice: AVSpeechSynthesisVoice,
                                 key: String,
                                 reason: TTSPrimeReason) {
        guard !isPrewarmingTTS else { return }

        speech.configureConversationSession()
        activeTTSPrimeReason = reason
        activeTTSPrimeVoiceKey = key
        isPrewarmingTTS = true

        if reason == .startup {
            trace("tts_prewarm_begin")
        } else {
            trace("tts_voice_prime_begin \(describeVoice(voice))")
        }

        let warmupUtterance = AVSpeechUtterance(string: "warmup")
        warmupUtterance.voice = voice
        warmupUtterance.volume = 0.0
        warmupUtterance.rate = 0.5
        warmupUtterance.pitchMultiplier = 1.0
        speech.synth.speak(warmupUtterance)
    }

    private func finishVoicePrime(cancelled: Bool) {
        let primeReason = activeTTSPrimeReason
        let primeKey = activeTTSPrimeVoiceKey
        activeTTSPrimeReason = nil
        activeTTSPrimeVoiceKey = nil
        isPrewarmingTTS = false

        if let primeKey, !cancelled {
            primedTTSVoiceKey = primeKey
        }
        hasPrewarmedTTS = !cancelled

        switch primeReason {
        case .startup:
            trace(cancelled ? "tts_prewarm_cancelled" : "tts_prewarm_finished")
        case .voiceSelection:
            trace(cancelled ? "tts_voice_prime_cancelled" : "tts_voice_prime_finished")
        case .none:
            break
        }

        if !cancelled {
            notifyAIReadyIfNeeded()
        }

        if primeReason == .voiceSelection && !cancelled {
            flushPendingSpeakAfterPrimeIfNeeded()
        }
    }

    private func flushPendingSpeakAfterPrimeIfNeeded() {
        guard let pendingText = pendingSpeakAfterPrime else { return }
        pendingSpeakAfterPrime = nil
        performSpeak(pendingText)
    }

    private func performSpeak(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        isAwaitingNativeTTSStop = false
        trace("speak_begin chars=\(trimmed.count)")
        if isPrewarmingTTS {
            speech.synth.stopSpeaking(at: .immediate)
        }
        hasPrewarmedTTS = true
        speech.configureConversationSession()
        ttsBatchId = UUID()
        let batch = ttsBatchId

        if speech.synth.isSpeaking || speech.synth.isPaused {
            speech.synth.stopSpeaking(at: .immediate)
        }

        let sentences = mcSplitIntoSentences(trimmed)
        ttsPending = sentences.count
        shouldEmitUserFacingTTSStarted = ttsPending > 0

        let voice = resolveAndCacheTTSVoice()
        let (rate, pitch) = ttsParams(for: mood)
        let finalPitch = max(0.5, min(voicePitch * pitch, 2.0))
        trace("tts_voice_resolved \(describeVoice(voice)) requestedGender=\(voiceGender.rawValue) configuredLanguage=\(voiceLanguage) configuredIdentifier=\(secretTag(voiceIdentifier ?? "")) configuredPitch=\(voicePitch) mood=\(mood.rawValue) finalPitch=\(finalPitch)")
        for (i, sentence) in sentences.enumerated() {
            let u = AVSpeechUtterance(string: sentence)
            u.voice = voice
            u.rate  = rate
            u.pitchMultiplier = finalPitch
            u.postUtteranceDelay = (i == 0) ? 0.0 : 0.12
            speech.synth.speak(u)
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            guard let self else { return }
            if self.ttsBatchId == batch, self.ttsPending == 0 {
                self.sendToUnity("tts_finished")
            }
        }
    }

    func sttFromFile(path: String) {
        trace("stt_file_start path=\((path as NSString).lastPathComponent)")
        stt.ensureAuthorization { [weak self] ok in
            guard let self else { return }
            guard ok else {
                self.sendToUnity("error:Speech permission is required.")
                return
            }
            self.stt.onPartial = { [weak self] t in self?.sendToUnity("stt_partial:" + t) }
            self.stt.onFinal   = { [weak self] t in
                self?.speech.configureConversationSession()
                self?.sendToUnity("stt_final:" + t)
            }
            self.stt.onError   = { [weak self] e in
                self?.speech.configureConversationSession()
                self?.sendToUnity("error:" + e)
            }

            self.stt.transcribeFile(atPath: path)
        }
    }

    private var ttsPending = 0
    private var ttsBatchId = UUID()
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                           didStart utterance: AVSpeechUtterance) {
        guard synthesizer === speech.synth else {
            trace("tts_callback_ignored_stale_synth event=didStart")
            return
        }
        if isPrewarmingTTS {
            if activeTTSPrimeReason == .startup {
                trace("tts_prewarm_started")
            } else {
                trace("tts_voice_prime_started")
            }
            return
        }
        trace("tts_started")
        if shouldEmitUserFacingTTSStarted {
            shouldEmitUserFacingTTSStarted = false
            sendToUnity("tts_started")
        }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                           didFinish utterance: AVSpeechUtterance) {
        guard synthesizer === speech.synth else {
            trace("tts_callback_ignored_stale_synth event=didFinish")
            return
        }
        if isPrewarmingTTS {
            finishVoicePrime(cancelled: false)
            return
        }
        if isAwaitingNativeTTSStop {
            trace("tts_stop_finish_callback")
            waitForNativeTTSStopCompletion()
            return
        }
        // Solo avisamos cuando ya terminó TODO
        ttsPending = max(0, ttsPending - 1)
        trace("tts_utterance_finished pending=\(ttsPending)")
        if ttsPending == 0 {
            sendToUnity("tts_finished")
        }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                           didCancel utterance: AVSpeechUtterance) {
        guard synthesizer === speech.synth else {
            trace("tts_callback_ignored_stale_synth event=didCancel")
            return
        }
        if isPrewarmingTTS {
            finishVoicePrime(cancelled: true)
            return
        }
        if isAwaitingNativeTTSStop {
            trace("tts_stop_cancel_callback")
            finalizeNativeTTSStopIfNeeded()
            return
        }
        trace("tts_cancelled")
        ttsPending = 0
        shouldEmitUserFacingTTSStarted = false
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

    private func resolveTTSVoice() -> AVSpeechSynthesisVoice? {
        func isSelectableVoice(_ voice: AVSpeechSynthesisVoice) -> Bool {
            let identifier = voice.identifier.lowercased()
            return !identifier.hasPrefix("com.apple.eloquence.")
                && !identifier.hasPrefix("com.apple.speech.synthesis.voice.")
        }

        if let identifier = voiceIdentifier,
           let exactVoice = AVSpeechSynthesisVoice(identifier: identifier) {
            return exactVoice
        }

        let configuredLanguage = voiceLanguage.trimmingCharacters(in: .whitespacesAndNewlines)
        let preferredLanguage: String
        if configuredLanguage.isEmpty {
            switch voiceGender {
            case .male:
                preferredLanguage = "en-GB"
            case .female:
                preferredLanguage = "en-US"
            case .neutral:
                preferredLanguage = "en-GB"
            }
        } else {
            preferredLanguage = configuredLanguage
        }

        let fallbackLanguage = AVSpeechSynthesisVoice.currentLanguageCode()
        let allVoices = AVSpeechSynthesisVoice.speechVoices()
        let languageMatched = allVoices.filter {
            $0.language.caseInsensitiveCompare(preferredLanguage) == .orderedSame
        }
        let regionalMatched = languageMatched.isEmpty
            ? allVoices.filter { $0.language.lowercased().hasPrefix(String(preferredLanguage.lowercased().prefix(2))) }
            : languageMatched
        let primaryPool = regionalMatched.isEmpty ? allVoices : regionalMatched
        let selectablePool = primaryPool.filter(isSelectableVoice)
        let rankedPool = selectablePool.isEmpty ? primaryPool : selectablePool

        func genderScore(for voice: AVSpeechSynthesisVoice) -> Int {
            switch voiceGender {
            case .male:
                switch voice.gender {
                case .male: return 3
                case .unspecified: return 1
                default: return 0
                }
            case .female:
                switch voice.gender {
                case .female: return 3
                case .unspecified: return 1
                default: return 0
                }
            case .neutral:
                switch voice.gender {
                case .unspecified: return 2
                default: return 1
                }
            }
        }

        func qualityScore(for voice: AVSpeechSynthesisVoice) -> Int {
            switch voice.quality {
            case .enhanced: return 1
            default: return 0
            }
        }

        if let preferred = rankedPool
            .sorted(by: {
                let leftGender = genderScore(for: $0)
                let rightGender = genderScore(for: $1)
                if leftGender != rightGender { return leftGender > rightGender }

                let leftQuality = qualityScore(for: $0)
                let rightQuality = qualityScore(for: $1)
                if leftQuality != rightQuality { return leftQuality > rightQuality }

                return $0.name < $1.name
            })
            .first(where: { genderScore(for: $0) > 0 }) {
            return preferred
        }

        return AVSpeechSynthesisVoice(language: preferredLanguage)
            ?? AVSpeechSynthesisVoice(language: fallbackLanguage)
            ?? AVSpeechSynthesisVoice(language: "en-GB")
            ?? AVSpeechSynthesisVoice(language: "en-US")
    }

    private func prewarmResponsePipelineIfNeeded() {
        guard !hasPrewarmedTTS, !isPrewarmingTTS else { return }
        speech.configureConversationSession()
        guard let voice = resolveAndCacheTTSVoice() else {
            trace("tts_prewarm_skipped_no_voice")
            return
        }
        startVoicePrime(voice: voice,
                        key: voiceKey(for: voice),
                        reason: .startup)
    }

    private func prewarmChatRequestIfNeeded() {
        guard !hasPrewarmedChat, !isPrewarmingChat else { return }

        let trimmedKey = nvidiaKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedKey.isEmpty else {
            trace("chat_prewarm_skipped_no_key")
            return
        }

        isPrewarmingChat = true
        let modelForChat = resolvedTextModel()
        trace("chat_prewarm_begin model=\(modelForChat) nvidia_key=\(secretTag(trimmedKey))")

        Task { [weak self] in
            guard let self else { return }
            let prewarmSucceeded = await self.prewarmChatWithModelFallback(apiKey: trimmedKey, preferredModel: modelForChat)
            DispatchQueue.main.async {
                self.isPrewarmingChat = false
                self.hasPrewarmedChat = prewarmSucceeded
                if prewarmSucceeded {
                    self.trace("chat_prewarm_finished")
                    self.notifyAIReadyIfNeeded()
                } else {
                    self.trace("chat_prewarm_not_ready")
                }
            }
        }
    }

    private func notifyAIReadyIfNeeded() {
        guard hasPrewarmedTTS, hasPrewarmedChat else { return }
        guard !hasSentAIReady else { return }

        hasSentAIReady = true
        trace("ai_ready")
        sendToUnity("ai_ready")
    }

    // MARK: - Public operations
    func setConfig(nvidiaKey: String, tavilyKey: String, hfToken: String, hfModel: String?) {
        self.nvidiaKey = nvidiaKey
        self.tavilyKey = tavilyKey
        self.hfToken   = hfToken
        if let m = hfModel, !m.isEmpty { self.hfModel = m }
        trace("config_applied chat_model=\(resolvedTextModel()) nvidia_key=\(secretTag(self.nvidiaKey)) hf_token=\(secretTag(self.hfToken))")
        prewarmChatRequestIfNeeded()
    }

    func setMood(_ m: MoodTag) { self.mood = m }

    func setVoiceGender(_ gender: VoiceGender) {
        voiceGender = gender
        voiceIdentifier = nil
        handleVoiceSelectionChanged(reason: "voice_gender_changed")
    }

    func setVoicePitch(_ pitch: Float) {
        voicePitch = max(0.5, min(pitch, 2.0))
    }

    func setVoiceLanguage(_ language: String) {
        voiceLanguage = language.trimmingCharacters(in: .whitespacesAndNewlines)
        handleVoiceSelectionChanged(reason: "voice_language_changed")
    }

    func setVoiceIdentifier(_ identifier: String?) {
        let trimmed = identifier?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        voiceIdentifier = trimmed.isEmpty ? nil : trimmed
        handleVoiceSelectionChanged(reason: "voice_identifier_changed")
    }

    func availableVoicesSummary() -> String {
        AVSpeechSynthesisVoice.speechVoices()
            .map { voice in
                let gender: String
                switch voice.gender {
                case .male: gender = "male"
                case .female: gender = "female"
                case .unspecified: gender = "unspecified"
                @unknown default: gender = "unknown"
                }

                let quality: String
                switch voice.quality {
                case .default: quality = "default"
                case .enhanced: quality = "enhanced"
                case .premium: quality = "premium"
                @unknown default: quality = "unknown"
                }

                return "\(voice.name) | \(voice.language) | \(gender) | \(voice.identifier) | \(quality)"
            }
            .joined(separator: "\n")
    }

    func sendAvailableVoicesToUnity() {
        sendToUnity("voices:" + availableVoicesSummary())
    }

    private var fallbackTextModel: String { "openai/gpt-oss-20b" }
    private let maxUnitySystemPromptChars = 4000
    private let maxUnityFewShotPayloadChars = 24000
    private let maxUnityFewShotMessages = 12

    private func resolvedTextModel() -> String {
        let configured = hfModel.trimmingCharacters(in: .whitespacesAndNewlines)
        return configured.isEmpty ? fallbackTextModel : configured
    }

    private func isNvidia404(_ error: Error) -> Bool {
        let nsError = error as NSError
        return nsError.domain == "NVIDIA" && nsError.code == 404
    }

    private func prewarmChatWithModelFallback(apiKey: String, preferredModel: String) async -> Bool {
        do {
            try await nvidia.prewarm(apiKey: apiKey, model: preferredModel)
            return true
        } catch {
            guard isNvidia404(error),
                  preferredModel != fallbackTextModel else {
                trace("chat_prewarm_failed error=\(error.localizedDescription)")
                print("AI chat warmup request failed:", error.localizedDescription)
                return false
            }

            trace("chat_prewarm_model_404 fallback_from=\(preferredModel) fallback_to=\(fallbackTextModel)")
            do {
                try await nvidia.prewarm(apiKey: apiKey, model: fallbackTextModel)
                trace("chat_prewarm_fallback_succeeded model=\(fallbackTextModel)")
                return true
            } catch {
                trace("chat_prewarm_fallback_failed error=\(error.localizedDescription)")
                print("AI chat warmup fallback failed:", error.localizedDescription)
                return false
            }
        }
    }

    private func sendTextChatWithModelFallback(messages: [ChatMessage],
                                               apiKey: String,
                                               preferredModel: String,
                                               fallbackModel: String,
                                               maxTokens: Int,
                                               temperature: Double,
                                               topP: Double,
                                               responseFormat: ResponseFormat?) async throws -> ChatResponse {
        do {
            return try await nvidia.send(messages: messages,
                                         apiKey: apiKey,
                                         model: preferredModel,
                                         maxTokens: maxTokens,
                                         temperature: temperature,
                                         topP: topP,
                                         responseFormat: responseFormat)
        } catch {
            guard isNvidia404(error),
                  preferredModel != fallbackModel else {
                throw error
            }

            trace("chat_model_404_fallback from=\(preferredModel) to=\(fallbackModel)")
            return try await nvidia.send(messages: messages,
                                         apiKey: apiKey,
                                         model: fallbackModel,
                                         maxTokens: maxTokens,
                                         temperature: temperature,
                                         topP: topP,
                                         responseFormat: responseFormat)
        }
    }

    // STT control
    func startSTT() {
        trace("stt_start_requested")
        // Stop TTS to avoid echo
        speech.synth.stopSpeaking(at: .immediate)
        stt.ensureAuthorization { [weak self] ok in
            guard let self else { return }
            guard ok else { self.sendToUnity("error:Microphone/Speech permission is required."); return }
            do {
                self.trace("stt_start_authorized")
                self.speech.configureConversationSession()
#if targetEnvironment(simulator)
                try self.stt.startListening(autoStopAfterSilence: 2.0, initialSpeechTimeout: 7.5)
#else
                try self.stt.startListening(autoStopAfterSilence: 1.6)
#endif
            } catch {
                self.sendToUnity("error:Could not start microphone: \(error.localizedDescription)")
            }
        }
    }
    func stopSTT(finalize: Bool) {
        trace("stt_stop_requested finalize=\(finalize)")
        stt.stopListening(finalize: finalize)
    }

    // TTS
    func speak(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let voice = resolveAndCacheTTSVoice()
        let key = voiceKey(for: voice)

        if primedTTSVoiceKey != key {
            pendingSpeakAfterPrime = trimmed
            if activeTTSPrimeVoiceKey != key {
                refreshSynth(reason: "prime_before_speak")
                maybePrimeVoiceIfIdle(voice: voice, key: key, reason: .voiceSelection)
            }
            return
        }

        performSpeak(trimmed)
    }
    func stopSpeak() {
        trace("stop_speak_requested hasTask=\(activeResponseTask != nil) isSpeaking=\(speech.synth.isSpeaking)")
        abandonActiveResponseTask()
        ttsBatchId = UUID()
        if speech.synth.isSpeaking || speech.synth.isPaused {
            ttsPending = 0
            shouldEmitUserFacingTTSStarted = false
            isAwaitingNativeTTSStop = true
            speech.synth.stopSpeaking(at: .immediate)
            waitForNativeTTSStopCompletion()
            return
        }

        speech.synth.stopSpeaking(at: .immediate)
        pendingSpeakAfterPrime = nil
        refreshSynth(reason: "stop_without_native_speech")
        trace("tts_stop_immediate_no_native_speech")
        sendToUnity("tts_finished")
    }

    // GPT-OSS text (minimal, no tool/search step here)
    func askText(_ prompt: String) {
        askText(prompt, preferredModel: nil, fallbackModel: nil, systemPrompt: nil, fewShotMessagesJson: nil)
    }

    func askText(_ prompt: String, preferredModel: String?, fallbackModel: String?) {
        askText(prompt, preferredModel: preferredModel, fallbackModel: fallbackModel, systemPrompt: nil, fewShotMessagesJson: nil)
    }

    func askText(_ prompt: String,
                 preferredModel: String?,
                 fallbackModel: String?,
                 systemPrompt: String?,
                 fewShotMessagesJson: String?) {
        cancelActiveResponseTask()
        let requestID = UUID()
        activeResponseRequestID = requestID
        let preferredTrimmed = preferredModel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let fallbackTrimmed = fallbackModel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let resolvedPreferredModel = preferredTrimmed.isEmpty ? resolvedTextModel() : preferredTrimmed
        let resolvedFallbackModel = fallbackTrimmed.isEmpty ? fallbackTextModel : fallbackTrimmed
        let resolvedSystemPrompt = resolveSystemPrompt(systemPrompt)
        let resolvedFewShotMessages = resolveFewShotMessages(fewShotMessagesJson)
        trace("askText_begin request=\(requestID.uuidString.prefix(8)) chars=\(prompt.count) model=\(resolvedPreferredModel) fallback=\(resolvedFallbackModel) fewshot_count=\(resolvedFewShotMessages.count)")
        activeResponseTask = Task { [weak self] in
            guard let self else { return }
            defer {
                if self.activeResponseRequestID == requestID {
                    self.activeResponseTask = nil
                }
            }

            do {
                let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmedPrompt.isEmpty else {
                    self.trace("askText_ignored_empty_prompt request=\(requestID.uuidString.prefix(8))")
                    self.sendResponseMessageToUnity("error:Empty prompt.", requestID: requestID)
                    return
                }
                let hasStyleDirective = trimmedPrompt.contains("<style>") && trimmedPrompt.contains("ToneDirective:")
                let user = hasStyleDirective
                    ? trimmedPrompt
                    : (styleBlock(for: mood) + "\n\n" + trimmedPrompt)
                var msgs: [ChatMessage] = [.init(role: "system", content: resolvedSystemPrompt)]
                self.trace("askText_style_source source=\(hasStyleDirective ? "unity" : "swift_fallback")")
                msgs.append(contentsOf: resolvedFewShotMessages)
                // Actual turn
                msgs.append(.init(role: "user", content: user))
                try Task.checkCancellation()
                let res = try await sendTextChatWithModelFallback(messages: msgs,
                                                                  apiKey: nvidiaKey,
                                                                  preferredModel: resolvedPreferredModel,
                                                                  fallbackModel: resolvedFallbackModel,
                                                                  maxTokens: 480,
                                                                  temperature: 0.65,
                                                                  topP: 0.95,
                                                                  responseFormat: nil)
                self.trace("askText_network_done request=\(requestID.uuidString.prefix(8))")
                try Task.checkCancellation()
                guard self.isCurrentResponseRequest(requestID) else { return }
                let answer = (res.choices.first?.message?.content ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                print("Respuesta de nvidia: \(answer)")
                try Task.checkCancellation()
                guard self.isCurrentResponseRequest(requestID) else { return }
                self.trace("askText_answer_ready request=\(requestID.uuidString.prefix(8)) chars=\(answer.count)")
                if answer.isEmpty { self.sendResponseMessageToUnity("answer:Sorry — I couldn’t generate an answer.", requestID: requestID) }
                else { self.sendResponseMessageToUnity("answer:" + answer, requestID: requestID) }
            } catch is CancellationError {
                self.trace("askText_cancelled request=\(requestID.uuidString.prefix(8))")
                print("askText cancelled")
            } catch {
                guard self.isCurrentResponseRequest(requestID) else { return }
                self.trace("askText_error request=\(requestID.uuidString.prefix(8)) error=\(error.localizedDescription)")
                self.sendResponseMessageToUnity("error:" + error.localizedDescription, requestID: requestID)
            }
        }
    }

    private func resolveSystemPrompt(_ systemPrompt: String?) -> String {
        let trimmed = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if trimmed.isEmpty {
            trace("askText_system_prompt_source source=swift_fallback reason=empty")
            return defaultSystemPrompt()
        }

        if trimmed.count > maxUnitySystemPromptChars {
            trace("askText_system_prompt_source source=swift_fallback reason=too_large chars=\(trimmed.count)")
            return defaultSystemPrompt()
        }

        trace("askText_system_prompt_source source=unity chars=\(trimmed.count)")
        return trimmed
    }

    private func defaultSystemPrompt() -> String {
        """
        You are Movio, a friendly movie-buff friend. Speak casually with natural contractions and short openers when it feels right (Yeah—, Oh totally—). Keep it brief (2–4 sentences), direct, and warm. Use PLAIN TEXT only (no Markdown). If a <style> ToneDirective block is present in the user message, follow it in the final prose.
        """
    }

    private func resolveFewShotMessages(_ fewShotMessagesJson: String?) -> [ChatMessage] {
        let trimmed = fewShotMessagesJson?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if trimmed.isEmpty {
            trace("askText_fewshot_source source=swift_fallback reason=empty")
            return defaultFewShotMessages()
        }

        if trimmed.count > maxUnityFewShotPayloadChars {
            trace("askText_fewshot_source source=swift_fallback reason=too_large chars=\(trimmed.count)")
            return defaultFewShotMessages()
        }

        guard let data = trimmed.data(using: .utf8) else {
            trace("askText_fewshot_source source=swift_fallback reason=encoding_failed")
            return defaultFewShotMessages()
        }

        guard let decoded = try? JSONDecoder().decode(UnityChatMessagesEnvelope.self, from: data) else {
            trace("askText_fewshot_source source=swift_fallback reason=decode_failed")
            return defaultFewShotMessages()
        }

        if decoded.messages.isEmpty {
            trace("askText_fewshot_source source=swift_fallback reason=no_messages")
            return defaultFewShotMessages()
        }

        let sanitized = decoded.messages
            .compactMap { message -> ChatMessage? in
                let role = message.role.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                let content = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !content.isEmpty else { return nil }
                guard role == "system" || role == "user" || role == "assistant" else { return nil }
                return ChatMessage(role: role, content: content)
            }
            .prefix(maxUnityFewShotMessages)

        let sanitizedMessages = Array(sanitized)
        if sanitizedMessages.isEmpty {
            trace("askText_fewshot_source source=swift_fallback reason=no_valid_messages")
            return defaultFewShotMessages()
        }

        let droppedCount = max(0, decoded.messages.count - sanitizedMessages.count)
        trace("askText_fewshot_source source=unity count=\(sanitizedMessages.count) dropped=\(droppedCount)")
        return sanitizedMessages
    }

    private func defaultFewShotMessages() -> [ChatMessage] {
        [
            .init(role: "user", content: "Best stage-to-film version of Macbeth?"),
            .init(role: "assistant", content: "I'd go with the 2015 one with Michael Fassbender-moody, brutal, and gorgeous. If you want something bolder, the Coen version with Denzel Washington is stark and theatrical. Your call: classic grit or minimalist chill."),
            .init(role: "user", content: "Feel-good musical tonight?"),
            .init(role: "assistant", content: "Go La La Land if you want dreamy jazz and bittersweet smiles. If you want pure sparkle, The Greatest Showman is popcorn joy. Tell me your vibe and I'll dial it in.")
        ]
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
        let modelForChat = resolvedTextModel()
        let res = try await sendTextChatWithModelFallback(messages: msgs,
                                                          apiKey: nvidiaKey,
                                                          preferredModel: modelForChat,
                                                          fallbackModel: fallbackTextModel,
                                                          maxTokens: 220,
                                                          temperature: 0.7,
                                                          topP: 0.95,
                                                          responseFormat: nil)
        return (res.choices.first?.message?.content ?? text).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // Qwen vision (expects base64 JPEG/PNG string from Unity)
    func askVision(base64Image: String, prompt: String) {
        cancelActiveResponseTask()
        let requestID = UUID()
        activeResponseRequestID = requestID
        activeResponseTask = Task { [weak self] in
            guard let self else { return }
            defer {
                if self.activeResponseRequestID == requestID {
                    self.activeResponseTask = nil
                }
            }

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
                try Task.checkCancellation()
                let text = try await hf.chat(messages: msgs,
                                             hfToken: hfToken,
                                             model: hfModel,
                                             maxTokens: 512,
                                             temperature: 0.5,
                                             responseFormat: nil)
                try Task.checkCancellation()
                guard self.isCurrentResponseRequest(requestID) else { return }
                let answer = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if answer.isEmpty { self.sendResponseMessageToUnity("answer:Sorry — I couldn’t generate an answer.", requestID: requestID) }
                else { self.sendResponseMessageToUnity("answer:" + answer, requestID: requestID) }
            } catch is CancellationError {
                print("askVision cancelled")
            } catch {
                guard self.isCurrentResponseRequest(requestID) else { return }
                self.sendResponseMessageToUnity("error:" + error.localizedDescription, requestID: requestID)
            }
        }
    }

    private func cancelActiveResponseTask() {
        activeResponseRequestID = UUID()
        activeResponseTask?.cancel()
        activeResponseTask = nil
    }

    private func abandonActiveResponseTask() {
        activeResponseRequestID = UUID()
        activeResponseTask = nil
    }

    private func waitForNativeTTSStopCompletion() {
        guard isAwaitingNativeTTSStop else { return }

        if speech.synth.isSpeaking || speech.synth.isPaused {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) { [weak self] in
                self?.waitForNativeTTSStopCompletion()
            }
            return
        }

        finalizeNativeTTSStopIfNeeded()
    }

    private func finalizeNativeTTSStopIfNeeded() {
        guard isAwaitingNativeTTSStop else { return }

        isAwaitingNativeTTSStop = false
        shouldEmitUserFacingTTSStarted = false
        refreshSynth(reason: "native_stop_completed")
        trace("tts_stop_completed")
        sendToUnity("tts_finished")
    }

    private func isCurrentResponseRequest(_ requestID: UUID) -> Bool {
        activeResponseRequestID == requestID
    }

    private func sendResponseMessageToUnity(_ message: String, requestID: UUID) {
        guard isCurrentResponseRequest(requestID) else { return }
        guard let cb = unityBytesCallback else { return }
        if message.hasPrefix("answer:") || message.hasPrefix("error:") {
            trace("send_to_unity message=\(message.prefix(24)) request=\(requestID.uuidString.prefix(8))")
        }

        let utf8 = Array(message.utf8)
        let len = utf8.count
        let ptr = UnsafeMutablePointer<UInt8>.allocate(capacity: len)
        ptr.initialize(from: utf8, count: len)

        DispatchQueue.main.async { [weak self] in
            guard let self else {
                ptr.deallocate()
                return
            }
            guard self.isCurrentResponseRequest(requestID) else {
                ptr.deallocate()
                return
            }

            cb(ptr, Int32(len))
            ptr.deallocate()
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
    MovioUnityBridge.shared.sendAvailableVoicesToUnity()
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

@_cdecl("movioAskTextWithModel")
public func movioAskTextWithModel(_ prompt: UnsafePointer<CChar>?,
                                  _ preferredModel: UnsafePointer<CChar>?,
                                  _ fallbackModel: UnsafePointer<CChar>?) {
    let p = prompt.flatMap { String(cString: $0) } ?? ""
    let preferred = preferredModel.flatMap { String(cString: $0) } ?? ""
    let fallback = fallbackModel.flatMap { String(cString: $0) } ?? ""
    MovioUnityBridge.shared.askText(p, preferredModel: preferred, fallbackModel: fallback)
}

@_cdecl("movioAskTextWithContext")
public func movioAskTextWithContext(_ prompt: UnsafePointer<CChar>?,
                                    _ preferredModel: UnsafePointer<CChar>?,
                                    _ fallbackModel: UnsafePointer<CChar>?,
                                    _ systemPrompt: UnsafePointer<CChar>?,
                                    _ fewShotMessagesJson: UnsafePointer<CChar>?) {
    let p = prompt.flatMap { String(cString: $0) } ?? ""
    let preferred = preferredModel.flatMap { String(cString: $0) } ?? ""
    let fallback = fallbackModel.flatMap { String(cString: $0) } ?? ""
    let system = systemPrompt.flatMap { String(cString: $0) } ?? ""
    let fewShot = fewShotMessagesJson.flatMap { String(cString: $0) } ?? ""
    MovioUnityBridge.shared.askText(
        p,
        preferredModel: preferred,
        fallbackModel: fallback,
        systemPrompt: system,
        fewShotMessagesJson: fewShot)
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
    
    @State private var apiKey: String = ""
    @State private var tavilyKey: String = ""
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
    @FocusState private var promptFocused: Bool
    @FocusState private var urlFocused: Bool

    // Hugging Face VLM (Qwen2.5-VL) token and screenshot selection
    @State private var hfToken: String = ""
    // Provider-qualified HF model for Qwen VL (e.g., ":nebius" suffix)
    @State private var hfModel: String = "Qwen/Qwen2.5-VL-72B-Instruct:nebius"
    private let modelScopeToken: String = ""
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
            HStack(spacing: 8) {
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
            print("NOTE: Requested GEMMA3-14B not found; using GEMMA3-12B via featherless-ai; Qwen-32B via fireworks-ai.")
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
                let r = (msg.content ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                if !r.isEmpty { out = r }
            }
        }
        // Pass 2: always enforce the selected mood tone (even Calm)
        if let rewriteTone = try? await client.send(messages: [
            .init(role: "system", content: "Rewrite to match this tone guide exactly. Keep meaning and facts. Plain text only. 2–4 short sentences. No Sources unless asked."),
            .init(role: "user", content: styleBlock(for: mood) + "\n\n" + out)
        ], apiKey: apiKey), let msg2 = rewriteTone.choices.first?.message {
            let toned = (msg2.content ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
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
        let voice = AVSpeechSynthesisVoice(language: "en-GB")
                 ?? AVSpeechSynthesisVoice(language: AVSpeechSynthesisVoice.currentLanguageCode())
        let (rate, pitch) = ttsParams(for: selectedMood)

        // 3) Interrupt anything currently speaking
        speech.synth.stopSpeaking(at: .immediate)

        // 4) Speak sentence by sentence with a tiny pause
        let sentences = splitIntoSentences(speakable)
        for (idx, sentence) in sentences.enumerated() {
            let u = AVSpeechUtterance(string: sentence)
            u.voice = voice
            u.rate = rate
            u.pitchMultiplier = pitch
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
        
