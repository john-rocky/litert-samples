This file gives the commands and code for moving an Android app from the
MediaPipe LLM Inference API (`com.google.mediapipe:tasks-genai`, with the
`LlmInference` and `LlmInferenceSession` classes) to LiteRT-LM. tasks-genai
0.10.35 marks both classes as deprecated: "Migrate to LiteRT LM instead." The
migrated app ends with the code of Steps 1 and 5 of the skill: the Gradle and
manifest entries of Step 1 and the `ChatViewModel` of Step 5, which takes over
from the app's `LlmInference` code.

The code in this file uses these imports:

    import android.content.Context
    import com.google.ai.edge.litertlm.Backend
    import com.google.ai.edge.litertlm.Contents
    import com.google.ai.edge.litertlm.Conversation
    import com.google.ai.edge.litertlm.ConversationConfig
    import com.google.ai.edge.litertlm.Engine
    import com.google.ai.edge.litertlm.EngineConfig
    import com.google.ai.edge.litertlm.Message
    import com.google.ai.edge.litertlm.SamplerConfig

## 1. Find the MediaPipe code

Read `app/build.gradle.kts` (or `app/build.gradle`) and
`gradle/libs.versions.toml`, where the dependency can be a version catalog
entry with the group `com.google.mediapipe` and the name `tasks-genai`. Then
find the code that the later steps change:

    grep -rnE 'com\.google\.mediapipe\.tasks\.genai|LlmInference|ProgressListener|PromptTemplates|\.task\b' --include='*.kt' --include='*.java' app/src

Note these points before you edit any file:

- The `.task` file that the app loads, and how it reaches the device: a download URL, `adb push`, or an AI pack.
- The options: `setPreferredBackend`, `setMaxTokens`, and the sampling options `setTopK`, `setTopP`, `setTemperature`, and `setRandomSeed`.
- Which class the app generates with: `generateResponse` and `generateResponseAsync` of `LlmInference` answer each prompt on its own, without the earlier prompts and replies, and an `LlmInferenceSession` keeps them.
- Code that wraps the prompt in the model's chat template, such as `<start_of_turn>user` or `<|im_start|>user`, and `setPromptTemplates`.
- Image or audio input, LoRA weights, and constrained decoding, which this skill doesn't migrate (see Out of scope).

## 2. Replace the model file

The LiteRT-LM documentation describes only `.litertlm` files: move the app to
the `.litertlm` file of the same model. LiteRT-LM 0.18.0 and 0.19.1 also
loaded `Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.task` of
`litert-community/Qwen2.5-1.5B-Instruct` and generated a reply with it on the
CPU of a Pixel 11 Pro and, with 0.18.0, on the GPU of a Galaxy S26, so for that
file the app can move to the LiteRT-LM API first and replace the file later.

- **A model from litert-community** : Look for a `.litertlm` file of the same model in litert-community, as Step 2 of the skill describes. Some repositories hold the same model variant as a `.task` file and as a `.litertlm` file; for example, `litert-community/Qwen2.5-1.5B-Instruct` holds `Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.task` and `Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm`. Some hold no `.litertlm` file at all. Read the file list of the repository. When it lists a `.litertlm` file, change the file name in the download URL, in the `adb push` command, and in the code to that name. When it lists none, keep the `.task` file and its name, and tell the user that the repository lists no `.litertlm` file and that LiteRT-LM might not load the `.task` file or generate a reply with it. When you can't read the list, keep the `.task` file and its name too, and tell the user that you couldn't read the list. Write no file name that the list doesn't show, such as the `.task` name with `.litertlm` in place of `.task`.
- **A `.task` file that the user made** : This skill doesn't convert models: tell the user that the app needs a `.litertlm` file, as [get a model](https://developer.android.com/agents/skills/device-ai/litert/litert-lm/references/get-a-model) describes. In one test on a Pixel 11 Pro, LiteRT-LM 0.18.0 and 0.19.1 generated a reply on the CPU with a `.litertlm` file that `litert-lm-builder` 0.19.1 built from the TFLite model and the tokenizer inside the Qwen2.5-1.5B `.task` file, with LLM metadata written by hand.

## 3. Swap the dependencies

Remove `com.google.mediapipe:tasks-genai` and add
`com.google.ai.edge.litertlm:litertlm-android:0.18.0` as Step 1 of the skill
shows. The swap changes these:

- **`minSdk`** : The tasks-genai AAR declares 21 and `litertlm-android` declares 24. Raise `minSdk` to 24 or higher.
- **Kotlin** : `litertlm-android` 0.18.0 is compiled with Kotlin 2.4 (metadata version 2.4.0). With the Kotlin plugin 2.2.21 or older, the build stops with `The binary version of its metadata is 2.4.0, expected version is ...`; Kotlin 2.3.0 builds it. When the Kotlin plugin is older than 2.3.0, raise it to 2.3.0 as part of the migration, also when the version is set in `gradle/libs.versions.toml`, and name the old and the new version in your report to the user. On Kotlin 2.3, `jvmTarget` is set in `compilerOptions` in place of `kotlinOptions`, and a Compose app applies the `org.jetbrains.kotlin.plugin.compose` plugin in place of `composeOptions`.
- **Native libraries** : tasks-genai has them for `arm64-v8a`, `armeabi-v7a`, `x86`, and `x86_64`, and `litertlm-android` only for `arm64-v8a` and `x86_64`. Tell the user that the migrated app can't run the model on a device that runs only 32-bit ARM code.
- **GPU manifest entries** : The tasks-genai AAR declares `libOpenCL.so`, `libOpenCL-car.so`, and `libOpenCL-pixel.so` as `uses-native-library` in its own manifest, and the build merges them into the app. `litertlm-android` declares none. For the GPU backend, add the two entries of Step 1 of the skill. Without them, the GPU backend failed on both phones tried: on a Pixel 11 Pro, `initialize` failed with `Failed to create engine` for Qwen3-0.6B and Qwen2.5-1.5B; on a Galaxy S26, `initialize` succeeded and the first reply ended in the error of Step 4 of the skill, for the same two models.
- **Both libraries during the move** : To move one feature at a time, keep the tasks-genai dependency until the last feature that uses it has moved; tasks-genai 0.10.35 and `litertlm-android` 0.18.0 in one APK both answered on the CPU, one after the other, in one app process on a Pixel 11 Pro.

## 4. Map the API

| tasks-genai 0.10.35 | LiteRT-LM 0.18.0 |
|---|---|
| `LlmInference.createFromOptions(context, options)` | `Engine(config)`, then `initialize` on the executor of Step 5 of the skill. `initialize` throws `LiteRtLmJniException` when the file is missing or isn't a model. |
| `setModelPath`, `setMaxTokens`, `setPreferredBackend`, `setMaxTopK` | `EngineConfig`, as `engineConfigFromOptions` builds it; `setMaxTopK` has no counterpart (`topK` is part of `SamplerConfig`) |
| `LlmInference.Backend.DEFAULT`, `CPU`, `GPU` | `Backend.CPU()` or `Backend.GPU()`; LiteRT-LM has no `DEFAULT`, and `EngineConfig` uses the CPU when no backend is passed |
| `LlmInferenceSession.createFromOptions(llmInference, options)` | `engine.createConversation(ConversationConfig(...))` |
| `setTopK`, `setTopP`, `setTemperature`, `setRandomSeed` | `SamplerConfig` in `ConversationConfig`, as `samplerFromSessionOptions` builds it |
| `setPromptTemplates` | Remove it: the conversation applies the chat template of the `.litertlm` file. `ConversationConfig` takes a system text as `systemInstruction`, and another Jinja template as `chatTemplate`. |
| `addQueryChunk(text)`, then `generateResponseAsync(progressListener)` | `conversation.sendMessageAsync(text)`, a `Flow<Message>` of chunks; the `ListenableFuture` that `generateResponseAsync` returned has no counterpart: collect the flow |
| `addQueryChunk(text)`, then `generateResponse()` | `conversation.sendMessage(text).toString()` |
| `LlmInference.generateResponse(text)` | `generateOnce`, a new conversation for each call |
| `cancelGenerateResponseAsync()` | `conversation.cancelProcess()`; the flow ends with `CancellationException`. That conversation takes no more messages: a later `sendMessage` throws `LiteRtLmJniException` (`Failed to call nativeSendMessage: CANCELLED: Task cancelled`), and the flow of a later `sendMessageAsync` ends with `CancellationException` before any chunk. |
| `sizeInTokens(text)` | No counterpart; `conversation.getTokenCount()` counts the tokens that the conversation holds |
| `cloneSession()`, `updateSessionOptions(...)` | No counterpart in 0.18.0; `conversationWithHistory` starts a new conversation from the earlier turns. 0.19.0 and 0.19.1 also have `Conversation.clone()`. |
| `close()` of the session and of `LlmInference` | `close()` of the conversation, then of the engine, each once; a second `close` throws `IllegalStateException` |

## 5. Rewrite the inference code

Never search and replace `LlmInference`. Let the `ChatViewModel` of Step 5 of
the skill own the engine and the conversation, and keep the app's own classes
and their public functions as its callers:

- **Loading** : `createFromOptions` on `viewModelScope` or `Dispatchers.IO` becomes `load` of Step 5 of the skill, which initializes on the ViewModel's own executor.
- **Streaming** : The `ProgressListener` that appended each partial result becomes the `collect` in `send` of Step 5 of the skill. The conversation keeps the earlier turns, as an `LlmInferenceSession` did. An app that called `generateResponse` on `LlmInference` answered without history: keep that with `generateOnce`.
- **Chat template** : The conversation wraps every message in the chat template of the `.litertlm` file. Remove the code that adds the template to the prompt; otherwise the model gets it twice.
- **Options** : Copy the values of the old options with `engineConfigFromOptions` and `samplerFromSessionOptions`. An app that set no sampling options ran with the defaults of tasks-genai, which `samplerFromSessionOptions` keeps. Without a `SamplerConfig`, the conversation uses the default values of the engine.
- **Stopping a reply** : After `cancelProcess` stops a reply, that conversation takes no more messages: a later `sendMessage` throws `LiteRtLmJniException`, and the flow of a later `sendMessageAsync` ends with `CancellationException` before any chunk. If the app kept using the same session after `cancelGenerateResponseAsync`, close the stopped conversation and go on in a new one that `conversationWithHistory` starts from the earlier turns (the app keeps them as pairs of user text and model reply).
- **Closing** : `close()` of an `LlmInferenceSession` throws `IllegalStateException` (`Previous invocation still processing. Wait for done=true.`) while a reply streams, so a close in `onCleared` crashes the app when the user leaves the screen during a reply. When the screen closes while `createFromOptions` still runs on `viewModelScope`, the engine that the call returns is never closed. In the ViewModel of Step 5 of the skill, `onCleared` waits for a running `load`, and `release` calls `cancelProcess` until the flow ends, then closes the conversation and the engine, each once.

The following functions carry the old options and calls over:

    // Replaces LlmInferenceOptions with setModelPath, setMaxTokens, and
    // setPreferredBackend, for the EngineConfig in load of Step 5 of the skill.
    // maxNumTokens counts the input and the output tokens, as maxTokens did.
    // When maxNumTokens is null, the default of the model or the engine
    // applies; an app that never called setMaxTokens ran with 512.
    fun engineConfigFromOptions(
        context: Context,
        modelPath: String,
        maxTokens: Int,
        backend: Backend
    ): EngineConfig =
        EngineConfig(
            modelPath = modelPath,
            backend = backend,
            maxNumTokens = maxTokens,
            cacheDir = context.cacheDir.path
        )

    // Replaces setTopK, setTopP, setTemperature, and setRandomSeed of
    // LlmInferenceSessionOptions, with the same defaults. Pass the result as
    // samplerConfig in the ConversationConfig of Step 5 of the skill.
    fun samplerFromSessionOptions(
        topK: Int = 40,
        topP: Float = 1.0f,
        temperature: Float = 0.8f,
        randomSeed: Int = 0
    ): SamplerConfig =
        SamplerConfig(
            topK = topK,
            topP = topP.toDouble(),
            temperature = temperature.toDouble(),
            seed = randomSeed
        )

    // Replaces LlmInference.generateResponse(text): no earlier prompts, and the
    // default session options of tasks-genai unless sampler is set. Call it on
    // the executor that owns the engine.
    fun generateOnce(
        engine: Engine,
        text: String,
        sampler: SamplerConfig = samplerFromSessionOptions()
    ): String =
        engine.createConversation(ConversationConfig(samplerConfig = sampler))
            .use { it.sendMessage(text).toString() }

    // Replaces cloneSession() and updateSessionOptions(), and goes on with a
    // chat after cancelProcess: a new conversation from the earlier turns (user
    // text, model reply). Pass the sampler and the systemInstruction of the
    // first conversation to keep them, or another sampler for other values.
    fun conversationWithHistory(
        engine: Engine,
        turns: List<Pair<String, String>>,
        sampler: SamplerConfig? = null,
        systemInstruction: Contents? = null
    ): Conversation =
        engine.createConversation(
            ConversationConfig(
                systemInstruction = systemInstruction,
                initialMessages = turns.flatMap { (user, model) ->
                    listOf(Message.user(user), Message.model(model))
                },
                samplerConfig = sampler
            )
        )

## 6. Verify in two stages

**Stage 1, on the workstation** : `./gradlew assembleDebug` passes, and the
following command prints nothing:

    grep -rnE 'com\.google\.mediapipe\.tasks\.genai|tasks-genai|LlmInference' --include='*.kt' --include='*.java' --include='*.kts' --include='*.gradle' --include='*.toml' --include='*.pro' --exclude-dir=build --exclude-dir=.gradle .

After a `.task` file is replaced by a `.litertlm` file, a search for the name of that `.task` file prints nothing too. The name of a `.task` file that the app keeps stays in the project.

**Stage 2, on a device** : Pick one path:

- **Path A (device connected)** : Put the model file on the device (Step 3 of the skill) and run the migrated app. Ask the question that the app answered before the migration: the reply streams. Ask a second question that depends on the first: the reply shows that the model still has the first turn, unless the app answered without history before. Then leave the screen while a reply streams: the app doesn't crash.
- **Path B (no device)** : Run Stage 1 only, and report that Stage 2 didn't run.

## Out of scope

This skill doesn't migrate image or audio input (`addImage`, `addAudio`), LoRA
weights (`setLoraPath`), constrained decoding (`setConstraintHandle`), or the
LLM Inference API on iOS and the web: when the app uses one of them, tell the
user before you edit any file. The other MediaPipe tasks (`tasks-vision`,
`tasks-text`, and `tasks-audio`) are outside this file.
