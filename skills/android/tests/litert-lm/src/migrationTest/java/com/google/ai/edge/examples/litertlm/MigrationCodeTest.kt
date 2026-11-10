/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.examples.litertlm

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.LiteRtLmJniException
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig
import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Runs the Kotlin of the skill's `references/migration.md`, which the build extracts into this test
 * APK, with a real model, and checks what the reference says about LiteRT-LM: the options that
 * `engineConfigFromOptions` and `samplerFromSessionOptions` carry over, `generateOnce` answering
 * without the earlier prompts while a conversation keeps them, `conversationWithHistory` starting
 * from earlier turns, a chat template added by hand reaching the model twice, `cancelProcess`
 * ending the flow with `CancellationException`, a later message on that conversation failing, and
 * a new conversation going on from the earlier turns, `getTokenCount`, a second `close`,
 * `initialize` failing on a missing file and on a file that is not a model, and a MediaPipe `.task`
 * file loading and answering. The tests share one engine; a test that makes an engine of its own
 * closes the shared one first, so that one engine is open at a time.
 *
 * Push a model first: `adb push Qwen3-0.6B.litertlm /data/local/tmp/`. Another file or the GPU:
 * `-e modelPath <path>` and `-e backend GPU`. The `.task` test runs only with `-e taskPath <path>`
 * naming a `.task` file on the device.
 */
@RunWith(AndroidJUnit4::class)
class MigrationCodeTest {
    @Test
    fun engineConfigFromOptions_carriesTheOptions() {
        val config = engineConfigFromOptions(context, modelPath, MAX_TOKENS, backend())
        assertEquals(modelPath, config.modelPath)
        assertEquals(MAX_TOKENS, config.maxNumTokens)
        assertEquals(context.cacheDir.path, config.cacheDir)
        assertEquals(backend().name, config.backend.name)
        // The shared engine comes from this function.
        assertEquals(MAX_TOKENS, engine.engineConfig.maxNumTokens)
        val reply = onEngineThread { generateOnce(engine, SHORT_PROMPT) }
        Log.i(TAG, "engineConfigFromOptions on ${backend().name}: '${reply.trim()}'")
        assertTrue("empty reply", reply.isNotBlank())
    }

    @Test
    fun samplerFromSessionOptions_keepsTheTasksGenaiDefaults() {
        val defaults = samplerFromSessionOptions()
        assertEquals(40, defaults.topK)
        assertEquals(1.0, defaults.topP, 0.0)
        assertEquals(0.8f.toDouble(), defaults.temperature, 0.0)
        assertEquals(0, defaults.seed)
        val sampler =
            samplerFromSessionOptions(topK = 10, topP = 0.95f, temperature = 0.7f, randomSeed = 7)
        assertEquals(10, sampler.topK)
        assertEquals(0.95f.toDouble(), sampler.topP, 0.0)
        assertEquals(0.7f.toDouble(), sampler.temperature, 0.0)
        assertEquals(7, sampler.seed)
        val reply =
            onEngineThread {
                engine.createConversation(ConversationConfig(samplerConfig = sampler)).use {
                    it.sendMessage(SHORT_PROMPT).toString()
                }
            }
        assertTrue("empty reply", reply.isNotBlank())
    }

    @Test
    fun generateOnce_answersWithoutTheEarlierPrompt_whileAConversationKeepsIt() {
        val first = onEngineThread { generateOnce(engine, NAME_PROMPT) }
        val once = onEngineThread { generateOnce(engine, ASK_PROMPT) }
        Log.i(TAG, "generateOnce twice: '${first.trim()}', then '${once.trim()}'")
        assertFalse("generateOnce kept the earlier prompt: $once", once.contains(NAME))
        val kept =
            onEngineThread {
                engine.createConversation(ConversationConfig(samplerConfig = GREEDY)).use {
                    it.sendMessage(NAME_PROMPT)
                    it.sendMessage(ASK_PROMPT).toString()
                }
            }
        Log.i(TAG, "one conversation: '${kept.trim()}'")
        assertTrue("the conversation lost the earlier turn: $kept", kept.contains(NAME))
    }

    @Test
    fun conversationWithHistory_startsFromTheTurns() {
        val withTurns =
            onEngineThread {
                conversationWithHistory(engine, listOf(NAME_PROMPT to "OK."), GREEDY).use {
                    it.sendMessage(ASK_PROMPT).toString()
                }
            }
        val withoutTurns =
            onEngineThread {
                conversationWithHistory(engine, emptyList(), GREEDY).use {
                    it.sendMessage(ASK_PROMPT).toString()
                }
            }
        // With the system instruction that load of Step 5 of the skill sets.
        val withInstruction =
            onEngineThread {
                conversationWithHistory(
                    engine,
                    listOf(NAME_PROMPT to "OK."),
                    GREEDY,
                    Contents.of(SYSTEM_INSTRUCTION)
                ).use { it.sendMessage(ASK_PROMPT).toString() }
            }
        Log.i(TAG, "with the turns: '${withTurns.trim()}', without: '${withoutTurns.trim()}'")
        Log.i(TAG, "with the turns and a system instruction: '${withInstruction.trim()}'")
        assertTrue("the turns did not reach the model: $withTurns", withTurns.contains(NAME))
        assertFalse("the name came without the turns: $withoutTurns", withoutTurns.contains(NAME))
        assertTrue(
            "the turns did not reach the model with a system instruction: $withInstruction",
            withInstruction.contains(NAME)
        )
    }

    @Test
    fun conversationWithHistory_whileTheFirstConversationIsOpen() {
        // A replacement for cloneSession() keeps the first conversation open next to the new one.
        val first =
            onEngineThread { engine.createConversation(ConversationConfig(samplerConfig = GREEDY)) }
        try {
            val firstReply = onEngineThread { first.sendMessage(NAME_PROMPT).toString() }
            val copy =
                onEngineThread {
                    conversationWithHistory(engine, listOf(NAME_PROMPT to firstReply), GREEDY)
                }
            try {
                val fromCopy = onEngineThread { copy.sendMessage(ASK_PROMPT).toString() }
                val fromFirst = onEngineThread { first.sendMessage(ASK_PROMPT).toString() }
                Log.i(TAG, "two open conversations: '${fromCopy.trim()}', '${fromFirst.trim()}'")
                assertTrue("the new conversation: $fromCopy", fromCopy.contains(NAME))
                assertTrue("the first conversation: $fromFirst", fromFirst.contains(NAME))
            } finally {
                onEngineThread { copy.close() }
            }
        } finally {
            onEngineThread { first.close() }
        }
    }

    @OptIn(ExperimentalApi::class)
    @Test
    fun chatTemplateAddedByHand_reachesTheModelTwice() {
        val (plain, byHand) =
            onEngineThread {
                engine.createConversation().use { conversation ->
                    val plain = conversation.renderMessageIntoString(Message.user(SHORT_PROMPT))
                    // The app's old code sent the prompt with the template already around it.
                    plain to conversation.renderMessageIntoString(Message.user(plain))
                }
            }
        val userPrefix = plain.substringBefore(SHORT_PROMPT)
        Log.i(TAG, "rendered: '${escape(plain)}', by hand: '${escape(byHand)}'")
        assertTrue(
            "the template adds nothing before the text: ${escape(plain)}",
            userPrefix.isNotBlank()
        )
        assertEquals(1, plain.split(userPrefix).size - 1)
        assertEquals(2, byHand.split(userPrefix).size - 1)
    }

    @Test
    fun cancelProcess_endsTheConversation_andConversationWithHistoryGoesOn() {
        val conversation =
            onEngineThread { engine.createConversation(ConversationConfig(samplerConfig = GREEDY)) }
        var open = true
        try {
            val firstReply = onEngineThread { conversation.sendMessage(NAME_PROMPT).toString() }
            var chunks = 0
            var completed = false
            var ending: Throwable? = null
            var seconds = 0.0
            runBlocking(executor.asCoroutineDispatcher()) {
                val firstChunk = CompletableDeferred<Unit>()
                val collector = launch {
                    try {
                        conversation.sendMessageAsync(LONG_PROMPT).collect {
                            chunks++
                            firstChunk.complete(Unit)
                        }
                        completed = true
                    } catch (e: Throwable) {
                        ending = e
                    }
                }
                withTimeout(TIMEOUT_SECONDS * 1000) { firstChunk.await() }
                val start = System.currentTimeMillis()
                conversation.cancelProcess()
                withTimeout(STOP_LIMIT_SECONDS * 1000) { collector.join() }
                seconds = (System.currentTimeMillis() - start) / 1000.0
            }
            Log.i(TAG, "cancelProcess after $chunks chunks: ended after $seconds s with $ending")
            assertFalse("the reply ran to its end", completed)
            assertTrue("the flow ended with $ending", ending is CancellationException)
            val stopped =
                assertThrows(LiteRtLmJniException::class.java) {
                    onEngineThread { conversation.sendMessage(SHORT_PROMPT) }
                }
            Log.i(TAG, "a message on the stopped conversation: ${stopped.message}")
            // A later sendMessageAsync on the stopped conversation, the call of `send` in Step 5.
            var laterChunks = 0
            var laterEnding: Throwable? = null
            runBlocking(executor.asCoroutineDispatcher()) {
                val later = launch {
                    try {
                        conversation.sendMessageAsync(SHORT_PROMPT).collect { laterChunks++ }
                    } catch (e: Throwable) {
                        laterEnding = e
                    }
                }
                withTimeout(STOP_LIMIT_SECONDS * 1000) { later.join() }
            }
            Log.i(
                TAG,
                "sendMessageAsync on the stopped conversation: " +
                    "$laterChunks chunks, ended with $laterEnding"
            )
            assertEquals("chunks from the stopped conversation", 0, laterChunks)
            assertTrue(
                "the later flow ended with $laterEnding",
                laterEnding is CancellationException
            )
            // A new conversation answers, also while the stopped one is open.
            val fresh = onEngineThread { generateOnce(engine, SHORT_PROMPT) }
            assertTrue("empty reply from a new conversation", fresh.isNotBlank())
            // The chat goes on in a new conversation, from the turn before the stopped reply.
            onEngineThread { conversation.close() }
            open = false
            val next =
                onEngineThread {
                    conversationWithHistory(engine, listOf(NAME_PROMPT to firstReply), GREEDY).use {
                        it.sendMessage(ASK_PROMPT).toString()
                    }
                }
            Log.i(TAG, "new conversation after cancelProcess: '${next.trim()}'")
            assertTrue("the new conversation lost the earlier turn: $next", next.contains(NAME))
        } finally {
            if (open) {
                onEngineThread { conversation.close() }
            }
        }
    }

    @Test
    fun getTokenCount_countsTheTokensTheConversationHolds() {
        val counts =
            onEngineThread {
                engine.createConversation().use { conversation ->
                    val counts = mutableListOf(conversation.getTokenCount())
                    conversation.sendMessage(SHORT_PROMPT)
                    counts += conversation.getTokenCount()
                    conversation.sendMessage(SHORT_PROMPT)
                    counts += conversation.getTokenCount()
                    counts
                }
            }
        Log.i(TAG, "getTokenCount: $counts")
        assertTrue(
            "the count did not grow: $counts",
            counts[1] > counts[0] && counts[2] > counts[1]
        )
    }

    @Test
    fun secondClose_throwsIllegalStateException() {
        val conversation = onEngineThread { engine.createConversation() }
        onEngineThread { conversation.close() }
        val conversationError =
            assertThrows(IllegalStateException::class.java) {
                onEngineThread { conversation.close() }
            }
        assertEquals("Conversation is closed already.", conversationError.message)
        // Another engine of the same model, on the CPU, closed twice.
        withTheSharedEngineClosed {
            val second = onEngineThread {
                Engine(EngineConfig(modelPath = modelPath, cacheDir = context.cacheDir.path)).also {
                    it.initialize()
                }
            }
            onEngineThread { second.close() }
            val engineError =
                assertThrows(IllegalStateException::class.java) {
                    onEngineThread { second.close() }
                }
            assertEquals("Engine is not initialized.", engineError.message)
        }
    }

    @Test
    fun initialize_onAMissingFileOrAFileThatIsNotAModel_throwsLiteRtLmJniException() {
        val notAModel = File(context.cacheDir, "not-a-model.litertlm")
        notAModel.writeBytes(ByteArray(4096))
        withTheSharedEngineClosed {
            for (path in listOf(MISSING_MODEL, notAModel.path)) {
                val failed =
                    Engine(
                        EngineConfig(
                            modelPath = path,
                            backend = backend(),
                            cacheDir = context.cacheDir.path
                        )
                    )
                val error =
                    assertThrows(LiteRtLmJniException::class.java) {
                        onEngineThread { failed.initialize() }
                    }
                Log.i(TAG, "initialize($path): ${error.message}")
                assertFalse(failed.isInitialized())
                if (path == MISSING_MODEL) {
                    assertEquals("Model file not found: $MISSING_MODEL", error.message)
                }
            }
        }
    }

    @Test
    fun initialize_onAMediaPipeTaskFile_loadsAndAnswers() {
        val argument = arguments.getString("taskPath")
        assumeTrue("no -e taskPath <path of a .task file on the device>", argument != null)
        val taskPath = checkNotNull(argument)
        assumeTrue("no file at $taskPath", File(taskPath).canRead())
        withTheSharedEngineClosed {
            val task = onEngineThread {
                Engine(
                    EngineConfig(
                        modelPath = taskPath,
                        backend = backend(),
                        cacheDir = context.cacheDir.path
                    )
                ).also { it.initialize() }
            }
            try {
                val reply = onEngineThread { generateOnce(task, TASK_PROMPT) }
                Log.i(TAG, "${File(taskPath).name} on ${backend().name}: '${reply.trim()}'")
                assertTrue("empty reply from the .task file", reply.isNotBlank())
            } finally {
                onEngineThread { task.close() }
            }
        }
    }

    private fun escape(text: String) = text.replace("\n", "\\n")

    companion object {
        private const val TAG = "MigrationCodeTest"
        private const val DEFAULT_MODEL = "/data/local/tmp/Qwen3-0.6B.litertlm"
        private const val MISSING_MODEL = "/data/local/tmp/no-such-model.litertlm"
        private const val MAX_TOKENS = 1024
        private const val NAME = "Ada"

        /** `/no_think` is a Qwen3 switch; other models read it as text, which is fine here. */
        private const val SHORT_PROMPT = "Reply with the single word: ready. /no_think"
        private const val NAME_PROMPT =
            "My name is $NAME. Reply with the single word: OK. /no_think"
        private const val ASK_PROMPT = "What is my name? Reply with the name only. /no_think"
        private const val LONG_PROMPT =
            "Write a story of at least 500 words about a lighthouse keeper."
        /** The system instruction of Step 5 of the skill. */
        private const val SYSTEM_INSTRUCTION = "You are a helpful assistant."
        /** A question with a short, well-known answer. */
        private const val TASK_PROMPT = "What are the three primary colors? Answer in one sentence."
        private const val TIMEOUT_SECONDS = 900L
        private const val STOP_LIMIT_SECONDS = 20L
        private val GREEDY = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0)

        private val arguments = InstrumentationRegistry.getArguments()
        private val modelPath = arguments.getString("modelPath") ?: DEFAULT_MODEL
        private val context = InstrumentationRegistry.getInstrumentation().targetContext

        /** The one thread that creates, uses, and closes every engine and conversation here. */
        private val executor = Executors.newSingleThreadExecutor()
        private lateinit var engine: Engine

        private fun backend(): Backend =
            if (arguments.getString("backend") == "GPU") {
                Backend.GPU()
            } else {
                Backend.CPU()
            }

        /** Runs [block] on [executor] and rethrows what it throws. */
        private fun <T> onEngineThread(block: () -> T): T =
            try {
                executor.submit(Callable { block() }).get(TIMEOUT_SECONDS, TimeUnit.SECONDS)
            } catch (e: ExecutionException) {
                throw e.cause ?: e
            }

        /** The engine that the tests share, made by `engineConfigFromOptions` of the reference. */
        private fun loadSharedEngine(): Engine =
            onEngineThread {
                Engine(engineConfigFromOptions(context, modelPath, MAX_TOKENS, backend())).also {
                    it.initialize()
                }
            }

        /**
         * Closes the shared engine, runs [block], which makes an engine of its own, and loads the
         * shared engine again: one engine at a time, as the `ChatViewModel` of the skill keeps it.
         */
        private fun <T> withTheSharedEngineClosed(block: () -> T): T {
            onEngineThread { engine.close() }
            try {
                return block()
            } finally {
                engine = loadSharedEngine()
            }
        }

        @JvmStatic
        @BeforeClass
        fun loadEngine() {
            assertTrue("no model at $modelPath: adb push one there", File(modelPath).canRead())
            engine = loadSharedEngine()
        }

        @JvmStatic
        @AfterClass
        fun closeEngine() {
            onEngineThread {
                if (::engine.isInitialized && engine.isInitialized()) {
                    engine.close()
                }
            }
            executor.shutdown()
        }
    }
}
