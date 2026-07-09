/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.ai.edge.examples.rt_detr_v2

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [RtDetr] detector and exposes a single [UiState] stream for the screen. The detector
 * loads two GPU graphs, so both model creation and every inference run off the main thread.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val LABELS_ASSET = "coco_labels.txt"
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var detector: RtDetr? = null
  private var labels: List<String> = emptyList()

  // RtDetr reuses native input/output buffers, so every model call must run on the same single
  // worker. Confining to one thread serializes model load, warm-up, and each detection.
  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model and warm the GPU on a bundled image, then show that first result.
    viewModelScope.launch(inferenceDispatcher) {
      try {
        labels = context.assets.open(LABELS_ASSET).bufferedReader().use { it.readLines() }
        detector = RtDetr(context)
        detect(context.decodeAssetBitmap(DEMO_IMAGE_ASSET), warmUp = true)
      } catch (t: Throwable) {
        _uiState.update { it.copy(errorMessage = t.message ?: "Failed to load model") }
      }
    }
  }

  /** Runs detection on a gallery image picked by the user. */
  fun detect(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isDetecting = true) }
      try {
        detect(context.loadOrientedBitmap(uri), warmUp = false)
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(isDetecting = false, errorMessage = t.message ?: "Detection failed")
        }
      }
    }
  }

  /** Clears the error once it has been surfaced to the user. */
  fun errorMessageShown() {
    _uiState.update { it.copy(errorMessage = null) }
  }

  private fun detect(source: Bitmap, warmUp: Boolean) {
    val detector = detector ?: return
    val square = source.squareResize(RtDetr.SIZE)
    val rgb = square.toRgbFloatArray()
    if (warmUp) detector.detect(rgb) // First run compiles GPU shaders; discard its timing.
    val startNanos = System.nanoTime()
    val detections = detector.detect(rgb)
    val elapsedMs = (System.nanoTime() - startNanos) / 1_000_000
    _uiState.update {
      UiState(
        isModelReady = true,
        isDetecting = false,
        image = square,
        detections = detections.map { it.toDisplayBox() },
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  private fun RtDetr.Detection.toDisplayBox(): DetectionBox {
    val label = labels.getOrNull(cls)?.ifBlank { "id $cls" } ?: "id $cls"
    return DetectionBox(cls, label, score, cx, cy, w, h)
  }

  override fun onCleared() {
    super.onCleared()
    detector?.close()
  }
}
