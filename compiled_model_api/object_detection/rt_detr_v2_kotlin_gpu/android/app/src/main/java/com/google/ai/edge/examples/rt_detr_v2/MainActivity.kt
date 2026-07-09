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

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.examples.rt_detr_v2.view.ApplicationTheme
import com.google.ai.edge.examples.rt_detr_v2.view.DetectionScreen

/**
 * D-FINE-S object detection demo. Both transformer graphs run on the LiteRT CompiledModel GPU; only
 * top-k selection, the per-token tail, decode and NMS run on the CPU (see [RtDetr]). The UI is a thin
 * Compose host over [MainViewModel]; all detection state lives in the view model.
 */
class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }
    setContent {
      ApplicationTheme {
        val context = LocalContext.current
        val uiState by viewModel.uiState.collectAsStateWithLifecycle()

        val galleryLauncher =
          rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
            if (uri != null) viewModel.detect(uri)
          }

        LaunchedEffect(uiState.errorMessage) {
          uiState.errorMessage?.let { message ->
            Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
            viewModel.errorMessageShown()
          }
        }

        DetectionScreen(
          uiState = uiState,
          onPickImage = {
            galleryLauncher.launch(
              PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
            )
          },
        )
      }
    }
  }
}
