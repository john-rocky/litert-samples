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

package com.google.ai.edge.examples.rt_detr_v2.view

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.Button
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Scaffold
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.ai.edge.examples.rt_detr_v2.DetectionBox
import com.google.ai.edge.examples.rt_detr_v2.R
import com.google.ai.edge.examples.rt_detr_v2.UiState

/** Top-level detection screen: a status header, an image picker, and the annotated result. */
@Composable
fun DetectionScreen(uiState: UiState, onPickImage: () -> Unit, modifier: Modifier = Modifier) {
  Scaffold(
    modifier = modifier.statusBarsPadding(),
    topBar = {
      TopAppBar(
        backgroundColor = MaterialTheme.colors.secondary,
        title = { Text(text = stringResource(R.string.app_name), color = Color.White) },
      )
    },
  ) { padding ->
    Column(modifier = Modifier.fillMaxSize().padding(padding).padding(16.dp)) {
      StatusHeader(uiState)
      Spacer(modifier = Modifier.height(12.dp))
      Button(onClick = onPickImage, enabled = uiState.isModelReady && !uiState.isDetecting) {
        Text(text = stringResource(R.string.action_pick_image))
      }
      Spacer(modifier = Modifier.height(12.dp))
      DetectionOverlay(
        image = uiState.image,
        boxes = uiState.detections,
        modifier = Modifier.fillMaxWidth().weight(1f),
      )
      ResultList(detections = uiState.detections)
    }
  }
}

@Composable
private fun StatusHeader(uiState: UiState) {
  val statusText =
    when {
      !uiState.isModelReady -> stringResource(R.string.status_loading)
      uiState.isDetecting -> stringResource(R.string.status_detecting)
      else ->
        stringResource(R.string.status_ready, uiState.detections.size, uiState.inferenceTimeMs)
    }
  Column {
    Text(text = statusText, fontSize = 16.sp, fontWeight = FontWeight.Medium)
    if (uiState.isModelReady && !uiState.isDetecting) {
      Text(text = stringResource(R.string.model_subtitle), fontSize = 13.sp, color = Color.Gray)
    }
  }
}

@Composable
private fun ResultList(detections: List<DetectionBox>, modifier: Modifier = Modifier) {
  if (detections.isEmpty()) {
    Text(text = stringResource(R.string.no_objects), modifier = modifier.padding(top = 8.dp))
    return
  }
  LazyColumn(modifier = modifier.fillMaxWidth().heightIn(max = 160.dp).padding(top = 8.dp)) {
    items(detections) { detection ->
      Text(
        text =
          stringResource(R.string.result_row, detection.label, (detection.score * 100).toInt()),
        fontSize = 14.sp,
      )
    }
  }
}
