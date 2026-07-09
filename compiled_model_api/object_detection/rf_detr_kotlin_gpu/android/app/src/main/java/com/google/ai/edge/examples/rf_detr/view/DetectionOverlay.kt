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

package com.google.ai.edge.examples.rf_detr.view

import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import com.google.ai.edge.examples.rf_detr.DetectionBox
import com.google.ai.edge.examples.rf_detr.RfDetr

private const val BOX_STROKE_WIDTH = 4f
private const val LABEL_TEXT_SIZE = 30f
private const val LABEL_PADDING = 6f
private const val LABEL_HEIGHT = 36f

/**
 * Draws the model input image scaled to fit [modifier]'s bounds, then overlays each detection box.
 * Box coordinates are normalized in the squared [RfDetr.SIZE] space, so they scale with the image.
 */
@Composable
fun DetectionOverlay(
  image: android.graphics.Bitmap?,
  boxes: List<DetectionBox>,
  modifier: Modifier = Modifier,
) {
  val boxPaint =
    Paint().apply {
      style = Paint.Style.STROKE
      strokeWidth = BOX_STROKE_WIDTH
    }
  val labelBackgroundPaint = Paint().apply { style = Paint.Style.FILL }
  val labelTextPaint =
    Paint().apply {
      color = Color.WHITE
      textSize = LABEL_TEXT_SIZE
      isFakeBoldText = true
    }

  Canvas(modifier) {
    val bitmap = image ?: return@Canvas
    val scale = minOf(size.width / bitmap.width, size.height / bitmap.height)
    drawIntoCanvas { canvas ->
      val nativeCanvas = canvas.nativeCanvas
      nativeCanvas.drawBitmap(
        bitmap,
        null,
        RectF(0f, 0f, bitmap.width * scale, bitmap.height * scale),
        null,
      )
      for (box in boxes) {
        val color = boxPalette[box.classId % boxPalette.size]
        boxPaint.color = color
        labelBackgroundPaint.color = color
        val left = (box.cx - box.width / 2) * RfDetr.SIZE * scale
        val top = (box.cy - box.height / 2) * RfDetr.SIZE * scale
        val right = (box.cx + box.width / 2) * RfDetr.SIZE * scale
        val bottom = (box.cy + box.height / 2) * RfDetr.SIZE * scale
        nativeCanvas.drawRect(left, top, right, bottom, boxPaint)

        val text = "${box.label} ${(box.score * 100).toInt()}%"
        val textWidth = labelTextPaint.measureText(text)
        nativeCanvas.drawRect(
          left,
          top - LABEL_HEIGHT,
          left + textWidth + LABEL_PADDING * 2,
          top,
          labelBackgroundPaint,
        )
        nativeCanvas.drawText(text, left + LABEL_PADDING, top - LABEL_PADDING - 2f, labelTextPaint)
      }
    }
  }
}
