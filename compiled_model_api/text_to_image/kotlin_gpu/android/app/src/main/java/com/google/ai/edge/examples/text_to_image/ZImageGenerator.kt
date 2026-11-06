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

package com.google.ai.edge.examples.text_to_image

import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litert.Environment
import java.io.File

/**
 * Full on-device Z-Image-Turbo generation with the chunked S3-DiT.
 *
 * The 6B S3-DiT is split into int8 graphs that each compile fully on the ML Drift GPU delegate and
 * load one at a time (FP32 compute — the adaLN/attention path overflows fp16), so the peak
 * footprint is a single sub-1 GB graph and the model fits an 8 GB phone. The text encoder,
 * tokenizer, RoPE and scheduler are precomputed on the host and staged as `.bin` (see the
 * conversion scripts). The denoising loop, the pad-token mask, the x/c concat, classifier-free
 * guidance and (un)patchify run on the host, and the VAE decodes the final latent to an image.
 *
 * Per step (Z-Image CFG): `pos = DiT(cond)`, `neg = DiT(uncond)`,
 * `noise_pred = -(pos + guidance * (pos - neg))`, `latent += dsigma * noise_pred`. The image branch
 * (embx -> refx) is shared by cond/uncond, the context branch (embc -> refc) is step-independent
 * and computed once per prompt.
 *
 * All graphs (`z_*.tflite`, `zc_*.tflite`, `zvae_int8_256.tflite`) and host inputs
 * (`gen_bins/`) are staged to the app's external files dir. The first launch before staging
 * fails with "model not found".
 */
class ZImageGenerator(context: Context) {

  private val dir = context.getExternalFilesDir(null)!!

  private fun readFloats(name: String) = readFloatBin(File(dir, "gen_bins/$name.bin"))

  private fun readInts(name: String) = readIntBin(File(dir, "gen_bins/$name.bin"))

  // Fixed host inputs (encoder output, RoPE, scheduler sigmas, permutations, pad tokens).
  private val capCond = readFloats("cap_b0")
  private val capUnc = readFloats("cap_b1")
  private val adaln = readFloats("adaln") // [STEPS * 256]
  private val xc = readFloats("xc")
  private val xs = readFloats("xs")
  // The cond and uncond prompts differ in length, so each branch carries its own context
  // RoPE (cc/cs), cap-pad mask and unified RoPE (uc/us). The image branch is shared.
  private val cc = readFloats("cc")
  private val cs = readFloats("cs")
  private val uc = readFloats("uc")
  private val us = readFloats("us")
  private val ccUnc = readFloats("cc_unc")
  private val csUnc = readFloats("cs_unc")
  private val ucUnc = readFloats("uc_unc")
  private val usUnc = readFloats("us_unc")
  private val cpadUnc = readFloats("cpad_unc")
  private val xpad = readFloats("xpad")
  private val cpad = readFloats("cpad")
  private val xpt = readFloats("xpt")
  private val cpt = readFloats("cpt")
  private val dsigma = readFloats("dsigma")
  private val patchPerm = readInts("patch_perm")
  private val unpatchPerm = readInts("unpatch_perm")
  private val initialLatent = readFloats("steps_0")

  /**
   * Generates the image for the precomputed prompt, reporting per-stage progress through
   * [onProgress]. A single [Environment] is shared across every graph.
   */
  fun generate(onProgress: (String) -> Unit): Bitmap =
    Environment.create().use { env -> denoise(env, onProgress) }

  private fun denoise(env: Environment, onProgress: (String) -> Unit): Bitmap {
    val latent = initialLatent.copyOf()
    val startMs = System.currentTimeMillis()

    // The context branch (no adaLN) is step-independent, so refine cond/uncond once.
    onProgress("Preparing context…")
    val contextCond = contextRef(env, capCond, cc, cs, cpad)
    val contextUnc = contextRef(env, capUnc, ccUnc, csUnc, cpadUnc)

    for (step in 0 until STEPS) {
      val stepAdaln = adaln.copyOfRange(step * ADALN_DIM, step * ADALN_DIM + ADALN_DIM)
      // Image branch, shared by cond + uncond.
      val imageTokens = gather(latent, patchPerm)
      val embedded = ChunkRunner.gpu(env, "z_embx.tflite", dir, listOf(imageTokens))
      val masked = padMask(embedded, xpad, xpt, NUM_IMAGE_TOKENS)
      val imageRef = ChunkRunner.gpu(env, "z_refx.tflite", dir, listOf(masked, xc, xs, stepAdaln))

      val positive = mainBranch(env, imageRef, contextCond, stepAdaln, uc, us)
      val negative = mainBranch(env, imageRef, contextUnc, stepAdaln, ucUnc, usUnc)

      // Z-Image classifier-free guidance + flow-matching Euler update (host).
      for (i in 0 until LATENT_SIZE) {
        val noisePred = -(positive[i] + GUIDANCE_SCALE * (positive[i] - negative[i]))
        latent[i] += dsigma[step] * noisePred
      }
      onProgress("Step ${step + 1}/$STEPS (${elapsed(startMs)})")
    }

    // The VAE consumes the denormalized latent: latent / scaling_factor + shift_factor.
    for (i in 0 until LATENT_SIZE) {
      latent[i] = latent[i] / VAE_SCALING_FACTOR + VAE_SHIFT_FACTOR
    }
    val image = ChunkRunner.gpu(env, "zvae_int8_256.tflite", dir, listOf(latent))
    onProgress("Decoded (${elapsed(startMs)})")
    return toBitmap(image)
  }

  /** Caption branch: embed -> host pad mask -> context refiner (step-independent). */
  private fun contextRef(
    env: Environment,
    caption: FloatArray,
    contextCos: FloatArray,
    contextSin: FloatArray,
    contextPad: FloatArray,
  ): FloatArray {
    val embedded = ChunkRunner.gpu(env, "z_embc.tflite", dir, listOf(caption))
    val masked = padMask(embedded, contextPad, cpt, NUM_CONTEXT_TOKENS)
    return ChunkRunner.gpu(env, "z_refc.tflite", dir, listOf(masked, contextCos, contextSin))
  }

  /** Main-layer stack + final for one branch: cat(imageRef, contextRef) -> mains -> final. */
  private fun mainBranch(
    env: Environment,
    imageRef: FloatArray,
    contextRef: FloatArray,
    stepAdaln: FloatArray,
    unifiedCos: FloatArray,
    unifiedSin: FloatArray,
  ): FloatArray {
    var hidden = imageRef + contextRef // concat along the sequence axis -> [288 * 3840]
    for (i in 0 until NUM_MAIN_CHUNKS) {
      val inputs = listOf(hidden, unifiedCos, unifiedSin, stepAdaln)
      hidden = ChunkRunner.gpu(env, "zc_main$i.tflite", dir, inputs)
    }
    val out = ChunkRunner.gpu(env, "zc_final.tflite", dir, listOf(hidden, stepAdaln))
    return gather(out, unpatchPerm) // unpatchify -> [16 * 32 * 32]
  }

  /** out[i] = src[perm[i]] (patchify / unpatchify via a precomputed index map). */
  private fun gather(src: FloatArray, perm: IntArray): FloatArray {
    val out = FloatArray(perm.size)
    for (i in perm.indices) {
      out[i] = src[perm[i]]
    }
    return out
  }

  /**
   * Substitutes the learned pad token at padded positions: `raw[t]` where `pad[t] == 0`,
   * `padToken` where `pad[t] == 1`. Done on the host because a MUL right after the embed FC trips
   * the Mali "bc coord for BATCH axis" compile wall.
   */
  private fun padMask(
    raw: FloatArray,
    pad: FloatArray,
    padToken: FloatArray,
    tokens: Int,
  ): FloatArray {
    val out = FloatArray(tokens * DIM)
    for (t in 0 until tokens) {
      val p = pad[t]
      val keep = 1f - p
      val base = t * DIM
      for (ch in 0 until DIM) {
        out[base + ch] = raw[base + ch] * keep + padToken[ch] * p
      }
    }
    return out
  }

  /** [1,3,256,256] planar RGB in [-1,1] -> ARGB bitmap. */
  private fun toBitmap(image: FloatArray): Bitmap {
    val hw = IMAGE_SIZE * IMAGE_SIZE
    val pixels = IntArray(hw)
    for (p in 0 until hw) {
      val r = ((image[p].coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
      val g = ((image[hw + p].coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
      val b = ((image[2 * hw + p].coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
      pixels[p] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
    return Bitmap.createBitmap(pixels, IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888)
  }

  private fun elapsed(startMs: Long) = "${(System.currentTimeMillis() - startMs) / 1000f}s"

  companion object {
    const val PROMPT = "a red apple on a wooden table, studio lighting"
    private const val STEPS = 8
    private const val VAE_SCALING_FACTOR = 0.3611f
    private const val VAE_SHIFT_FACTOR = 0.1159f
    private const val NUM_MAIN_CHUNKS = 6
    private const val NUM_IMAGE_TOKENS = 256
    private const val NUM_CONTEXT_TOKENS = 32
    private const val ADALN_DIM = 256
    private const val DIM = 3840
    private const val IMAGE_SIZE = 256
    private const val LATENT_SIZE = 16 * 32 * 32
    private const val GUIDANCE_SCALE = 1.0f

    private fun readFloatBin(file: File): FloatArray {
      val bytes = file.readBytes()
      val out = FloatArray(bytes.size / 4)
      var j = 0
      for (i in out.indices) {
        out[i] = Float.fromBits(littleEndianInt(bytes, j))
        j += 4
      }
      return out
    }

    private fun readIntBin(file: File): IntArray {
      val bytes = file.readBytes()
      val out = IntArray(bytes.size / 4)
      var j = 0
      for (i in out.indices) {
        out[i] = littleEndianInt(bytes, j)
        j += 4
      }
      return out
    }

    private fun littleEndianInt(bytes: ByteArray, offset: Int): Int {
      return (bytes[offset].toInt() and 0xff) or
        ((bytes[offset + 1].toInt() and 0xff) shl 8) or
        ((bytes[offset + 2].toInt() and 0xff) shl 16) or
        ((bytes[offset + 3].toInt() and 0xff) shl 24)
    }
  }
}
