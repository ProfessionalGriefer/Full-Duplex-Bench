#!/usr/bin/env node

// cli.js: Stream a WAV file to OpenAI Realtime API and record the combined conversation.

import fs from "fs";
import minimist from "minimist";
import fetch from "node-fetch";
import pkg from "wrtc";

const { RTCPeerConnection, nonstandard } = pkg;
import "dotenv/config";
import wav from "wav";

(async () => {
  ///////////////////////////////////////
  // IMPORTANT: Set your OpenAI API key
  const apiKey = process.env.OPENAI_API_KEY;
  ///////////////////////////////////////

  // Parse command-line arguments
  const argv = minimist(process.argv.slice(2), {
    string: ["input", "output", "model"],
    alias: { i: "input", o: "output", m: "model" },
  });

  // Normalize input path
  const inputRaw = argv.input;
  const inputPath = Array.isArray(inputRaw)
    ? inputRaw[inputRaw.length - 1]
    : inputRaw;

  // Normalize output path
  const outputRaw = argv.output;
  const outputPath = Array.isArray(outputRaw)
    ? outputRaw[outputRaw.length - 1]
    : outputRaw || "combined.wav";

  const model = argv.model || "gpt-4o-realtime-preview-2024-12-17";

  console.log(`[cli] Input: ${inputPath}`);
  console.log(`[cli] Output: ${outputPath}`);
  console.log(`[cli] Model: ${model}`);

  if (!inputPath) {
    console.error(
      "Usage: cli.js --input <path/to.wav> [--output <out.wav>] [--model <model-name>]",
    );
    process.exit(1);
  }

  // Read and decode WAV
  console.log(`[cli] Reading WAV file: ${inputPath}`);
  const reader = new wav.Reader();
  const inStream = fs.createReadStream(inputPath);
  let format;
  const pcmChunks = [];

  reader.on("format", (fmt) => {
    format = fmt;
  });
  reader.on("data", (data) => pcmChunks.push(data));
  reader.on("error", (err) => {
    console.error(`[cli] WAV reader error: ${err.message}`);
  });
  inStream.on("error", (err) => {
    console.error(`[cli] File stream error: ${err.message}`);
  });

  await new Promise((resolve) => {
    reader.on("end", resolve);
    inStream.pipe(reader);
  });

  if (!format) {
    console.error("[cli] Failed to parse WAV format from input file.");
    process.exit(1);
  }
  console.log(
    `[cli] WAV format: ${format.sampleRate}Hz, ${format.bitDepth}bit, ${format.channels}ch`,
  );

  // Flatten samples to Int16Array (downmix to mono)
  let samples;
  const { sampleRate: origSampleRate, bitDepth, channels } = format;
  const bytesPerSample = bitDepth / 8;
  const totalFrames =
    Buffer.concat(pcmChunks).length / (bytesPerSample * channels);
  samples = new Int16Array(totalFrames);
  const buffer = Buffer.concat(pcmChunks);
  for (let i = 0; i < totalFrames; i++) {
    let acc = 0;
    for (let c = 0; c < channels; c++) {
      const offset = (i * channels + c) * bytesPerSample;
      const sample =
        bitDepth === 32
          ? Math.round(
              Math.max(-1, Math.min(1, buffer.readFloatLE(offset))) * 32767,
            )
          : buffer.readInt16LE(offset);
      acc += sample;
    }
    samples[i] = Math.round(acc / channels);
  }

  // Resample to 48000 Hz
  const targetRate = 48000;
  let resampled;
  if (origSampleRate !== targetRate) {
    const ratio = targetRate / origSampleRate;
    const newLen = Math.floor(samples.length * ratio);
    resampled = new Int16Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const idx = i / ratio;
      const i0 = Math.floor(idx);
      const i1 = Math.min(i0 + 1, samples.length - 1);
      const frac = idx - i0;
      resampled[i] = Math.round(samples[i0] * (1 - frac) + samples[i1] * frac);
    }
  } else {
    resampled = samples;
  }

  // Build PCM buffer
  const inputPcm = Buffer.alloc(resampled.length * 2);
  for (let i = 0; i < resampled.length; i++) {
    inputPcm.writeInt16LE(resampled[i], i * 2);
  }

  const sampleRate = targetRate;
  const frameSize = sampleRate / 100; // 480 samples
  const frameBytes = frameSize * 2;

  // Session token
  if (!apiKey) {
    console.error("[cli] Missing OPENAI_API_KEY in environment");
    process.exit(1);
  }
  console.log("[cli] Requesting session token...");
  const sess = await fetch("https://api.openai.com/v1/realtime/sessions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: model,
      voice: "alloy", //"verse"
    }),
  });
  console.log(`[cli] Session response status: ${sess.status}`);
  const tokData = await sess.json();
  if (!tokData.client_secret?.value) {
    console.error(
      "[cli] Failed to obtain session token:",
      JSON.stringify(tokData, null, 2),
    );
    process.exit(1);
  }
  const token = tokData.client_secret.value;
  console.log("[cli] Session token obtained");

  // WebRTC setup
  console.log("[cli] Setting up WebRTC connection...");
  const pc = new RTCPeerConnection();
  const source = new nonstandard.RTCAudioSource();
  const track = source.createTrack();
  pc.addTrack(track);
  const gptBuffers = []; // { samples: Int16Array, time: bigint }
  let done = false;

  pc.onconnectionstatechange = () => {
    console.log(`[cli] WebRTC connection state: ${pc.connectionState}`);
  };
  pc.oniceconnectionstatechange = () => {
    console.log(`[cli] ICE connection state: ${pc.iceConnectionState}`);
  };

  pc.ontrack = ({ track: incomingTrack }) => {
    console.log("[cli] Received remote audio track");
    const sink = new nonstandard.RTCAudioSink(incomingTrack);
    sink.ondata = ({ samples }) => {
      gptBuffers.push({
        samples: new Int16Array(samples),
        time: process.hrtime.bigint(),
      });
    };
  };

  const dc = pc.createDataChannel("oai_events");
  dc.onopen = () => {
    console.log("[cli] Data channel opened");
  };
  dc.onerror = (err) => {
    console.error("[cli] Data channel error:", err);
  };
  dc.onclose = () => {
    console.log("[cli] Data channel closed");
  };
  dc.onmessage = (e) => {
    const m = JSON.parse(e.data);
    console.log(`[cli] Event: ${m.type}`);
    if (m.type === "error") {
      console.error("[cli] API error:", JSON.stringify(m, null, 2));
    }
    if (m.type === "response.done") done = true;
  };

  // SDP exchange
  console.log("[cli] Creating SDP offer...");
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  console.log("[cli] Sending SDP offer to OpenAI...");
  const sigRes = await fetch(
    `https://api.openai.com/v1/realtime?model=${model}`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/sdp",
      },
      body: offer.sdp,
    },
  );
  console.log(`[cli] SDP response status: ${sigRes.status}`);
  if (!sigRes.ok) {
    const errBody = await sigRes.text();
    console.error(`[cli] SDP exchange failed: ${errBody}`);
    process.exit(1);
  }
  const ans = await sigRes.text();
  await pc.setRemoteDescription({ type: "answer", sdp: ans });
  console.log("[cli] SDP exchange complete");

  // Record start time and Stream input
  const totalFrameCount = Math.ceil(inputPcm.length / frameBytes);
  console.log(
    `[cli] Streaming ${totalFrameCount} frames (${((totalFrameCount * 10) / 1000).toFixed(1)}s of audio)...`,
  );
  const startTime = process.hrtime.bigint();
  for (let off = 0; off < inputPcm.length; off += frameBytes) {
    let chunk = inputPcm.slice(off, off + frameBytes);
    if (chunk.length < frameBytes) {
      const pad = Buffer.alloc(frameBytes - chunk.length);
      chunk = Buffer.concat([chunk, pad]);
    }
    const temp = new Int16Array(chunk.buffer, chunk.byteOffset, frameSize);
    const frame = Int16Array.from(temp);
    source.onData({
      samples: frame,
      sampleRate,
      bitsPerSample: 16,
      channelCount: 1,
    });
    await new Promise((r) => setTimeout(r, 10));
  }

  // Signal end of input
  console.log("[cli] Finished streaming input, waiting for response...");
  track.stop();

  // Wait with timeout
  const timeoutMs = 60000;
  const waitStart = Date.now();
  while (!done) {
    if (Date.now() - waitStart > timeoutMs) {
      console.error(
        `[cli] Timed out after ${timeoutMs / 1000}s waiting for response.done. ` +
          `Received ${gptBuffers.length} audio chunks so far.`,
      );
      process.exit(1);
    }
    await new Promise((r) => setTimeout(r, 50));
  }
  console.log(
    `[cli] Response complete. Received ${gptBuffers.length} audio chunks.`,
  );

  // First, save GPT response as a separate WAV file
  console.log("Saving GPT response...");

  // Concatenate all GPT audio chunks
  let totalGptSamples = 0;
  gptBuffers.forEach(({ samples }) => {
    totalGptSamples += samples.length;
  });

  const gptAudio = new Int16Array(totalGptSamples);
  let gptOffset = 0;
  gptBuffers.forEach(({ samples }) => {
    gptAudio.set(samples, gptOffset);
    gptOffset += samples.length;
  });

  // Calculate when GPT response should start (based on first buffer timestamp)
  let gptStartOffset = 0;
  if (gptBuffers.length > 0) {
    const firstGptTime = gptBuffers[0].time;
    const delta = Number(firstGptTime - startTime) / 1e6; // ms
    gptStartOffset = Math.round((delta * sampleRate) / 1000);
  }

  // Now combine the two audio files
  const inputSamples = new Int16Array(inputPcm.buffer);

  // Truncate GPT response to input length before saving
  let gptAudioTruncated = gptAudio;
  if (gptAudio.length > inputSamples.length) {
    gptAudioTruncated = gptAudio.slice(0, inputSamples.length);
    console.log(
      "Note: GPT response file was truncated to match input duration",
    );
  }

  // Write GPT response to separate file
  const gptPath = outputPath.replace(".wav", "_gpt_response.wav");
  const gptWriter = new wav.Writer({ sampleRate, channels: 1, bitDepth: 16 });
  const gptStream = fs.createWriteStream(gptPath);
  gptWriter.pipe(gptStream);
  gptWriter.write(
    Buffer.from(
      gptAudioTruncated.buffer,
      gptAudioTruncated.byteOffset,
      gptAudioTruncated.byteLength,
    ),
  );
  gptWriter.end();

  // Wait for GPT file to finish writing
  await new Promise((resolve) => {
    gptStream.on("finish", resolve);
  });
  console.log(`GPT response saved to: ${gptPath}`);

  dc.close();
  pc.close();
  process.exit(0);
})();

