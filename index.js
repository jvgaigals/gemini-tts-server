// index.js — Gemini 2.5 TTS server for VAPI
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { GoogleGenAI } from '@google/genai';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import crypto from 'crypto';

// ---------- Config ----------
const PORT = process.env.PORT || 3000;
const DEFAULT_MODEL = process.env.MODEL || 'gemini-2.5-flash-preview-tts';
const DEFAULT_VOICE = process.env.VOICE || 'Kore';
const API_KEY = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;

if (!API_KEY) {
  console.error('❌ Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env');
  process.exit(1);
}

// ---------- SDK ----------
const ai = new GoogleGenAI({ apiKey: API_KEY });

// ---------- App ----------
const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));

// Static file hosting for URL-returned audio
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PUBLIC_AUDIO_DIR = path.join(__dirname, 'public', 'audio');
fs.mkdirSync(PUBLIC_AUDIO_DIR, { recursive: true });
app.use('/audio', express.static(PUBLIC_AUDIO_DIR)); // http://localhost:3000/audio/<file>.wav

// ---------- Helpers ----------
function pcm16ToWav(pcmBuffer, sampleRate = 24000, channels = 1, bitsPerSample = 16) {
  const header = Buffer.alloc(44);
  const byteRate = sampleRate * channels * (bitsPerSample / 8);
  const blockAlign = channels * (bitsPerSample / 8);

  header.write('RIFF', 0);
  header.writeUInt32LE(36 + pcmBuffer.length, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);       // fmt chunk size
  header.writeUInt16LE(1, 20);        // PCM format
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitsPerSample, 34);
  header.write('data', 36);
  header.writeUInt32LE(pcmBuffer.length, 40);

  return Buffer.concat([header, pcmBuffer]);
}

// Simple in-memory cache (cuts repeat latency)
const audioCache = new Map(); // key -> { wavBuf: Buffer, createdAt: number }
const CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes
const makeKey = ({ text, voice, model }) => `${model}::${voice}::${text}`;

async function synthesizeToWav({ text, voice, model }) {
  const key = makeKey({ text, voice, model });
  const now = Date.now();
  const hit = audioCache.get(key);
  if (hit && (now - hit.createdAt) < CACHE_TTL_MS) return hit.wavBuf;

  const response = await ai.models.generateContent({
    model,
    contents: [{ parts: [{ text }] }],
    config: {
      responseModalities: ['AUDIO'],
      speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } } },
    },
  });

  const base64 = response?.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (!base64) throw new Error('No audio produced by model');

  const pcm = Buffer.from(base64, 'base64');      // 24 kHz, 16-bit PCM mono
  const wavBuf = pcm16ToWav(pcm, 24000, 1, 16);   // wrap as WAV

  audioCache.set(key, { wavBuf, createdAt: now });
  return wavBuf;
}

// ---------- Routes ----------
app.get('/health', (_req, res) => {
  res.json({ ok: true, uptime: process.uptime(), model: DEFAULT_MODEL, voice: DEFAULT_VOICE });
});

app.get('/', (_req, res) => {
  res.send('Gemini TTS server ready. Endpoints: POST /tts, POST /tts-url, POST /batch-url, GET /health');
});

// POST /tts
// Body: { text: string, voice?: string, model?: string, return?: "base64" }
// - If return === "base64": responds JSON { audio (base64 wav), mimeType, sampleRate, channels }
// - Else: streams audio/wav bytes
app.post('/tts', async (req, res) => {
  try {
    const { text, voice, model, return: ret } = req.body || {};
    const t = (text || '').toString();
    if (!t.trim()) return res.status(400).json({ error: 'Missing required "text" string.' });

    const useModel = model || DEFAULT_MODEL;
    const useVoice = voice || DEFAULT_VOICE;

    const wavBuf = await synthesizeToWav({ text: t, voice: useVoice, model: useModel });

    if (ret === 'base64') {
      return res.json({
        audio: wavBuf.toString('base64'),
        mimeType: 'audio/wav',
        sampleRate: 24000,
        channels: 1,
        model: useModel,
        voice: useVoice
      });
    }

    res.setHeader('Content-Type', 'audio/wav');
    res.setHeader('Content-Disposition', 'inline; filename="speech.wav"');
    res.send(wavBuf);
  } catch (err) {
    console.error('POST /tts error:', err);
    res.status(500).json({ error: 'TTS failed', detail: err?.message });
  }
});

// POST /tts-url
// Body: { text: string, voice?: string, model?: string }
// Returns: { url, sampleRate, channels, model, voice }
app.post('/tts-url', async (req, res) => {
  try {
    const { text, voice, model } = req.body || {};
    const t = (text || '').toString();
    if (!t.trim()) return res.status(400).json({ error: 'Missing required "text" string.' });

    const useModel = model || DEFAULT_MODEL;
    const useVoice = voice || DEFAULT_VOICE;

    const wavBuf = await synthesizeToWav({ text: t, voice: useVoice, model: useModel });

    const id = crypto.randomUUID();
    const filename = `${id}.wav`;
    fs.writeFileSync(path.join(PUBLIC_AUDIO_DIR, filename), wavBuf);

    const url = `http://localhost:${PORT}/audio/${filename}`;
    res.json({ url, sampleRate: 24000, channels: 1, model: useModel, voice: useVoice });
  } catch (err) {
    console.error('POST /tts-url error:', err);
    res.status(500).json({ error: 'TTS failed', detail: err?.message });
  }
});

// POST /batch-url
// Body: { items: [{ text: string, voice?: string, model?: string }, ...] }
// Returns: { results: [{ text, url, model, voice, sampleRate, channels }...] }
app.post('/batch-url', async (req, res) => {
  try {
    const { items } = req.body || {};
    if (!Array.isArray(items) || items.length === 0) {
      return res.status(400).json({ error: 'Provide items: an array of { text, voice?, model? }' });
    }

    // Run in parallel for speed on larger batches
    const jobs = items.map(async (item) => {
      try {
        const t = (item?.text || '').toString();
        if (!t.trim()) return { error: 'Missing text' };
        const useModel = item?.model || DEFAULT_MODEL;
        const useVoice = item?.voice || DEFAULT_VOICE;

        const wavBuf = await synthesizeToWav({ text: t, voice: useVoice, model: useModel });
        const id = crypto.randomUUID();
        const filename = `${id}.wav`;
        fs.writeFileSync(path.join(PUBLIC_AUDIO_DIR, filename), wavBuf);
        const url = `http://localhost:${PORT}/audio/${filename}`;
        return { text: t, url, model: useModel, voice: useVoice, sampleRate: 24000, channels: 1 };
      } catch (e) {
        return { error: e?.message || 'synthesis failed' };
      }
    });

    const results = await Promise.all(jobs);
    res.json({ results });
  } catch (err) {
    console.error('POST /batch-url error:', err);
    res.status(500).json({ error: 'Batch TTS failed', detail: err?.message });
  }
});

// ---------- Start ----------
app.listen(PORT, () => {
  console.log(`✅ TTS server listening on http://localhost:${PORT}`);
  console.log('Routes: POST /tts, POST /tts-url, POST /batch-url, GET /health, GET /audio/<file>');
});
