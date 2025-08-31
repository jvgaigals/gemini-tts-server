// scripts/test-tts.js
import 'dotenv/config';
import { GoogleGenAI } from '@google/genai';
import wav from 'wav';
import fs from 'fs';

console.log('GEMINI_API_KEY length:', (process.env.GEMINI_API_KEY || '').length);
console.log('GEMINI_API_KEY prefix/suffix:', (process.env.GEMINI_API_KEY || '').slice(0,4), '...', (process.env.GEMINI_API_KEY || '').slice(-4));


const MODEL = process.env.MODEL || 'gemini-2.5-flash-preview-tts';
const VOICE = process.env.VOICE || 'Kore';

// Helper to save PCM16 24kHz mono as WAV
function saveWav(filename, pcmBuffer, { channels = 1, sampleRate = 24000, bitDepth = 16 } = {}) {
  return new Promise((resolve, reject) => {
    const writer = new wav.FileWriter(filename, { channels, sampleRate, bitDepth });
    writer.on('finish', resolve);
    writer.on('error', reject);
    writer.write(pcmBuffer);
    writer.end();
  });
}

async function main() {
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY });
  const text = 'Say cheerfully: Sveiki! Rezervācija ir apstiprināta uz pulksten pieciem vakarā.'; // test line

  const response = await ai.models.generateContent({
    model: MODEL,
    contents: [{ parts: [{ text }] }],
    config: {
      responseModalities: ['AUDIO'],
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: VOICE } },
      },
    },
  });

  const base64 = response?.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (!base64) throw new Error('No audio data returned');

  const pcm = Buffer.from(base64, 'base64');
  await saveWav('out.wav', pcm);
  console.log('Wrote out.wav');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
