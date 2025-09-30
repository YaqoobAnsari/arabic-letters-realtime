#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serve_realtime_fastapi.py
Headless realtime demo for your fine-tuned wav2vec2-base Arabic letters model.
Enhanced with pronunciation hints based on accuracy levels.

- Pure FastAPI + vanilla HTML/JS (records mic, encodes WAV in-browser)
- No Gradio. No server-side ffmpeg.
- GPU if available. Serves a premium dashboard with metrics.
- Optional --share uses ngrok if NGROK_AUTHTOKEN is set.

Run:
  python serve_realtime_fastapi.py --host 0.0.0.0 --port 7860
  # or with a public link (requires: pip install pyngrok && export NGROK_AUTHTOKEN=...):
  python serve_realtime_fastapi.py --host 0.0.0.0 --port 7860 --share
"""

from __future__ import annotations
import os, io, time, json, argparse, base64, threading
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse
import uvicorn

try:
    import resampy
    HAVE_RESAMPY = True
except Exception:
    HAVE_RESAMPY = False

from transformers import AutoProcessor, AutoModelForAudioClassification


# --------------------------- Pronunciation Hints Database ---------------------------

# Mapping from English letter names to Arabic characters
LETTER_NAME_TO_ARABIC = {
    "Alif": "ا", "Baa": "ب", "Taa": "ت", "Thaa": "ث", "Jeem": "ج",
    "Haa": "ح", "Khaa": "خ", "Daal": "د", "Dhaal": "ذ", "Raa": "ر",
    "Zay": "ز", "Seen": "س", "Sheen": "ش", "Saad": "ص", "Daad": "ض",
    "Taa (emphatic)": "ط", "Zaa": "ظ", "Ayn": "ع", "Ghayn": "غ",
    "Faa": "ف", "Qaaf": "ق", "Kaaf": "ك", "Laam": "ل", "Meem": "م",
    "Noon": "ن", "Haa (light)": "ه", "Waaw": "و", "Yaa": "ي",
    # Alternative spellings
    "Ta": "ط", "Ghain": "غ", "Qaf": "ق", "Kaf": "ك", "Waw": "و", "Ya": "ي"
}

PRONUNCIATION_HINTS = {
    # Throat letters (most challenging)
    "ع": {
        "name": "Ayn",
        "low": [
            "Ayn (ع) is produced deep in your pharynx (throat). <strong>Position:</strong> Open your mouth wide, drop your jaw, and constrict the muscles at the very back of your throat while pushing air through.",
            "<strong>Technique:</strong> Imagine you're gently choking while saying 'ah'. The sound comes from your throat walls squeezing together, NOT from your mouth or tongue. Your tongue should remain flat and relaxed.",
            "<strong>Practice drill:</strong> Say 'ah' normally, then gradually tighten your throat muscles mid-sound. You should feel pressure and constriction deep in your throat. The sound will become more compressed and darker.",
            "<strong>Common mistake:</strong> Don't just say a vowel 'a' sound - Ayn requires actual muscular constriction in your pharynx. Place your hand on your throat and feel the muscles engage when you pronounce it correctly."
        ],
        "medium": [
            "<strong>Refinement needed:</strong> Your Ayn needs more pharyngeal constriction. The sound must originate from deep throat compression, not oral cavity.",
            "<strong>Tongue position:</strong> Keep your tongue flat and LOW in your mouth - it should NOT touch your teeth or palate. All work happens in your throat muscles.",
            "<strong>Practice:</strong> Try pronouncing عين (eye) - squeeze your throat muscles at the start. The constriction should feel like pressure building up before release."
        ],
        "high": [
            "<strong>Excellent!</strong> Your pharyngeal constriction is strong. To perfect it further: ensure consistent throat muscle engagement throughout the entire sound duration. The constriction shouldn't weaken mid-pronunciation."
        ]
    },
    
    "غ": {
        "name": "Ghayn",
        "low": [
            "Ghayn (غ) is a voiced uvular fricative. <strong>Position:</strong> The back of your tongue rises toward your uvula (the hanging tissue at the back of your throat) while your vocal cords vibrate.",
            "<strong>How to produce it:</strong> Gargle water, then try to make that same gargling sensation WITHOUT water. Your uvula vibrates against the back of your raised tongue while air flows through.",
            "<strong>Vocal cords:</strong> MUST vibrate! Put your hand on your throat - you should feel strong buzzing. This is what distinguishes Ghayn from Khaa (which has no vibration).",
            "<strong>Practice sequence:</strong> (1) Gargle water, (2) Make gargling sound without water, (3) Try to speak through that position. Your soft palate and uvula area should feel active and vibrating."
        ],
        "medium": [
            "<strong>Your Ghayn needs more uvular vibration.</strong> Ensure the back of your tongue rises HIGH toward your soft palate, creating friction as air passes.",
            "<strong>Key difference:</strong> Ghayn is VOICED (vocal cords vibrate) - it should feel like humming while gargling. Don't confuse it with Khaa (خ) which is voiceless.",
            "<strong>Drill:</strong> Hum continuously, then raise the back of your tongue to create the gargling friction. The humming + friction = perfect Ghayn."
        ],
        "high": [
            "<strong>Great uvular articulation!</strong> Keep that vocal cord vibration consistent. The sound should be smooth and continuous, not choppy or interrupted."
        ]
    },
    
    "ح": {
        "name": "Haa",
        "low": [
            "Haa (ح) is a voiceless pharyngeal fricative. <strong>Position:</strong> Your pharynx (middle throat) constricts while air flows forcefully through - like fogging a mirror with your throat, not your mouth.",
            "<strong>Airflow technique:</strong> Push a strong, continuous stream of air from deep in your throat. The air should feel HOT as it passes through the constricted pharyngeal passage.",
            "<strong>No vocal cords:</strong> Unlike Ayn (ع), your vocal cords do NOT vibrate. It's pure air friction. Put your hand on your throat - you should feel warmth and airflow, but no buzzing.",
            "<strong>Visual cue:</strong> Imagine breathing on cold glass to fog it up, but move the constriction point from your mouth back to your throat. Your mouth stays open, throat does all the work."
        ],
        "medium": [
            "<strong>More pharyngeal friction needed.</strong> Push MORE air through a MORE constricted throat passage. The sound should be breathy but forceful.",
            "<strong>Position check:</strong> Your tongue should be flat and low. All constriction happens in your throat walls, creating a narrow passage for air.",
            "<strong>Don't confuse with:</strong> ه (light Ha) is much softer. ح requires strong pharyngeal engagement - you should feel throat muscle tension."
        ],
        "high": [
            "<strong>Excellent pharyngeal airflow!</strong> Maintain that strong, continuous air stream. The sound should be consistent from start to finish without weakening."
        ]
    },
    
    "خ": {
        "name": "Khaa",
        "low": [
            "Khaa (خ) is a voiceless uvular fricative - the harsh 'ch' in Scottish 'loch'. <strong>Position:</strong> Raise the back of your tongue toward your soft palate/uvula, creating friction WITHOUT vocal cord vibration.",
            "<strong>Technique:</strong> Clear your throat gently (that 'khhh' sound), then hold that position and push air through. It should sound raspy and harsh.",
            "<strong>Tongue position:</strong> Back of tongue rises HIGH, almost touching your soft palate. Air squeezes through the narrow gap, creating a rough, gravelly sound.",
            "<strong>Key distinction:</strong> NO vocal cord vibration (that's what makes it different from غ Ghayn). It's pure friction sound - rough, breathy, voiceless."
        ],
        "medium": [
            "<strong>Make it rougher and harsher.</strong> Raise the back of your tongue HIGHER to narrow the air passage more, creating more friction.",
            "<strong>Voiceless check:</strong> Ensure NO vocal cord vibration. It should be pure breathy friction - like wind through a narrow passage.",
            "<strong>Practice:</strong> German 'Bach' or Spanish 'jota' (José) have similar sounds. Channel that harsh, throaty roughness."
        ],
        "high": [
            "<strong>Solid Khaa!</strong> Keep that harsh, voiceless quality consistent. The friction should be continuous and rough throughout the sound."
        ]
    },
    
    "ق": {
        "name": "Qaf",
        "low": [
            "Qaf (ق) is a voiceless uvular plosive - a deep, forceful 'k' from your throat. <strong>Position:</strong> The very back of your tongue presses firmly against your soft palate/uvula, then releases explosively.",
            "<strong>How to produce it:</strong> (1) Raise the back of your tongue ALL THE WAY UP to touch your soft palate, (2) Build up air pressure behind the closure, (3) Release explosively. It should sound like a suppressed cough.",
            "<strong>Location is KEY:</strong> This happens MUCH deeper than English 'k'. English 'k' touches the hard palate (roof of mouth), but Qaf touches the SOFT palate (further back, near your throat).",
            "<strong>Force and depth:</strong> This sound requires POWER. It should feel deep, heavy, and forceful - like you're pushing the sound from deep in your throat. Practice: قلب (qalb - heart), قرآن (Quran)."
        ],
        "medium": [
            "<strong>Go deeper and more forceful.</strong> Your tongue contact point needs to be FURTHER BACK - at your soft palate/uvular area, not the hard palate.",
            "<strong>Don't pronounce like English 'k' or Kaaf (ك).</strong> Qaf is much heavier and deeper. You should feel the contact point near where you feel Khaa and Ghayn.",
            "<strong>Power technique:</strong> Build up air pressure behind the tongue closure, then release explosively. It should sound like a controlled throat punch."
        ],
        "high": [
            "<strong>Excellent uvular stop!</strong> Your depth and force are correct. Maintain that heavy, guttural quality - this is one of Arabic's most distinctive sounds."
        ]
    },
    
    # Emphatic letters
    "ص": {
        "name": "Saad",
        "low": [
            "Saad (ص) is an emphatic (pharyngealized) 'S'. <strong>Tongue technique:</strong> (1) Position tongue tip near upper teeth for 's', (2) Simultaneously RAISE and TENSE the back/sides of your tongue, (3) Pull the middle of your tongue slightly back.",
            "<strong>The 'emphasis' explained:</strong> While making 's', create a secondary constriction in your pharynx by raising the back of your tongue. This creates a 'darker', 'thicker', 'heavier' resonance.",
            "<strong>Tongue shape:</strong> Your tongue forms a 'trough' or 'spoon' shape - edges raised, middle hollowed. This pharyngealization deepens all surrounding vowels.",
            "<strong>Compare directly:</strong> Say seen (س) 'sa' (light), then Saad (ص) 'sa' (heavy). Feel how your whole tongue changes position - back rises, creates throat resonance."
        ],
        "medium": [
            "<strong>More emphasis needed.</strong> Raise the back of your tongue HIGHER while maintaining the 's' sound. Your tongue should feel tense and raised.",
            "<strong>Pharyngealization check:</strong> The vowels around Saad should sound darker/deeper. If they sound the same as regular 's', you're not emphasizing enough.",
            "<strong>Technique:</strong> Think of it as making TWO sounds simultaneously - 's' in the front of your mouth + throat constriction in the back."
        ],
        "high": [
            "<strong>Good emphatic quality!</strong> Keep that tongue tension and pharyngeal resonance consistent. The 'heaviness' should be maintained throughout."
        ]
    },
    
    "ض": {
        "name": "Daad",
        "low": [
            "Daad (ض) is THE unique Arabic sound - an emphatic 'D' articulated from the SIDES of your tongue. <strong>Position:</strong> Press one or both sides of your tongue firmly against your upper molars (back teeth).",
            "<strong>Complex articulation:</strong> (1) Sides of tongue touch upper molars, (2) Build air pressure, (3) Release with emphasis while keeping back of tongue RAISED for pharyngealization. Hold the sound briefly.",
            "<strong>Why it's difficult:</strong> Most languages use tongue TIP for 'd', but Daad uses tongue SIDES + emphasis. Your tongue edges should feel firm pressure against your molars.",
            "<strong>The emphasis component:</strong> While articulating from tongue sides, simultaneously raise the back of your tongue toward your soft palate. This creates the 'heavy' quality. Your cheeks may puff slightly from the air pressure."
        ],
        "medium": [
            "<strong>Refine your lateral articulation.</strong> Ensure FIRM contact between tongue SIDES and upper molars. The release should be forceful and emphatic.",
            "<strong>Don't pronounce like regular د (Daal).</strong> Daad requires: (1) lateral (side) tongue contact, (2) pharyngealization (raised tongue back), (3) more force and duration.",
            "<strong>Practice sensation:</strong> You should feel your tongue sides pressing HARD against your back teeth, with tension in the back of your tongue."
        ],
        "high": [
            "<strong>Excellent Daad!</strong> You've mastered this uniquely Arabic sound. Keep that lateral articulation and emphatic quality strong."
        ]
    },
    
    "ط": {
        "name": "Taa (emphatic)",
        "low": [
            "Emphatic Taa (ط) is a pharyngealized 'T'. <strong>Position:</strong> Tongue tip presses firmly against upper gum ridge (like regular 't'), BUT simultaneously raise and tense the back of your tongue.",
            "<strong>The emphasis technique:</strong> While making 't' sound, create secondary throat constriction by raising your tongue back toward soft palate. This creates a 'fuller', 'deeper' resonance.",
            "<strong>Force and tension:</strong> Use MORE muscular force than English 't'. Your tongue should press HARDER against the gum ridge, and the back of your tongue should feel TENSE and RAISED.",
            "<strong>Acoustic effect:</strong> The emphasis darkens surrounding vowels. Compare: تين (teen - regular t, light) vs طين (teen - emphatic t, heavy). Feel the difference in tongue posture."
        ],
        "medium": [
            "<strong>Add more emphasis and force.</strong> Your tongue tip presses harder against gums, while tongue back raises higher toward soft palate.",
            "<strong>Don't pronounce like light ت (Taa).</strong> Emphatic ط requires full tongue body involvement - not just the tip.",
            "<strong>Check:</strong> Surrounding vowels should sound darker/deeper. If 'a' sounds the same as in light letters, increase your pharyngealization."
        ],
        "high": [
            "<strong>Strong emphatic Taa!</strong> Maintain that forceful articulation and pharyngeal quality. The heaviness should be unmistakable."
        ]
    },
    
    "ظ": {
        "name": "Zaa (emphatic)",
        "low": [
            "Emphatic Zaa (ظ) is a pharyngealized 'th' (as in 'the'). <strong>Position:</strong> Tongue tip touches bottom edge of upper front teeth (like 'th'), BUT raise the back of your tongue for emphasis.",
            "<strong>Voiced + emphatic:</strong> (1) Vocal cords MUST vibrate (touch your throat to feel buzzing), (2) Tongue back raises toward soft palate (creates 'heavy' quality), (3) More force than light ذ (Dhaal).",
            "<strong>Tongue configuration:</strong> Front: tongue tip at teeth for 'th'. Back: tongue body raised and tense for pharyngealization. This creates the characteristic 'thick' sound.",
            "<strong>Common struggle:</strong> Maintaining BOTH the dental position AND the pharyngeal emphasis simultaneously. Practice slowly: establish 'th' position, then add tongue back raising."
        ],
        "medium": [
            "<strong>Your emphasis needs strengthening.</strong> While keeping tongue tip at teeth, raise the back of your tongue HIGHER to create more pharyngeal resonance.",
            "<strong>Full vibration needed.</strong> Ensure strong vocal cord vibration throughout - it should buzz noticeably in your throat.",
            "<strong>Don't confuse with light ذ (Dhaal).</strong> Zaa is much heavier, deeper, and more forceful. The emphasis should be obvious."
        ],
        "high": [
            "<strong>Great emphatic Zaa!</strong> You're maintaining both the dental articulation and the pharyngeal emphasis well. Keep it consistent."
        ]
    },
    
    # Commonly confused pairs
    "ث": {
        "name": "Thaa",
        "low": [
            "Thaa (ث) is the voiceless 'th' as in 'think'. <strong>Position:</strong> Place your tongue tip against or between your upper and lower front teeth, creating a narrow gap for air to pass through.",
            "<strong>Airflow:</strong> Push air through the narrow gap between your tongue and teeth. The air friction creates the 'th' sound - it should sound like a soft whistle or hiss.",
            "<strong>VOICELESS is key:</strong> Your vocal cords do NOT vibrate. Place your hand on your throat - you should feel NO buzzing. It's pure air sound.",
            "<strong>Tongue position precision:</strong> Tongue tip lightly touches the bottom of your upper front teeth OR rests between upper and lower teeth. Keep it relaxed and flat."
        ],
        "medium": [
            "<strong>Ensure it's truly voiceless.</strong> No vocal cord vibration - only air friction. Compare with ذ (Dhaal) which DOES vibrate.",
            "<strong>Tongue placement:</strong> Keep tongue tip visible between teeth or touching upper teeth. Don't retract it inside your mouth."
        ],
        "high": [
            "<strong>Perfect Thaa!</strong> Clean voiceless articulation. The airflow is clear and unvoiced."
        ]
    },
    
    "ذ": {
        "name": "Dhaal",
        "low": [
            "Dhaal (ذ) is the voiced 'th' as in 'this' or 'the'. <strong>Position:</strong> Same tongue position as Thaa (tongue tip at/between teeth), BUT your vocal cords MUST vibrate.",
            "<strong>Voiced component:</strong> While air passes through the tongue-teeth gap, your vocal cords vibrate simultaneously. Put your hand on your throat - you MUST feel strong buzzing/vibration.",
            "<strong>The key difference from Thaa:</strong> Thaa = voiceless (no throat buzz). Dhaal = voiced (throat buzzes). Same mouth position, different vocal cord activity.",
            "<strong>Practice technique:</strong> Start humming with your mouth closed. While humming, open mouth and place tongue at teeth position - that continuous vocal vibration is Dhaal."
        ],
        "medium": [
            "<strong>Strengthen the voicing.</strong> Your vocal cord vibration needs to be stronger and more continuous throughout the sound.",
            "<strong>Don't confuse with Zay (ز).</strong> Dhaal keeps tongue at TEETH; Zay has tongue at GUM RIDGE. Both are voiced but different positions."
        ],
        "high": [
            "<strong>Excellent voiced Dhaal!</strong> The vibration is strong and the articulation is precise."
        ]
    },
    
    "ر": {
        "name": "Raa",
        "low": [
            "Raa (ر) is a lightly rolled or tapped 'R'. <strong>Position:</strong> Tongue tip rises to the alveolar ridge (the bumpy area just behind your upper front teeth) and taps once or twice rapidly.",
            "<strong>The tap/trill:</strong> Your tongue tip should be relaxed and loose. It taps against the alveolar ridge 1-2 times as air flows past. Think of it like a very light drumbeat.",
            "<strong>Not English 'R':</strong> English 'R' keeps the tongue bunched and never touches the roof of the mouth. Arabic Raa MUST contact the alveolar ridge with a light tap.",
            "<strong>Practice:</strong> Say 'butter' or 'ladder' in American English - that quick tongue tap on the 't'/'dd' is similar to Arabic Raa. Now apply that tap to make 'r' sounds."
        ],
        "medium": [
            "<strong>Add a slight roll/tap.</strong> Your tongue tip needs to make brief contact with the alveolar ridge. It should bounce lightly.",
            "<strong>Avoid heavy rolling</strong> (like Spanish 'rr') but also avoid English 'R' (no contact). Arabic Raa is a light, quick tap - one or two beats."
        ],
        "high": [
            "<strong>Great Raa with proper roll!</strong> That light tapping quality is perfect. Keep the tongue relaxed."
        ]
    },
    
    "ش": {
        "name": "Sheen",
        "low": [
            "Sheen (ش) is like 'sh' in 'ship' - but ensure clarity.",
            "Place your tongue near the roof of your mouth and push air through.",
            "Don't confuse with س (Seen) - Sheen is broader and softer.",
            "Practice: Feel the wider air stream for 'sh' versus narrow 's'."
        ],
        "medium": [
            "Your Sheen is close! Ensure the sound is broad, not too sharp or hissing.",
            "Keep it light - avoid making it heavy even before emphatic letters."
        ],
        "high": [
            "Clear Sheen pronunciation! Well done."
        ]
    },
    
    "س": {
        "name": "Seen",
        "low": [
            "Seen (س) is a clear 's' sound - similar to English.",
            "Keep your tongue near your upper teeth ridge and create a hissing sound.",
            "Don't confuse with ص (Saad) which is emphatic and heavier.",
            "Practice: This should sound light and sharp, like a whistle."
        ],
        "medium": [
            "Your Seen is good, but keep it light and don't let it become emphatic.",
            "Ensure it's clearly different from ص (Saad) - no depth or heaviness."
        ],
        "high": [
            "Perfect light Seen! Crystal clear pronunciation."
        ]
    },
    
    # Short vowels are critical
    "َ": {  # Fatha
        "name": "Fatha",
        "low": [
            "Fatha (َ) is a short 'a' sound like in 'cat' or 'father'.",
            "Keep it short and crisp - don't elongate like alif (ا).",
            "Your mouth should be open, tongue flat.",
            "Practice: Quick 'ah' sound, not 'aaa'."
        ],
        "medium": [
            "Your Fatha needs to be shorter and clearer - avoid English 'uh' sound.",
            "Keep it pure 'a' without drifting to 'e' or 'o'."
        ],
        "high": [
            "Excellent short Fatha! Perfect vowel control."
        ]
    },
    
    # Common beginner letters
    "ا": {
        "name": "Alif",
        "low": [
            "Alif (ا) is a long 'aa' vowel sound - open and clear.",
            "Don't say 'A-leaf' - it's 'A-lif' with a short second syllable.",
            "Open your mouth wide and sustain the 'aa' sound.",
            "Practice: Think 'father' stretched out - 'faaaather'."
        ],
        "medium": [
            "Your Alif is good, but ensure it's consistently long - don't shorten it.",
            "Keep mouth position open and relaxed throughout."
        ],
        "high": [
            "Perfect long Alif! Good vowel length."
        ]
    },
    
    "ب": {
        "name": "Baa",
        "low": [
            "Baa (ب) is like English 'b' - press your lips together and release.",
            "Ensure clean pronunciation with proper lip closure.",
            "Practice: Feel your lips touch firmly before releasing the sound."
        ],
        "medium": [
            "Your Baa is nearly there - ensure complete lip closure for clarity."
        ],
        "high": [
            "Clean Baa pronunciation! Well done."
        ]
    },
    
    "ت": {
        "name": "Taa",
        "low": [
            "Taa (ت) is a light 't' sound - tongue tip touches upper gum ridge.",
            "Not emphatic like ط - keep it light and crisp.",
            "Practice: English 'tea' but keep it light and clear."
        ],
        "medium": [
            "Your Taa is good - just ensure it stays light, not heavy like ط.",
            "Keep tongue position precise at the gum ridge."
        ],
        "high": [
            "Perfect light Taa! Clear articulation."
        ]
    },
    
    "ج": {
        "name": "Jeem",
        "low": [
            "Jeem (ج) is like English 'j' in 'jump'.",
            "Press your tongue to the roof of your mouth and release with sound.",
            "Keep it voiced - vocal cords should vibrate."
        ],
        "medium": [
            "Your Jeem is close! Ensure it's clearly a 'j' sound, not too soft."
        ],
        "high": [
            "Excellent Jeem! Clear and strong."
        ]
    },
    
    "د": {
        "name": "Daal",
        "low": [
            "Daal (د) is a light 'd' sound - tongue tip at upper teeth/gums.",
            "Not emphatic like ض - keep it light and quick.",
            "Practice: English 'dog' - quick and clear."
        ],
        "medium": [
            "Your Daal is good! Keep it light without any emphatic quality.",
            "Precise tongue position at the gum line."
        ],
        "high": [
            "Perfect light Daal! Clean pronunciation."
        ]
    },
    
    "ر": {
        "name": "Raa",
        "low": [
            "Raa (ر) requires a slight tongue roll - like Spanish or Italian 'r'.",
            "Tap your tongue against the roof of your mouth once or twice.",
            "Don't over-roll it or make it English 'r' - keep it light and tapped.",
            "Practice: Think 'butter' in American English - that quick tap."
        ],
        "medium": [
            "Your Raa needs a bit more roll. Tap your tongue lightly against the palate.",
            "Find the balance - not flat like English 'r', not heavy rolled."
        ],
        "high": [
            "Great Raa with proper roll! Keep that lightness."
        ]
    },
    
    "ز": {
        "name": "Zay",
        "low": [
            "Zay (ز) is like English 'z' in 'zoo' - voiced and buzzing.",
            "Keep your tongue at the gum ridge and vibrate vocal cords.",
            "Not emphatic like ظ - should be light and clear."
        ],
        "medium": [
            "Your Zay is nearly perfect! Keep it consistently voiced.",
            "Ensure it's clearly 'z', not 's' or the 'th' sounds."
        ],
        "high": [
            "Perfect Zay! Clear voiced pronunciation."
        ]
    },
    
    "ف": {
        "name": "Faa",
        "low": [
            "Faa (ف) is like English 'f' - upper teeth touch lower lip.",
            "Blow air through the gap between teeth and lip.",
            "Keep it light and clear, similar to 'fan' or 'far'."
        ],
        "medium": [
            "Your Faa is good! Ensure consistent airflow and teeth-lip contact."
        ],
        "high": [
            "Perfect Faa! Clear fricative sound."
        ]
    },
    
    "ك": {
        "name": "Kaaf",
        "low": [
            "Kaaf (ك) is like English 'k' in 'kite' - light and from front of mouth.",
            "Not deep like ق (Qaf) - tongue touches the soft palate lightly.",
            "Practice: 'King' in English - that light, clear 'k'."
        ],
        "medium": [
            "Your Kaaf is close! Keep it forward in your mouth, not deep like Qaf.",
            "Ensure it's clearly lighter than ق."
        ],
        "high": [
            "Excellent light Kaaf! Clear distinction from Qaf."
        ]
    },
    
    "ل": {
        "name": "Laam",
        "low": [
            "Laam (ل) is like English 'l' - tongue tip touches upper gum ridge.",
            "Let the sound flow around the sides of your tongue.",
            "Keep it clear and light, as in 'lamp' or 'love'."
        ],
        "medium": [
            "Your Laam is good! Ensure proper tongue contact for clarity."
        ],
        "high": [
            "Perfect Laam! Clear lateral sound."
        ]
    },
    
    "م": {
        "name": "Meem",
        "low": [
            "Meem (م) is like English 'm' - close your lips completely.",
            "Sound should come through your nose while lips are closed.",
            "Practice: 'Mama' - feel the nasal resonance."
        ],
        "medium": [
            "Your Meem is nearly there! Ensure complete lip closure.",
            "Let the sound resonate in your nose."
        ],
        "high": [
            "Perfect Meem! Clear nasal sound."
        ]
    },
    
    "ن": {
        "name": "Noon",
        "low": [
            "Noon (ن) is like English 'n' - tongue tip at upper gum ridge.",
            "Sound should come through your nose, similar to Meem.",
            "Practice: 'Nine' - clear nasal sound with tongue up."
        ],
        "medium": [
            "Your Noon is good! Keep the nasal quality consistent."
        ],
        "high": [
            "Excellent Noon! Clear nasal pronunciation."
        ]
    },
    
    "ه": {
        "name": "Haa (light)",
        "low": [
            "Light Haa (ه) is like English 'h' in 'hat' - soft and breathy.",
            "Much lighter than ح (Haa from throat) - barely any throat constriction.",
            "Sound comes from deep in throat but with light airflow."
        ],
        "medium": [
            "Your light Haa is close! Keep it softer than throat Haa (ح).",
            "Minimal effort - just a whisper of breath."
        ],
        "high": [
            "Perfect light Haa! Nice and soft."
        ]
    },
    
    "و": {
        "name": "Waaw",
        "low": [
            "Waaw (و) is like English 'w' in 'water' - round your lips.",
            "Can be a consonant 'w' or long vowel 'oo' depending on context.",
            "Practice: Round lips as if saying 'woo'."
        ],
        "medium": [
            "Your Waaw is good! Ensure clear lip rounding.",
            "Keep the 'w' sound clean and the 'oo' vowel pure."
        ],
        "high": [
            "Excellent Waaw! Clear pronunciation."
        ]
    },
    
    "ي": {
        "name": "Yaa",
        "low": [
            "Yaa (ي) is like English 'y' in 'yes' or long 'ee' vowel.",
            "Raise middle of tongue toward roof of mouth.",
            "Can be consonant 'y' or long vowel 'ee' depending on position."
        ],
        "medium": [
            "Your Yaa is close! Ensure proper tongue position for both 'y' and 'ee' sounds.",
            "Keep vowel length consistent when it's a long vowel."
        ],
        "high": [
            "Perfect Yaa! Clear articulation in both forms."
        ]
    }
}

def get_pronunciation_hints(letter: str, probability: float) -> List[str]:
    """
    Return pronunciation hints based on the predicted letter and confidence.
    
    Args:
        letter: The predicted letter (could be Arabic character or English name like "Alif")
        probability: Confidence score (0-1)
    
    Returns:
        List of helpful pronunciation hints
    """
    # Convert English name to Arabic character if needed
    if letter in LETTER_NAME_TO_ARABIC:
        letter = LETTER_NAME_TO_ARABIC[letter]
    
    if letter not in PRONUNCIATION_HINTS:
        return []
    
    hints_data = PRONUNCIATION_HINTS[letter]
    
    # Determine hint level based on probability
    if probability < 0.55:  # Very low confidence (< 55%)
        return hints_data.get("low", [])
    elif probability < 0.85:  # Medium confidence (55-85%)
        return hints_data.get("medium", [])
    else:  # High confidence (85%+)
        return hints_data.get("high", [])


# --------------------------- Repo paths & model discovery ---------------------------

def resolve_repo_paths(script_file: Path) -> Dict[str, Path]:
    code_dir = script_file.resolve().parent
    root = code_dir.parent
    return {
        "root": root,
        "code": code_dir,
        "dataset": root / "Dataset",
        "models": root / "Models",
        "results": root / "Results",
    }

def preferred_model_dir(paths: Dict[str, Path]) -> Path:
    res = paths["results"] / "facebook__wav2vec2-base" / "model"
    mod = paths["models"]  / "facebook__wav2vec2-base"
    if (res / "config.json").exists():
        return res
    if (mod / "config.json").exists():
        return mod
    raise FileNotFoundError(
        f"Could not find the fine-tuned wav2vec2-base model.\n"
        f"Looked in:\n  {res}\n  {mod}\n"
        "Make sure finetune_eval.py finished for facebook/wav2vec2-base."
    )

def optional_confusion_paths(paths: Dict[str, Path]) -> Tuple[Optional[Path], Optional[Path]]:
    png = paths["results"] / "facebook__wav2vec2-base" / "confusion_matrix.png"
    csv = paths["results"] / "facebook__wav2vec2-base" / "per_class_report.csv"
    return (png if png.exists() else None, csv if csv.exists() else None)


# --------------------------- Audio helpers ---------------------------

def to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32, copy=False)

def resample_if_needed(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y
    if HAVE_RESAMPY:
        return resampy.resample(y, sr_in, sr_out).astype(np.float32, copy=False)
    # Linear resample fallback
    duration = len(y) / float(sr_in)
    new_len = max(1, int(round(duration * sr_out)))
    x_old = np.linspace(0, len(y), num=len(y), endpoint=False)
    x_new = np.linspace(0, len(y), num=new_len, endpoint=False)
    return np.interp(x_new, x_old, y).astype(np.float32, copy=False)

def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) >= target_len:
        return y[:target_len]
    pad = target_len - len(y)
    return np.pad(y, (0, pad), mode="constant")


# --------------------------- Plot helpers (return base64 PNG) ---------------------------

def fig_to_base64_png(fig, dpi: int = 160) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    b = base64.b64encode(buf.getvalue()).decode("ascii")
    buf.close()
    return f"data:image/png;base64,{b}"

def plot_waveform_b64(y: np.ndarray, sr: int) -> str:
    t = np.arange(len(y)) / float(sr)
    fig = plt.figure(figsize=(8, 1.8))
    ax = fig.add_subplot(111)
    ax.plot(t, y, linewidth=1.0, color='#3b82f6')
    ax.set_xlabel("Seconds", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.set_title("Recorded waveform", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig_to_base64_png(fig)

def plot_prob_bars_b64(labels: List[str], probs: np.ndarray, top: int = 10) -> str:
    idxs = np.argsort(-probs)[:top]
    top_labels = [labels[i] for i in idxs]
    top_probs = probs[idxs]
    
    # Create color gradient based on probability
    colors = ['#22c55e' if p > 0.7 else '#3b82f6' if p > 0.4 else '#ef4444' for p in top_probs]
    
    fig = plt.figure(figsize=(8, 3.5))
    ax = fig.add_subplot(111)
    bars = ax.bar(top_labels, top_probs, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability", fontsize=10, fontweight='bold')
    ax.set_title(f"Top-{top} Predicted Letters", fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.2)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.tight_layout()
    return fig_to_base64_png(fig)


# --------------------------- Model wrapper ---------------------------

class Wav2Vec2Classifier:
    def __init__(self, model_dir: Path, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        self.model = AutoModelForAudioClassification.from_pretrained(str(model_dir))
        self.model.to(self.device).eval()

        cfg = self.model.config
        if getattr(cfg, "id2label", None):
            self.id2label = {int(k): v for k, v in cfg.id2label.items()}
        else:
            summary = model_dir.parent / "summary.json"
            if summary.exists():
                with open(summary, "r", encoding="utf-8") as f:
                    labels = json.load(f).get("labels", [])
                self.id2label = {i: lab for i, lab in enumerate(labels)}
            else:
                self.id2label = {i: f"CLASS_{i}" for i in range(cfg.num_labels)}

        sr = getattr(self.processor, "sampling_rate", None)
        if sr is None and hasattr(self.processor, "feature_extractor"):
            sr = getattr(self.processor.feature_extractor, "sampling_rate", None)
        self.sampling_rate = int(sr) if sr else 16000

        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.size_mb = round(sum(p.element_size() * p.nelement() for p in self.model.parameters()) / (1024**2), 2)

    @torch.no_grad()
    def infer_numpy(
        self,
        y: np.ndarray,
        sr_in: int,
        top_k: int = 5,
        fixed_seconds: float = 1.0
    ) -> Dict:
        y = to_mono(y)
        y = resample_if_needed(y, sr_in, self.sampling_rate)
        max_len = int(round(fixed_seconds * self.sampling_rate))
        y = pad_or_trim(y, max_len)

        t0 = time.perf_counter()
        inputs = self.processor(
            [y],
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        t1 = time.perf_counter()

        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        top_k = max(1, min(top_k, probs.shape[-1]))
        idxs = np.argsort(-probs)[:top_k]
        labels = [self.id2label[int(i)] for i in idxs]
        scores = [float(probs[i]) for i in idxs]

        return {
            "top1_label": labels[0],
            "top1_prob": scores[0],
            "topk": list(zip(labels, scores)),
            "probs": probs,
            "latency_ms": (t1 - t0) * 1000.0,
            "sr_used": self.sampling_rate,
            "processed_audio": y,  # return for plotting
        }


# --------------------------- App state ---------------------------

app = FastAPI(title="Arabic Alphabet Realtime (wav2vec2-base)", version="1.0.0")
_MODEL: Optional[Wav2Vec2Classifier] = None
_CONF_PNG: Optional[Path] = None
_CONF_CSV: Optional[Path] = None
_PATHS: Optional[Dict[str, Path]] = None

def _load_once():
    global _MODEL, _CONF_PNG, _CONF_CSV, _PATHS
    if _MODEL is None:
        _PATHS = resolve_repo_paths(Path(__file__))
        mdir = preferred_model_dir(_PATHS)
        _MODEL = Wav2Vec2Classifier(mdir, device="auto")
        _CONF_PNG, _CONF_CSV = optional_confusion_paths(_PATHS)
    return _MODEL


# --------------------------- Routes: static training artifacts ---------------------------

@app.get("/confusion", response_class=Response)
def get_confusion():
    _load_once()
    if _CONF_PNG is None:
        return Response(status_code=204)
    return FileResponse(str(_CONF_PNG))

@app.get("/per_class_report", response_class=Response)
def get_per_class_report():
    _load_once()
    if _CONF_CSV is None:
        return Response(status_code=204)
    return FileResponse(str(_CONF_CSV), media_type="text/csv", filename="per_class_report.csv")


# --------------------------- Main page (HTML + JS) ---------------------------

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Arabic Alphabet Pronunciation Trainer</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #0b0f19; --panel:#121828; --border:#1d2742; --muted:#9aa4b2; --text:#e6e9ef;
    --primary:#3b82f6; --primary-600:#2563eb; --danger:#ef4444; --accent:#22c55e; 
    --card:#0f1525; --warning:#f59e0b; --success:#10b981;
  }
  body { margin:0; font-family: 'Segoe UI', system-ui, -apple-system, Roboto, Ubuntu, Cantarell;
         background: var(--bg); color: var(--text); line-height:1.6; }
  
  header { padding: 20px 24px; border-bottom: 1px solid var(--border); 
           background: linear-gradient(135deg, var(--panel) 0%, #0d1219 100%); 
           box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
  header h2 { margin:0; font-weight:700; letter-spacing:.3px; font-size: 24px;
              background: linear-gradient(135deg, #3b82f6, #22c55e);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent;
              background-clip: text; }
  
  main { padding: 24px; max-width: 1400px; margin: 0 auto; }
  .row { display:flex; gap:22px; flex-wrap:wrap; margin-bottom:22px; }
  .col { flex:1 1 380px; }
  
  .card { background: var(--panel); border:1px solid var(--border); border-radius:16px; 
          padding:20px; box-shadow: 0 12px 40px rgba(0,0,0,.3); 
          transition: transform 0.2s, box-shadow 0.2s; }
  .card:hover { transform: translateY(-2px); box-shadow: 0 16px 50px rgba(0,0,0,.4); }
  .card h3 { margin: 0 0 16px 0; font-size: 19px; font-weight:600; 
             color: var(--primary); display:flex; align-items:center; gap:8px; }
  .card h3::before { content:''; width:4px; height:20px; 
                     background: linear-gradient(180deg, var(--primary), var(--accent)); 
                     border-radius:2px; }
  
  .btn { padding:11px 18px; border-radius:10px; border:1px solid var(--border); 
         cursor:pointer; margin:4px 8px 4px 0; background:#0f1525; color:var(--text);
         font-size:14px; font-weight:500; transition: all 0.2s; display:inline-flex;
         align-items:center; gap:6px; }
  .btn:hover { border-color:#2a355a; transform: translateY(-1px); }
  .btn:active { transform: translateY(0); }
  .btn:disabled { opacity:0.5; cursor:not-allowed; transform:none; }
  .btn-primary { background: linear-gradient(135deg, var(--primary), var(--primary-600)); 
                 border-color: var(--primary-600); color: #fff; }
  .btn-primary:hover { box-shadow: 0 4px 12px rgba(59,130,246,0.4); }
  .btn-danger  { background: linear-gradient(135deg, var(--danger), #b91c1c); 
                 border-color: #b91c1c; color: #fff; }
  .btn-danger:hover { box-shadow: 0 4px 12px rgba(239,68,68,0.4); }
  .btn-ghost   { background: transparent; color: var(--muted); border-color:var(--border); }
  
  .muted { color: var(--muted); font-size:14px; }
  .kv { display:grid; grid-template-columns: 150px auto; gap:8px 14px; font-size:14px; }
  .kv strong { font-size: 18px; color:var(--accent); }
  
  .meter { position:relative; width:100%; height:12px; background:#0b1223; 
           border-radius:8px; overflow:hidden; border:1px solid var(--border);
           box-shadow: inset 0 2px 4px rgba(0,0,0,0.3); }
  .meter > span { position:absolute; left:0; top:0; height:100%; width:0%; 
                  background: linear-gradient(90deg, #22c55e 0%, #f59e0b 50%, #ef4444 100%); 
                  transition: width .08s ease-out; box-shadow: 0 0 10px rgba(34,197,94,0.5); }
  
  audio { width:100%; margin-top:12px; }
  input[type="range"] { width:100%; accent-color: var(--primary); }
  input[type="file"] { padding:8px; background:var(--card); border:1px solid var(--border);
                       border-radius:8px; color:var(--text); cursor:pointer; }
  
  table { width:100%; border-collapse: collapse; font-size:14px; }
  th, td { padding: 10px 12px; border-bottom:1px solid var(--border); text-align:left; }
  th { color: var(--muted); font-weight:600; text-transform:uppercase; font-size:12px;
       letter-spacing:0.5px; background:rgba(59,130,246,0.05); }
  tr:hover { background:rgba(59,130,246,0.05); }
  
  .imgbox { display:flex; align-items:center; justify-content:center; min-height:160px; 
            border:2px dashed var(--border); border-radius:12px; background: var(--card); 
            padding:12px; }
  .imgbox img { max-width:100%; height:auto; border-radius:8px; }
  
  a { color: var(--primary); text-decoration: none; transition: color 0.2s; }
  a:hover { color: var(--accent); text-decoration: underline; }
  
  .hints-section { background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(34,197,94,0.08));
                   border: 2px solid var(--border); border-radius:12px; padding:18px;
                   margin-top:18px; }
  .hints-section h4 { margin:0 0 12px 0; color:var(--accent); font-size:17px;
                      display:flex; align-items:center; gap:8px; }
  .hints-section h4::before { content:'💡'; font-size:22px; }
  
  .hint-item { background:var(--panel); padding:12px 16px; border-radius:8px; 
               margin-bottom:10px; border-left:4px solid var(--primary);
               box-shadow: 0 2px 8px rgba(0,0,0,0.2); line-height:1.7; }
  .hint-item:last-child { margin-bottom:0; }
  
  .accuracy-badge { display:inline-block; padding:6px 14px; border-radius:20px;
                    font-weight:600; font-size:14px; margin-left:10px; }
  .accuracy-low { background:rgba(239,68,68,0.2); color:#ef4444; border:1px solid #ef4444; }
  .accuracy-medium { background:rgba(245,158,11,0.2); color:#f59e0b; border:1px solid #f59e0b; }
  .accuracy-high { background:rgba(34,197,94,0.2); color:#22c55e; border:1px solid #22c55e; }
  
  .status-indicator { display:inline-flex; align-items:center; gap:8px; padding:6px 12px;
                      border-radius:8px; font-size:13px; font-weight:500; }
  .status-idle { background:rgba(156,163,175,0.15); color:var(--muted); }
  .status-recording { background:rgba(239,68,68,0.15); color:#ef4444; 
                      animation: pulse 2s infinite; }
  .status-captured { background:rgba(34,197,94,0.15); color:#22c55e; }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }
  
  .stat-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
               gap:12px; margin-top:12px; }
  .stat-card { background:var(--card); padding:14px; border-radius:10px; 
               border:1px solid var(--border); text-align:center; }
  .stat-card .label { font-size:12px; color:var(--muted); text-transform:uppercase;
                      letter-spacing:0.5px; margin-bottom:6px; }
  .stat-card .value { font-size:22px; font-weight:700; color:var(--accent); }
</style>
</head>
<body>
<header>
  <h2>🔤 Arabic Alphabet Pronunciation Trainer</h2>
  <div class="muted" style="margin-top:8px">Powered by wav2vec2-base • Real-time feedback with AI-driven pronunciation hints</div>
</header>

<main>
  <div class="row">
    <div class="col">
      <div class="card">
        <h3>🎙️ Record or Upload</h3>
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:12px;">
          <button id="btnStart" class="btn btn-primary">▶ Start Recording</button>
          <button id="btnStop"  class="btn btn-danger" disabled>⏹ Stop</button>
          <button id="btnReset" class="btn btn-ghost" disabled>↺ Reset</button>
        </div>
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px; flex-wrap:wrap;">
          <span class="muted">Status:</span>
          <span id="recState" class="status-indicator status-idle">⚪ Idle</span>
          <span class="muted" style="margin-left:8px;">Duration:</span>
          <span id="timer" style="font-weight:600; color:var(--accent);">0.0s</span>
        </div>
        <div style="margin-bottom:12px">
          <div class="meter"><span id="level" style="width:0%"></span></div>
        </div>
        <div style="margin-bottom:12px">
          <input type="file" id="upload" accept="audio/wav,audio/*" />
        </div>
        <audio id="preview" controls></audio>
      </div>
    </div>

    <div class="col">
      <div class="card">
        <h3>⚙️ Settings</h3>
        <div style="margin-bottom:16px">
          <label for="topk" class="muted" style="display:block; margin-bottom:6px;">
            Top-K Predictions
          </label>
          <input id="topk" type="range" min="1" max="28" step="1" value="5" />
          <div style="margin-top:4px;"><span class="muted">Value: </span>
            <span id="topkVal" style="font-weight:600; color:var(--primary);">5</span>
          </div>
        </div>
        <div style="margin-bottom:18px">
          <label for="fixed" class="muted" style="display:block; margin-bottom:6px;">
            Recording Window (seconds)
          </label>
          <input id="fixed" type="range" min="0.6" max="2.5" step="0.1" value="1.0" />
          <div style="margin-top:4px;"><span class="muted">Value: </span>
            <span id="fixedVal" style="font-weight:600; color:var(--primary);">1.0</span>
          </div>
        </div>
        <div>
          <button id="btnClassify" class="btn btn-primary" disabled>🚀 Analyze Pronunciation</button>
        </div>
      </div>
    </div>

    <div class="col">
      <div class="card">
        <h3>🎯 Top Prediction</h3>
        <div class="kv">
          <div class="muted">Letter</div>
          <div><strong id="top1Label" style="font-size:32px;">—</strong></div>
          <div class="muted">Confidence</div>
          <div><strong id="top1Prob">—</strong><span id="accuracyBadge"></span></div>
          <div class="muted">Processing Time</div>
          <div><strong id="latency">—</strong></div>
        </div>
        <div class="stat-grid" style="margin-top:18px;">
          <div class="stat-card">
            <div class="label">Device</div>
            <div class="value" id="deviceType" style="font-size:16px;">—</div>
          </div>
          <div class="stat-card">
            <div class="label">Sample Rate</div>
            <div class="value" id="sampleRate" style="font-size:16px;">—</div>
          </div>
        </div>
      </div>
      
      <!-- HINTS SECTION - Always visible, simple and functional -->
      <div class="card" style="margin-top:22px; min-height:150px;">
        <h3>💡 Pronunciation Tips</h3>
        <div id="hintsContent" style="color: var(--muted); font-style: italic;">
          Analyze your pronunciation to see helpful tips here...
        </div>
      </div>
    </div>
  </div>

  <div class="row">
    <div class="col">
      <div class="card">
        <h3>📊 Probability Distribution</h3>
        <div class="imgbox"><img id="probImg" style="max-width:100%"/></div>
      </div>
    </div>
    <div class="col">
      <div class="card">
        <h3>🌊 Audio Waveform</h3>
        <div class="imgbox"><img id="waveImg" style="max-width:100%"/></div>
      </div>
    </div>
  </div>

  <div class="row">
    <div class="col">
      <div class="card">
        <h3>📋 Detailed Predictions</h3>
        <table id="topkTable">
          <thead><tr>
            <th style="text-align:right; width:60px;">Rank</th>
            <th>Letter</th>
            <th>Confidence</th>
            <th style="width:150px;">Bar</th>
          </tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>

    <div class="col">
      <div class="card">
        <h3>📈 Training Diagnostics</h3>
        <div><img id="confImg" style="max-width:100%; display:none; border-radius:8px;"/></div>
        <div style="margin-top:12px">
          <a id="reportLink" class="muted" href="#" style="display:none" download="per_class_report.csv">
            📥 Download per-class report
          </a>
        </div>
        <div id="diagNote" class="muted" style="margin-top:12px"></div>
        <div id="modelInfo" class="muted" style="margin-top:16px; padding:12px; background:var(--card);
             border-radius:8px; border:1px solid var(--border); line-height:1.8;">—</div>
      </div>
    </div>
  </div>
</main>

<script>
// --- Simple WAV encoder using WebAudio (16-bit PCM) ---
class WavRecorder {
  constructor() {
    this.chunks = [];
    this.sampleRate = 44100;
    this.recording = false;
    this._audioCtx = null;
    this._processor = null;
    this._source = null;
    this._stream = null;
    this.onlevel = null;
  }

  async start() {
    if (this.recording) return;
    this._stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this._audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    this.sampleRate = this._audioCtx.sampleRate;
    const source = this._audioCtx.createMediaStreamSource(this._stream);
    const processor = this._audioCtx.createScriptProcessor(4096, 1, 1);
    this._source = source;
    this._processor = processor;
    this.chunks = [];
    processor.onaudioprocess = (e) => {
      if (!this.recording) return;
      const input = e.inputBuffer.getChannelData(0);
      this.chunks.push(new Float32Array(input));
      if (this.onlevel) {
        let sum = 0;
        for (let i=0;i<input.length;i++) { const v=input[i]; sum += v*v; }
        const rms = Math.sqrt(sum / input.length);
        this.onlevel(Math.min(1, rms*1.6));
      }
    };
    source.connect(processor);
    processor.connect(this._audioCtx.destination);
    this.recording = true;
  }

  async stop() {
    if (!this.recording) return null;
    this.recording = false;
    try { this._processor.disconnect(); } catch(_){}
    try { this._source.disconnect(); } catch(_){}
    try { this._stream.getTracks().forEach(t => t.stop()); } catch(_){}
    try { await this._audioCtx.close(); } catch(_){}

    const length = this.chunks.reduce((sum, arr) => sum + arr.length, 0);
    const pcm = new Float32Array(length);
    let offset = 0;
    for (const arr of this.chunks) { pcm.set(arr, offset); offset += arr.length; }
    return this._encodeWav(pcm, this.sampleRate);
  }

  _encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], {type: 'audio/wav'});
  }
}

// --- UI Logic ---
const btnStart   = document.getElementById('btnStart');
const btnStop    = document.getElementById('btnStop');
const btnReset   = document.getElementById('btnReset');
const btnClassify= document.getElementById('btnClassify');
const timer      = document.getElementById('timer');
const upload     = document.getElementById('upload');
const preview    = document.getElementById('preview');
const topk       = document.getElementById('topk');
const topkVal    = document.getElementById('topkVal');
const fixed      = document.getElementById('fixed');
const fixedVal   = document.getElementById('fixedVal');

const top1Label  = document.getElementById('top1Label');
const top1Prob   = document.getElementById('top1Prob');
const latency    = document.getElementById('latency');
const modelInfo  = document.getElementById('modelInfo');
const probImg    = document.getElementById('probImg');
const waveImg    = document.getElementById('waveImg');
const topkTable  = document.getElementById('topkTable').querySelector('tbody');
const confImg    = document.getElementById('confImg');
const reportLink = document.getElementById('reportLink');
const diagNote   = document.getElementById('diagNote');
const level      = document.getElementById('level');
const recState   = document.getElementById('recState');
const accuracyBadge = document.getElementById('accuracyBadge');
const deviceType = document.getElementById('deviceType');
const sampleRate = document.getElementById('sampleRate');
const hintsContent = document.getElementById('hintsContent');

let recorder = null;
let currentBlob = null;
let timerId = null;
let autoStopId = null;
let t0 = 0;

function stopTimers(){
  if (timerId) clearInterval(timerId);
  if (autoStopId) clearTimeout(autoStopId);
  timerId = null; autoStopId = null;
}

function updateStatusIndicator(status) {
  recState.className = 'status-indicator';
  if (status === 'idle') {
    recState.classList.add('status-idle');
    recState.innerHTML = '⚪ Idle';
  } else if (status === 'recording') {
    recState.classList.add('status-recording');
    recState.innerHTML = '🔴 Recording';
  } else if (status === 'captured') {
    recState.classList.add('status-captured');
    recState.innerHTML = '🟢 Ready';
  }
}

topk.addEventListener('input', () => topkVal.textContent = topk.value);
fixed.addEventListener('input', () => fixedVal.textContent = fixed.value);

btnStart.addEventListener('click', async () => {
  btnStart.disabled = true; btnStop.disabled = false; btnClassify.disabled = true; btnReset.disabled = false;
  upload.value = '';
  currentBlob = null;
  updateStatusIndicator('recording');
  level.style.width = '0%';

  recorder = new WavRecorder();
  recorder.onlevel = (rms) => { level.style.width = Math.round(Math.min(100, rms*100)) + '%'; };
  await recorder.start();

  t0 = performance.now();
  timer.textContent = '0.0s';
  timerId = setInterval(() => {
    const dt = (performance.now() - t0) / 1000.0;
    timer.textContent = dt.toFixed(1) + 's';
  }, 100);

  const windowSec = parseFloat(fixed.value || '1.0');
  autoStopId = setTimeout(async ()=>{
    if (recorder && recorder.recording) {
      const blob = await recorder.stop();
      window.afterAutoStop(blob);
    }
  }, Math.max(100, windowSec * 1000));
});

btnStop.addEventListener('click', async () => {
  btnStop.disabled = true;
  if (recorder && recorder.recording) {
    const blob = await recorder.stop();
    currentBlob = blob;
    preview.src = URL.createObjectURL(blob);
    preview.play();
  }
  stopTimers();
  level.style.width = '0%';
  updateStatusIndicator('captured');
  btnStart.disabled = false;
  btnClassify.disabled = !currentBlob ? true : false;
  recorder = null;
});

window.afterAutoStop = async (blob)=>{
  currentBlob = blob;
  preview.src = URL.createObjectURL(blob);
  preview.play();

  stopTimers();
  level.style.width = '0%';
  btnStart.disabled = false;
  btnStop.disabled = true;
  btnClassify.disabled = false;
  updateStatusIndicator('captured');
  recorder = null;
};

btnReset.addEventListener('click', async ()=>{
  try { if (recorder && recorder.recording) { await recorder.stop(); } } catch(_){}
  stopTimers();
  level.style.width = '0%';

  currentBlob = null;
  preview.src = '';
  probImg.src = '';
  waveImg.src = '';
  topkTable.innerHTML = '';
  top1Label.textContent = '—';
  top1Prob.textContent  = '—';
  latency.textContent   = '—';
  accuracyBadge.innerHTML = '';
  hintsContent.innerHTML = '<span style="color: var(--muted); font-style: italic;">Analyze your pronunciation to see helpful tips here...</span>';
  updateStatusIndicator('idle');
  timer.textContent     = '0.0s';

  btnStart.disabled    = false;
  btnStop.disabled     = true;
  btnClassify.disabled = true;
  btnReset.disabled    = true;
  recorder = null;
});

upload.addEventListener('change', () => {
  if (upload.files && upload.files[0]) {
    currentBlob = upload.files[0];
    preview.src = URL.createObjectURL(currentBlob);
    preview.play();
    btnClassify.disabled = false;
    btnReset.disabled = false;
    stopTimers();
    level.style.width = '0%';
    btnStart.disabled = false;
    btnStop.disabled = true;
    updateStatusIndicator('captured');
  }
});

btnClassify.addEventListener('click', async () => {
  if (!currentBlob) return;
  btnClassify.disabled = true;
  btnClassify.innerHTML = '⏳ Analyzing...';
  
  try {
    const fd = new FormData();
    fd.append('audio', currentBlob, 'input.wav');
    fd.append('top_k', topk.value);
    fd.append('fixed_seconds', fixed.value);

    const res = await fetch('/infer', { method: 'POST', body: fd });
    if (!res.ok) {
      const text = await res.text();
      alert('Request failed: ' + text);
      btnClassify.disabled = false;
      btnClassify.innerHTML = '🚀 Analyze Pronunciation';
      return;
    }
    const data = await res.json();

    // Update main prediction
    top1Label.textContent = data.top1_label;
    const prob = data.top1_prob;
    top1Prob.textContent = (prob * 100).toFixed(1) + '%';
    latency.textContent = data.latency_ms.toFixed(1) + ' ms';
    
    // Accuracy badge
    let badgeClass = 'accuracy-low';
    let badgeText = 'Needs Practice';
    if (prob >= 0.85) {
      badgeClass = 'accuracy-high';
      badgeText = 'Excellent!';
    } else if (prob >= 0.55) {
      badgeClass = 'accuracy-medium';
      badgeText = 'Good';
    }
    accuracyBadge.innerHTML = `<span class="accuracy-badge ${badgeClass}">${badgeText}</span>`;
    
    // Model info
    modelInfo.innerHTML = data.model_info_html;
    deviceType.textContent = data.device || 'CPU';
    sampleRate.textContent = data.sr_used ? (data.sr_used / 1000).toFixed(1) + ' kHz' : '—';
    
    // Visuals
    probImg.src = data.prob_b64;
    waveImg.src = data.wave_b64;

    // Top-K table with progress bars
    topkTable.innerHTML = '';
    data.topk.forEach((row, idx) => {
      const tr = document.createElement('tr');
      const confidence = (row[1] * 100).toFixed(1);
      const barWidth = Math.round(row[1] * 100);
      const barColor = row[1] > 0.7 ? '#22c55e' : row[1] > 0.4 ? '#3b82f6' : '#ef4444';
      tr.innerHTML = `
        <td style="text-align:right; font-weight:600; color:var(--muted);">${idx+1}</td>
        <td style="font-size:18px; font-weight:600;">${row[0]}</td>
        <td style="font-weight:600;">${confidence}%</td>
        <td><div style="background:#0b1223; border-radius:6px; height:20px; width:100%; position:relative; border:1px solid var(--border);">
          <div style="background:${barColor}; height:100%; width:${barWidth}%; border-radius:5px; transition:width 0.3s;"></div>
        </div></td>
      `;
      topkTable.appendChild(tr);
    });

    // Pronunciation hints - SIMPLE AND FUNCTIONAL
    hintsContent.innerHTML = '';
    if (data.hints && data.hints.length > 0) {
      data.hints.forEach((hint, index) => {
        const hintDiv = document.createElement('div');
        hintDiv.style.cssText = 'background:var(--card); padding:12px; margin-bottom:10px; border-radius:8px; border-left:3px solid var(--accent);';
        hintDiv.innerHTML = `<strong style="color:var(--accent);">Tip ${index + 1}:</strong> ${hint}`;
        hintsContent.appendChild(hintDiv);
      });
    } else {
      hintsContent.innerHTML = '<div style="padding:12px; color:var(--muted);">Great job! Keep practicing for consistency.</div>';
    }

    // Diagnostics
    if (data.confusion_available) {
      confImg.style.display = 'block';
      confImg.src = '/confusion';
      reportLink.style.display = data.per_class_available ? 'inline-block' : 'none';
      reportLink.href = '/per_class_report';
      diagNote.textContent = '';
    } else {
      confImg.style.display = 'none';
      reportLink.style.display = 'none';
      diagNote.textContent = 'Training diagnostics not available.';
    }
  } catch (err) {
    alert('Error: ' + err);
  } finally {
    btnClassify.disabled = false;
    btnClassify.innerHTML = '🚀 Analyze Pronunciation';
  }
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    _load_once()
    return HTMLResponse(INDEX_HTML)


# --------------------------- Inference endpoint ---------------------------

@app.post("/infer", response_class=JSONResponse)
async def infer(audio: UploadFile = File(...), top_k: str = "5", fixed_seconds: str = "1.0"):
    model = _load_once()

    # Read WAV
    content = await audio.read()
    try:
        y, sr = sf.read(io.BytesIO(content), dtype="float32", always_2d=False)
    except Exception as e:
        return PlainTextResponse(f"Could not read audio (expect WAV). Error: {type(e).__name__}: {e}", status_code=400)

    if y.ndim > 1:
        y = y.mean(axis=1)

    try:
        tk = int(top_k)
    except:
        tk = 5
    try:
        fx = float(fixed_seconds)
    except:
        fx = 1.0

    out = model.infer_numpy(y, sr_in=sr, top_k=tk, fixed_seconds=fx)

    # Get pronunciation hints based on prediction
    top_letter = out["top1_label"]
    top_prob = out["top1_prob"]
    hints = get_pronunciation_hints(top_letter, top_prob)

    # Visuals
    wave_b64 = plot_waveform_b64(y, sr)
    labels_all = [model.id2label[i] for i in range(len(model.id2label))]
    prob_b64 = plot_prob_bars_b64(labels_all, np.asarray(out["probs"]), top=min(tk, len(labels_all)))

    # Stats
    info_html = (
        f"<div style='margin-bottom:6px;'><b>Model:</b> facebook/wav2vec2-base</div>"
        f"<div style='margin-bottom:6px;'><b>Device:</b> {model.device.type.upper()}</div>"
        f"<div style='margin-bottom:6px;'><b>Sampling Rate:</b> {model.sampling_rate} Hz</div>"
        f"<div style='margin-bottom:6px;'><b>Classes:</b> {len(model.id2label)}</div>"
        f"<div style='margin-bottom:6px;'><b>Parameters:</b> {model.num_params:,}</div>"
        f"<div><b>Model Size:</b> {model.size_mb} MB</div>"
    )

    return JSONResponse({
        "top1_label": out["top1_label"],
        "top1_prob": float(out["top1_prob"]),
        "topk": out["topk"],
        "latency_ms": float(out["latency_ms"]),
        "prob_b64": prob_b64,
        "wave_b64": wave_b64,
        "model_info_html": info_html,
        "confusion_available": _CONF_PNG is not None,
        "per_class_available": _CONF_CSV is not None,
        "hints": hints,
        "device": model.device.type.upper(),
        "sr_used": model.sampling_rate,
    })


# --------------------------- CLI & optional ngrok share ---------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--log-level", default="info")
    ap.add_argument("--share", action="store_true", help="Expose a public link via ngrok (requires NGROK_AUTHTOKEN).")
    return ap.parse_args()

def start_uvicorn_in_thread(host: str, port: int, log_level: str):
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level, workers=1)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread

def maybe_start_ngrok(host: str, port: int):
    try:
        from pyngrok import ngrok, conf
    except Exception:
        print("[WARN] --share requested but pyngrok not installed. `pip install pyngrok` and set NGROK_AUTHTOKEN.")
        return None
    token = os.environ.get("NGROK_AUTHTOKEN")
    if not token:
        print("[WARN] --share requested but NGROK_AUTHTOKEN is not set. Set it to enable public link.")
        return None
    conf.get_default().auth_token = token
    tunnel = ngrok.connect(addr=f"http://{host}:{port}", proto="http")
    print(f"[SHARE] Public URL: {tunnel.public_url}")
    return tunnel

if __name__ == "__main__":
    args = parse_args()
    _load_once()

    if args.share:
        srv, th = start_uvicorn_in_thread(args.host, args.port, args.log_level)
        maybe_start_ngrok(args.host, args.port)
        th.join()
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)