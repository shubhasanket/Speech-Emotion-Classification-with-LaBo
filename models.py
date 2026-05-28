import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ClapModel, ClapProcessor
import os

os.environ["TRANSFORMERS_UNSAFE_TORCH_LOAD"] = "1"

CLAP_CHECKPOINT = "laion/clap-htsat-unfused"

def load_clap(device):
    model = ClapModel.from_pretrained(CLAP_CHECKPOINT).to(device)
    processor = ClapProcessor.from_pretrained(CLAP_CHECKPOINT)
    model.eval()
    return model, processor


@torch.no_grad()
def get_audio_embeddings(clap_model, processor, waveforms, device, subbatch=8):
    all_embs = []
    for i in range(0, len(waveforms), subbatch):
        chunk = waveforms[i:i+subbatch]
        inputs = processor(audio=chunk, return_tensors="pt", sampling_rate=48000).to(device)
        outputs = clap_model.audio_model(**inputs)        
        emb = clap_model.audio_projection(outputs.pooler_output)
                
        all_embs.append(F.normalize(emb, dim=-1))
    return torch.cat(all_embs, dim=0)  # B x d


@torch.no_grad()
def get_text_embeddings(clap_model, processor, texts, device):
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)    
    outputs = clap_model.text_model(**inputs)           
    emb = clap_model.text_projection(outputs.pooler_output) # N x d    
    return F.normalize(emb, dim=-1)

class CLAPProbe(nn.Module):

    def __init__(self, n_classes, d_clap=512, hidden=256, dropout=0.3):
        
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_clap),
            nn.Dropout(dropout),
            nn.Linear(d_clap, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )

    
    def forward(self, emb):  
        return self.head(emb)

class E2ECNN(nn.Module):

    def __init__(self, n_classes, c=32, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c, c*2, 3, padding=1), nn.BatchNorm2d(c*2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c*2, c*4, 3, padding=1), nn.BatchNorm2d(c*4), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c*4, c*4, 3, padding=1), nn.BatchNorm2d(c*4), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c*4, n_classes))

    def forward(self, mel):
        x = self.features(mel)    # B x c*4 x H' x W'
        x = x.mean(dim=[2, 3])    # B x c*4
        return self.classifier(x)


concepts = {
    "angry": [
        "loud speech with sharp rising pitch and clipped harsh consonants",
        "forceful aggressive voice with high intensity and fast rate",
        "shouting with tense strained vocal quality and abrupt pauses",
        "speech with loud volume spikes and hard glottal attacks",
        "harsh rough voice with rapid tense delivery and strong stress",
        "high energy speech with irregular rhythm and loud bursts",
        "tense creaky voice with explosive plosives and elevated pitch",
        "a voice with strong harsh resonance and confrontational tone",
    ],
    "happy": [
        "bright high-pitched voice with fast upbeat tempo and rising intonation",
        "warm lively speech with frequent laughter and light breathy quality",
        "upbeat animated voice with wide pitch range and energetic delivery",
        "cheerful voice with smiling quality and rapid speech rate",
        "melodic expressive speech with light resonance and joyful inflections",
        "speech with frequent pitch peaks and smooth flowing rhythm",
        "a warm resonant voice with buoyant rhythm and positive energy",
        "high-energy voice with bright timbre and enthusiastic cadence",
    ],
    "sad": [
        "slow quiet speech with falling pitch and monotone delivery",
        "low-pitched voice with reduced energy and long pauses between words",
        "breathy soft voice with downward intonation and slow tempo",
        "flat subdued speech with minimal pitch variation and weak intensity",
        "trembling voice with irregular breath patterns and slow rate",
        "quiet mumbling speech with low resonance and dragging rhythm",
        "a voice that fades at the end of phrases with low energy",
        "speech with frequent hesitations and a muffled mournful quality",
    ],
    "neutral": [
        "flat steady speech with minimal pitch variation and moderate tempo",
        "even-paced monotone voice with consistent volume and no strong inflection",
        "calm measured speech with regular rhythm and neutral resonance",
        "a voice with little emotional prosody and steady moderate pitch",
        "speech read in a plain even tone without emphasis or energy peaks",
        "low variation in pitch and loudness with a consistent speaking rate",
        "a clear matter-of-fact voice with unremarkable acoustic properties",
        "speech with uniform stress patterns and no prominent prosodic features",
    ],
    "frustrated": [
        "strained tense voice with slow deliberate speech and audible sighing",
        "a voice with rising exasperated intonation and drawn-out words",
        "speech with heavy breathing clipped words and forced controlled tempo",
        "a tight constricted voice with irregular pauses and rising pitch",
        "low energy irritated speech with a monotone edge and trailing volume",
        "voice with involuntary pitch breaks and labored breath patterns",
        "pressured speech that slows and tenses as if suppressing emotion",
        "a voice caught between calm and anger with strained resonance",
    ],
    "excited": [
        "very fast high-pitched speech with wide pitch range and loud volume",
        "rapid breathless voice with rising intonation and energetic bursts",
        "highly animated speech with frequent pitch peaks and fast tempo",
        "a voice with accelerating rate and heightened breathiness",
        "enthusiastic expressive speech with strong stress and lively rhythm",
        "speech with rapidly shifting pitch and high acoustic energy",
        "loud bright voice talking very quickly with great enthusiasm",
        "high-energy speech with short breath groups and emphatic stress",
    ],
    "fear": [
        "quiet trembling voice with high pitch and rapid shallow breathing",
        "nervous whispering speech with irregular tempo and breathy quality",
        "a shaky thin voice with rising pitch and frequent hesitations",
        "soft panicked speech with rapid rate and unstable vocal quality",
        "a voice with audible swallowing gasping and strained high pitch",
        "tense whispered speech with erratic rhythm and low volume",
        "speech interrupted by sharp intakes of breath and trembling pitch",
        "a fragile wavering voice with high tension and constricted resonance",
    ],
    "surprise": [
        "sudden high-pitched exclamation with sharp pitch rise and loud onset",
        "a voice with abrupt volume increase and rapid pitch jump",
        "startled speech with wide pitch swings and gasping intake of breath",
        "speech that begins with a sharp high-pitched burst and then slows",
        "an astonished voice with elongated vowels and rising intonation",
        "a voice suddenly jumping in pitch with a loud breathy exclamation",
        "speech with an abrupt onset high energy burst and falling tail",
        "exclamatory speech with strong glottal emphasis and wide dynamic range",
    ],
    "disgust": [
        "low flat voice with slow deliberate pacing and nasal resonance",
        "speech with drawn-out words falling intonation and lip tension",
        "a contemptuous voice with slow rate and harsh vocal texture",
        "dismissive speech with low pitch dropped volume and clipped endings",
        "a voice with lip curl quality reduced resonance and flat affect",
        "slow disdainful speech with minimal pitch variation and cold tone",
        "a voice with heavy nasal quality low energy and flat cadence",
        "speech delivered slowly with a flat harsh edge and low intensity",
    ],
    "other": [
        "speech that does not fit a clear emotional category",
        "ambiguous vocal affect with mixed or unclear emotional tone",
        "flat unremarkable speech with no dominant emotional quality",
        "a voice with inconsistent prosody and no single prevailing emotion",
        "speech switching between emotional registers mid-utterance",
        "monotone delivery with occasional unpredictable pitch shifts",
        "a neutral-to-slightly-expressive voice with no clear label",
        "speech that sounds neither calm nor aroused with weak affect",
    ]
}



class LaBoAudio(nn.Module):
    def __init__(self, class_names, clap_model, processor, device, concepts=None):
        super().__init__()
        self.class_names = class_names

        if concepts is None:
            concepts = {k: concepts.get(k, [f"a person speaking with {k}"]) for k in class_names}
        concept_list, concept_labels = [], []
        for i, name in enumerate(class_names):
            for c in concepts[name]:
                concept_list.append(c)
                concept_labels.append(i)

        self.concept_list = concept_list
        N_C = len(concept_list)
        n_cls = len(class_names)
        embs = get_text_embeddings(clap_model, processor, concept_list, device)  # N_C x d
        self.register_buffer("concept_embs", embs) # renaming for ease

        W_init = torch.zeros(n_cls, N_C)
        for c_idx, cls_idx in enumerate(concept_labels):
            W_init[cls_idx, c_idx] = 1.0
        self.W = nn.Parameter(W_init)

    def forward(self, audio_emb): # B x d 
        scores = audio_emb @ self.concept_embs.T # B x N_C
        W_norm = F.softmax(self.W, dim=0) # n_cls x N_C
        return scores @ W_norm.T # B x n_cls

    def top_concepts(self, cls_idx, k=5):
        W_norm = F.softmax(self.W, dim=0)
        weights = W_norm[cls_idx].detach().cpu()
        top = weights.argsort(descending=True)[:k]
        return [(self.concept_list[i], round(float(weights[i]), 4)) for i in top]

class CLAPZeroShot(nn.Module):
    def __init__(self, class_names, clap_model, processor, device, temperature=0.07):
        super().__init__()
        self.temperature = temperature

        prompts = [f"a person speaking with {name}" for name in class_names]
        text_emb = get_text_embeddings(clap_model, processor, prompts, device)  # n_cls x d
        self.register_buffer("text_embs", text_emb) # renaming for ease

    
    def forward(self, audio_emb): # B x d 
        return (audio_emb @ self.text_embs.T) / self.temperature
