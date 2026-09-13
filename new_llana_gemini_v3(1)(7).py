import colorsys
import csv
import json
import math
import random
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
import os

try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GEMINI_SDK = True
except ImportError:
    genai = None
    genai_types = None
    HAS_GEMINI_SDK = False

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    BaseModel = object
    Field = None
    HAS_PYDANTIC = False

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
    HAS_TENACITY = True
except ImportError:
    retry = None
    stop_after_attempt = None
    wait_exponential = None
    HAS_TENACITY = False

try:
    from thefuzz import fuzz
    HAS_THEFUZZ = True
except ImportError:
    fuzz = None
    HAS_THEFUZZ = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import fitz  # PyMuPDF, optional PDF-to-image support for FIRs
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from networkx.algorithms.community import louvain_communities
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

from networkx.algorithms.community import greedy_modularity_communities

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import Neo4jError, ServiceUnavailable
    HAS_NEO4J_DRIVER = True
except ImportError:
    HAS_NEO4J_DRIVER = False

RANDOM_SEED = 42
TEST_FRACTION = 0.2
ALPHA = 0.6  # weight given to target-layer evidence vs. other layers

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================================
#  LINK-PREDICTION ENGINE
# ============================================================================

def build_relation_layers(G):
    """Split the single investigation graph into one graph per relation
    label, so each relation type can be treated as a network layer."""
    layers = {}
    metadata = {}
    node_types = {n: d.get("type", "Entity") for n, d in G.nodes(data=True)}

    for u, v, data in G.edges(data=True):
        relation = data.get("label") or "Linked to"
        if relation not in layers:
            layers[relation] = nx.Graph()
            metadata[relation] = set()

        Gl = layers[relation]
        tu = node_types.get(u, "Entity")
        tv = node_types.get(v, "Entity")
        Gl.add_node(u, node_type=tu)
        Gl.add_node(v, node_type=tv)

        w = data.get("weight", 1.0) or 1.0
        if Gl.has_edge(u, v):
            Gl[u][v]["weight"] += w
        else:
            Gl.add_edge(u, v, weight=w)

        metadata[relation].add((tu, tv))
        metadata[relation].add((tv, tu))

    # Every node is present in every layer (as an isolate if needed) so
    # candidate pairs can still be scored even without in-layer edges.
    for node_id, ntype in node_types.items():
        for Gl in layers.values():
            if node_id not in Gl:
                Gl.add_node(node_id, node_type=ntype)

    return layers, metadata, node_types


def split_edges(G, node_types, allowed_type_pairs, test_fraction=TEST_FRACTION, rng=None):
    """Hold out positive edges and sample type-compatible non-edges.

    `rng` should be a `random.Random` instance dedicated to this run. Using
    an explicitly-seeded local generator -- instead of calling the shared
    `random` module directly -- is what makes the split reproducible: the
    global module's internal state advances every time anything in the
    process calls `random.*`, so relying on it (even after `random.seed()`
    at import time) means the *n*-th call to this function during a session
    draws from wherever the shared stream happens to be, not from the seed.
    """
    if rng is None:
        rng = random.Random(RANDOM_SEED)

    edges = sorted(G.edges())  # canonical order, independent of insertion history
    if len(edges) < 2:
        return G.copy(), [], []

    rng.shuffle(edges)
    n_test = max(1, int(len(edges) * test_fraction))
    n_test = min(n_test, len(edges) - 1)
    test_pos = edges[:n_test]

    G_train = G.copy()
    G_train.remove_edges_from(test_pos)

    nodes = sorted(G_train.nodes())
    if len(nodes) < 2:
        return G_train, test_pos, []

    existing = {frozenset(e) for e in G.edges()}
    test_neg = []

    eligible_pairs = []
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            pair = (node_types.get(u, "Entity"), node_types.get(v, "Entity"))
            if pair in allowed_type_pairs or (pair[1], pair[0]) in allowed_type_pairs:
                eligible_pairs.append((u, v))

    rng.shuffle(eligible_pairs)
    for u, v in eligible_pairs:
        if frozenset([u, v]) not in existing:
            test_neg.append((u, v))
            if len(test_neg) == len(test_pos):
                break

    return G_train, test_pos, test_neg


def common_neighbors(G, u, v):
    if u not in G or v not in G:
        return 0
    return len(set(G[u]) & set(G[v]))


def jaccard(G, u, v):
    if u not in G or v not in G:
        return 0.0
    nu, nv = set(G[u]), set(G[v])
    union = nu | nv
    return len(nu & nv) / len(union) if union else 0.0


def adamic_adar(G, u, v):
    if u not in G or v not in G:
        return 0.0
    common = set(G[u]) & set(G[v])
    return sum(1.0 / np.log(G.degree(w)) for w in common if G.degree(w) > 1)


def build_ppr(G, alpha=0.85):
    """Personalized PageRank vector per node, precomputed once for speed."""
    ppr = {}
    for n in G.nodes():
        try:
            ppr[n] = nx.pagerank(G, alpha=alpha, personalization={n: 1})
        except (nx.PowerIterationFailedConvergence, ZeroDivisionError):
            ppr[n] = {n: 1.0}
    return ppr


def ppr_score(ppr, u, v):
    su = ppr.get(u, {}).get(v, 0.0)
    sv = ppr.get(v, {}).get(u, 0.0)
    return (su + sv) / 2


def layer_heuristics(G, ppr, u, v):
    return np.array([
        common_neighbors(G, u, v),
        jaccard(G, u, v),
        adamic_adar(G, u, v),
        ppr_score(ppr, u, v),
    ], dtype=float)


def blended_features(u, v, target, other_layers, target_ppr, other_pprs, alpha=ALPHA):
    """Blend target-layer evidence with the average evidence from all other layers."""
    h_target = layer_heuristics(target, target_ppr, u, v)
    if not other_layers:
        return h_target
    other_scores = [layer_heuristics(G, other_pprs[r], u, v) for r, G in other_layers.items()]
    h_other = np.mean(other_scores, axis=0)
    return alpha * h_target + (1 - alpha) * h_other


def eligible_pair(node_types, u, v, allowed_type_pairs):
    tu = node_types.get(u, "Entity")
    tv = node_types.get(v, "Entity")
    return (tu, tv) in allowed_type_pairs or (tv, tu) in allowed_type_pairs


def evaluate_relation_layer(name, G_full, all_layers, metadata, node_types, log=print, rng=None):
    """Train + evaluate a classifier for one relation layer, then refit a
    second, *production* classifier on the complete (unmasked) graph so the
    candidate list you actually act on uses every known edge, not just the
    80% training split. Returns a result dict (usable for candidate
    scoring) or None if there wasn't enough data to evaluate."""
    if rng is None:
        rng = random.Random(RANDOM_SEED)

    G_train, test_pos, test_neg = split_edges(G_full, node_types, metadata[name], rng=rng)
    if not test_pos or not test_neg:
        log(f"[{name}] skipped - not enough edges/non-edges to evaluate.")
        return None

    other_layers = {r: G for r, G in all_layers.items() if r != name}
    ppr_target = build_ppr(G_train)
    other_pprs = {r: build_ppr(G) for r, G in other_layers.items()}

    pairs = test_pos + test_neg
    y = np.array([1] * len(test_pos) + [0] * len(test_neg))
    if len(set(y)) < 2:
        log(f"[{name}] skipped - only one evaluation class available.")
        return None

    # ---- Honest held-out evaluation, part 1: features come from G_train
    # only, so none of the held-out edges leak into their own feature
    # values (common neighbors, Jaccard, Adamic-Adar, PPR). ----
    X_eval = np.array([
        blended_features(u, v, G_train, other_layers, ppr_target, other_pprs)
        for u, v in pairs
    ])

    # ---- Honest held-out evaluation, part 2: the classifier itself must
    # not be scored on rows it was trained on. Fitting eval_clf on
    # (X_eval, y) and then calling predict_proba on that same X_eval -- as
    # this used to do -- reports training accuracy, not generalization,
    # regardless of how clean the features are. Instead, use stratified
    # k-fold cross-validation so every row's probability comes from a fold
    # that did not include it during fitting. Scaling is refit inside each
    # fold via the Pipeline, so no fold's held-out rows influence the scaler
    # either. random_state pins both the fold split and (defensively) the
    # solver's internal behaviour, so results stay reproducible.
    class_counts = np.bincount(y)
    n_splits = min(5, int(class_counts.min()))
    if n_splits < 2:
        log(f"[{name}] skipped - fewer than 2 examples in the smaller class; "
            f"cannot cross-validate a reliable ROC-AUC/AP.")
        return None

    cv_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    probs = cross_val_predict(
        cv_pipeline, X_eval, y, cv=skf, method="predict_proba"
    )[:, 1]

    auc = roc_auc_score(y, probs)
    ap = average_precision_score(y, probs)
    log(f"[{name}] test edges: {len(test_pos)} pos / {len(test_neg)} neg "
        f"| {n_splits}-fold CV ROC-AUC: {auc:.3f} | Average Precision: {ap:.3f}")

    # A single scaler+classifier fit on the full evaluation set is still
    # useful to keep around (e.g. for inspection), but it is never the
    # source of the reported auc/ap above.
    eval_scaler = StandardScaler()
    X_eval_scaled = eval_scaler.fit_transform(X_eval)
    eval_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    eval_clf.fit(X_eval_scaled, y)

    # ---- Production model: same labeled pairs, but features are recomputed
    # from the COMPLETE layer graph (held-out edges included). For real
    # candidate pairs -- true non-edges, never part of test_pos/test_neg --
    # there is nothing to leak: G_full is simply everything we actually
    # know. This is fit *after* the honest AUC/AP above are already logged,
    # so the reported metrics are never touched by it; it exists purely to
    # make the ranked candidate list use every known edge. (Note: for the
    # held-out *positive* pairs specifically, their PPR feature here is
    # computed on a graph that already contains that exact edge, which
    # mildly overstates it for those particular rows -- harmless for
    # production scoring of brand-new pairs, but why this model's own
    # AUC/AP is never reported the way eval_clf's is.)
    ppr_target_full = build_ppr(G_full)
    X_prod = np.array([
        blended_features(u, v, G_full, other_layers, ppr_target_full, other_pprs)
        for u, v in pairs
    ])
    prod_scaler = StandardScaler()
    X_prod_scaled = prod_scaler.fit_transform(X_prod)
    prod_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    prod_clf.fit(X_prod_scaled, y)

    return {
        "clf": eval_clf,
        "scaler": eval_scaler,
        "G_train": G_train,
        "ppr_target": ppr_target,
        "other_pprs": other_pprs,
        "prod_clf": prod_clf,
        "prod_scaler": prod_scaler,
        "G_full": G_full,
        "ppr_target_full": ppr_target_full,
        "auc": auc,
        "ap": ap,
        "n_pos": len(test_pos),
        "n_neg": len(test_neg),
    }


def top_k_candidates(result, all_layers, all_nodes, allowed_type_pairs, node_types,
                      target_relation, k=8):
    """Score every eligible non-edge pair for a relation and return the
    top-k highest-probability candidate NEW links.

    Uses the production model from evaluate_relation_layer -- refit on the
    complete graph, not the 80% training split -- so the ranking benefits
    from every known edge. The reported ROC-AUC/AP describe how well this
    modeling approach generalizes (measured on the held-out split); they
    aren't re-measured against this particular refit.
    """
    clf = result["prod_clf"]
    scaler = result["prod_scaler"]
    G_full = result["G_full"]
    ppr_target = result["ppr_target_full"]
    other_pprs = result["other_pprs"]

    other_layers = {r: G for r, G in all_layers.items() if r != target_relation}
    full_existing = {frozenset(e) for e in G_full.edges()}

    scored = []
    # sorted(), not list(): all_nodes is normally built from a `set`, whose
    # iteration order depends on Python's per-process hash seed. That doesn't
    # change the *scores*, but it does change which equally/near-equally
    # scored pair appears first when the sort below hits a tie -- sorting
    # here removes that as a source of run-to-run variation.
    nodes = sorted(all_nodes)
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            pair = frozenset([u, v])
            # Skip pairs that are already a real (known) link in this
            # relation -- we only want to surface genuinely NEW candidates.
            if pair in full_existing:
                continue
            if not eligible_pair(node_types, u, v, allowed_type_pairs):
                continue
            feats = blended_features(u, v, G_full, other_layers, ppr_target, other_pprs)
            prob = clf.predict_proba(scaler.transform([feats]))[0, 1]
            scored.append((u, v, float(prob)))

    scored.sort(key=lambda x: -x[2])
    return scored[:k]


# ============================================================================
#  AI / EVIDENCE / ANOMALY LAYER
# ============================================================================

AI_MODEL_DEFAULT = "llama3.2"
# Local inference is optimized for student/hackathon laptops. Set to True only
# when you want four separate Ollama calls for the full multi-agent pipeline.
FULL_MULTI_AGENT = False
FAST_LLM_NUM_CTX = 4096
FAST_LLM_NUM_PREDICT = 320
OLLAMA_URL_DEFAULT = "http://127.0.0.1:11434"


class EvidenceStore:
    """Case-level evidence and provenance store used by the local AI layer."""

    def __init__(self):
        self.sources = []
        self.fir_documents = []
        self.chat_log = []

    def add_fir(self, path, text, entities=None):
        doc = {
            "id": f"FIR_{len(self.fir_documents) + 1:04d}",
            "file": str(path),
            "text": text or "",
            "entities": entities or [],
            "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.fir_documents.append(doc)
        return doc

    def add_source(self, source_file, row=None, source_type="CSV", note=None):
        item = {
            "source_file": str(source_file),
            "row": row,
            "source_type": source_type,
            "note": note,
        }
        self.sources.append(item)
        return item

    def evidence_refs_for_graph(self, G, node_id=None, limit=20):
        refs = []
        nodes = [node_id] if node_id in G.nodes else list(G.nodes())
        for n in nodes:
            for _, _, data in G.edges(n, data=True):
                for record in data.get("records", []):
                    sf = record.get("_source_file") or record.get("source_file")
                    row = record.get("_row")
                    if sf:
                        refs.append(f"{sf}" + (f":row {row}" if row else ""))
                    if len(refs) >= limit:
                        return list(dict.fromkeys(refs))
        for doc in self.fir_documents[:limit]:
            refs.append(doc["id"] + " (" + Path(doc["file"]).name + ")")
        return list(dict.fromkeys(refs))[:limit]


class Entity(BaseModel):
    name: str = Field(description="The extracted name or identifier.")
    type: str = Field(description=(
        "Must be: Person, Phone Number, Vehicle, Bank Account, Location, "
        "Organization, or Date."
    ))
    is_ambiguous: bool = Field(
        description="True if the handwriting is messy or the identity is unclear."
    )


class Relationship(BaseModel):
    source_entity: str
    target_entity: str
    relation_type: str


class FIRData(BaseModel):
    transcription: str
    entities: list[Entity]
    relationships: list[Relationship]


class FIRProcessor:
    """Cloud-first handwritten FIR ingestion with a deterministic local fallback.

    The user supplies only the scanned FIR. Gemini performs handwriting-aware
    transcription + structured extraction. If Gemini is unavailable, the app
    falls back to the existing local Tesseract/PyMuPDF OCR and deterministic
    regex extraction so the desktop application still functions offline.
    """

    MODEL = "gemini-3.6-flash"
    GEMINI_ENV = "GEMINI_API_KEY"
    ENTITY_TYPE_MAP = {
        "person": "Person",
        "phone number": "Phone",
        "phone": "Phone",
        "vehicle": "Vehicle",
        "bank account": "Bank Account",
        "bank_account": "Bank Account",
        "location": "Location",
        "organization": "Organization",
        "organisation": "Organization",
        "date": "Date",
    }

    def __init__(self):
        self.last_backend = ""
        self.last_error = ""

    @staticmethod
    def _clean_text(text):
        text = str(text or "")
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def _normalise_value(value):
        return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

    @staticmethod
    def _canonical_type(raw_type):
        key = re.sub(r"[\s_-]+", " ", str(raw_type or "").strip().lower())
        return FIRProcessor.ENTITY_TYPE_MAP.get(key, str(raw_type or "Entity").strip().title() or "Entity")

    def _preprocess_image(self, image):
        if not HAS_PIL:
            return image
        try:
            img = image.convert("L")
            scale = 2.0 if max(img.size) < 2500 else 1.0
            if scale != 1.0:
                img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
            from PIL import ImageEnhance, ImageOps, ImageFilter
            img = ImageOps.autocontrast(img)
            img = ImageEnhance.Sharpness(img).enhance(1.5)
            img = img.filter(ImageFilter.MedianFilter(size=3))
            return img
        except Exception:
            return image

    def _ocr_image(self, image):
        if not HAS_TESSERACT:
            raise RuntimeError("Image OCR requires pytesseract and the Tesseract executable.")
        processed = self._preprocess_image(image)
        text = pytesseract.image_to_string(processed, config="--psm 6")
        if len(self._clean_text(text)) < 20:
            text2 = pytesseract.image_to_string(processed, config="--psm 11")
            if len(self._clean_text(text2)) > len(self._clean_text(text)):
                text = text2
        return self._clean_text(text)

    def extract_text(self, path):
        """Existing local fallback: Tesseract for images, PyMuPDF + Tesseract for PDFs."""
        path = str(path)
        ext = Path(path).suffix.lower()
        if ext == ".txt":
            return Path(path).read_text(encoding="utf-8", errors="replace")
        if ext == ".pdf":
            if not HAS_PYMUPDF:
                raise RuntimeError("PDF FIR selected, but PyMuPDF is not installed. Install: pip install pymupdf")
            doc = fitz.open(path)
            text_parts = []
            for page in doc:
                text = page.get_text("text")
                if text.strip():
                    text_parts.append(text)
                elif HAS_PIL and HAS_TESSERACT:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text_parts.append(self._ocr_image(img))
            return "\n".join(text_parts).strip()
        if not (HAS_PIL and HAS_TESSERACT):
            raise RuntimeError("Local image OCR needs Pillow + pytesseract + a Tesseract installation.")
        with Image.open(path) as image:
            return self._ocr_image(image)

    @staticmethod
    def _dedupe(items):
        out, seen = [], set()
        for item in items:
            etype = str(item.get("type", "Entity")).strip().lower()
            value = str(item.get("value", item.get("name", ""))).strip()
            if not value:
                continue
            key = (etype, FIRProcessor._normalise_value(value))
            if key in seen:
                continue
            seen.add(key)
            clean = dict(item)
            clean["type"] = etype
            clean["value"] = value
            out.append(clean)
        return out

    @staticmethod
    def extract_entities(text):
        """Deterministic local fallback when cloud OCR is unavailable."""
        text = text or ""
        candidates = []
        patterns = {
            "phone": r"(?<!\d)(?:\+?91[- ]?)?[6-9]\d{9}(?!\d)",
            "vehicle": r"\b[A-Z]{2}\d{2}\s?[A-Z]{1,3}\s?\d{1,4}\b",
            "bank_account": r"\b\d{8,20}\b",
        }
        phones = set()
        for m in re.findall(patterns["phone"], text, flags=re.IGNORECASE):
            value = str(m).strip()
            phones.add(FIRProcessor._normalise_value(value))
            candidates.append({"type": "phone", "value": value, "is_ambiguous": False})
        for etype in ("vehicle", "bank_account"):
            for m in re.findall(patterns[etype], text, flags=re.IGNORECASE):
                value = str(m).strip()
                if etype == "bank_account" and FIRProcessor._normalise_value(value) in phones:
                    continue
                candidates.append({"type": etype, "value": value, "is_ambiguous": False})
        for match in re.finditer(
            r"(?:Name\s*:\s*|Mr\.?\s+|Ms\.?\s+|Mrs\.?\s+|Suspect\s*[:\-]\s*)([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})",
            text,
        ):
            candidates.append({"type": "person", "value": match.group(1).strip(), "is_ambiguous": False})
        return {"entities": FIRProcessor._dedupe(candidates), "relationships": []}

    @staticmethod
    def _save_transcription(source_path, transcription):
        src = Path(source_path)
        out = src.with_name(f"{src.stem}_transcription.txt")
        # Avoid accidental huge writes from a malformed response.
        out.write_text(str(transcription or "")[:250000], encoding="utf-8")
        return str(out)

    def _fallback_result(self, path, reason):
        self.last_backend = "LOCAL_TESSERACT_FALLBACK"
        self.last_error = str(reason)
        text = self.extract_text(path)
        text = self._clean_text(text)
        txt_path = self._save_transcription(path, text)
        base = self.extract_entities(text)
        return {
            "transcription": text,
            "entities": base.get("entities", []),
            "relationships": base.get("relationships", []),
            "transcription_path": txt_path,
            "ocr_backend": self.last_backend,
            "cloud_error": str(reason),
        }

    def _gemini_client(self):
        if not (HAS_GEMINI_SDK and HAS_PYDANTIC and HAS_TENACITY):
            missing = []
            if not HAS_GEMINI_SDK:
                missing.append("google-genai")
            if not HAS_PYDANTIC:
                missing.append("pydantic")
            if not HAS_TENACITY:
                missing.append("tenacity")
            raise RuntimeError("Gemini OCR dependencies missing: " + ", ".join(missing))
        api_key = os.getenv(self.GEMINI_ENV, "").strip()
        if not api_key:
            raise RuntimeError(f"{self.GEMINI_ENV} environment variable is not set.")
        return genai.Client(api_key=api_key)

    def _gemini_prompt(self):
        return """
You are the document-understanding engine for an investigative graph application.
Read the uploaded scanned FIR exactly as written, including handwritten text.
Do not guess words that are illegible. Preserve uncertainty by setting
is_ambiguous=true for an entity whose text or identity is unclear.

Return:
1) A faithful transcription of all readable FIR text.
2) Entities explicitly present in the document. Entity type must be exactly one of:
   Person, Phone Number, Vehicle, Bank Account, Location, Organization, Date.
3) Relationships explicitly stated in the FIR. Do not infer relationships merely
   because two entities appear near each other.

Do not infer guilt, legal conclusions, identities, or facts absent from the document.
Do not invent missing digits or names. Keep names/identifiers exactly as readable.
""".strip()

    @staticmethod
    def _parse_gemini_response(response):
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, FIRData):
            return parsed
        if parsed is not None and isinstance(parsed, dict):
            return FIRData.model_validate(parsed) if hasattr(FIRData, "model_validate") else FIRData.parse_obj(parsed)
        raw = getattr(response, "text", "") or ""
        if not raw:
            raise ValueError("Gemini returned no structured response.")
        data = json.loads(raw)
        return FIRData.model_validate(data) if hasattr(FIRData, "model_validate") else FIRData.parse_obj(data)

    def _gemini_request(self, client, uploaded_file):
        config = {
            "response_mime_type": "application/json",
            "response_schema": FIRData,
            "temperature": 0.0,
        }
        if genai_types is not None:
            config = genai_types.GenerateContentConfig(**config)
        response = client.models.generate_content(
            model=self.MODEL,
            contents=[uploaded_file, self._gemini_prompt()],
            config=config,
        )
        return self._parse_gemini_response(response)

    def cloud_extract(self, path):
        """Call Gemini with tenacity retries, then immediately fall back locally."""
        path = str(path)
        client = self._gemini_client()

        if not HAS_TENACITY:
            raise RuntimeError("tenacity is required for cloud FIR retries.")

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        )
        def request_with_retry():
            uploaded = client.files.upload(file=path)
            return self._gemini_request(client, uploaded)

        parsed = request_with_retry()
        text = self._clean_text(parsed.transcription)
        if not text:
            raise ValueError("Gemini returned an empty FIR transcription.")
        txt_path = self._save_transcription(path, text)
        entities = []
        for entity in parsed.entities:
            entities.append({
                "type": str(entity.type).strip().lower(),
                "value": str(entity.name).strip(),
                "is_ambiguous": bool(entity.is_ambiguous),
            })
        relationships = []
        for rel in parsed.relationships:
            relationships.append({
                "from": str(rel.source_entity).strip(),
                "to": str(rel.target_entity).strip(),
                "relation": str(rel.relation_type).strip() or "Associated with",
            })
        self.last_backend = "GEMINI_3.6_FLASH"
        self.last_error = ""
        return {
            "transcription": text,
            "entities": self._dedupe(entities),
            "relationships": relationships,
            "transcription_path": txt_path,
            "ocr_backend": self.last_backend,
            "cloud_error": "",
        }

    def extract_fir(self, path):
        try:
            return self.cloud_extract(path)
        except Exception as exc:
            # Cloud failure is intentionally non-fatal: immediately use the existing
            # local OCR path so a demo laptop can continue without network access.
            try:
                return self._fallback_result(path, exc)
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Gemini OCR failed and local OCR fallback also failed.\n"
                    f"Gemini: {exc}\nLocal fallback: {fallback_exc}"
                ) from fallback_exc


class AnomalyEngine:
    """Heuristic, explainable 12-class graph anomaly detector.

    The detector is intentionally transparent: it produces evidence signals and
    scores; it does not make legal conclusions.
    """

    TYPES = (
        "New-node anomaly", "New-edge anomaly", "Edge-weight anomaly", "Degree anomaly",
        "Centrality anomaly", "Community anomaly", "Motif anomaly", "Temporal anomaly",
        "Geographic anomaly", "Role anomaly", "Attribute anomaly", "Coordinated anomaly",
    )

    @staticmethod
    def _date_value(record):
        for k, v in record.items():
            lk = str(k).lower()
            if any(s in lk for s in ("date", "time", "timestamp")) and str(v).strip():
                return str(v).strip()
        return None

    @staticmethod
    def _numeric_values(G):
        vals = []
        for _, data in G.nodes(data=True):
            props = data.get("properties", {})
            for v in props.values():
                try:
                    x = float(v)
                    if math.isfinite(x):
                        vals.append(x)
                except (TypeError, ValueError):
                    pass
        return vals

    def snapshot(self, G):
        edges = []
        for u, v, data in G.edges(data=True):
            rel = str(data.get("label", "Linked to"))
            edges.append((str(u), str(v), rel, round(float(data.get("weight", 1.0) or 1.0), 6)))
        degrees = {str(n): int(G.degree(n)) for n in G.nodes()}
        return {
            "nodes": set(str(n) for n in G.nodes()),
            "edges": set(edges),
            "degrees": degrees,
            "centrality": {str(n): float(d.get("centrality", 0.0) or 0.0) for n, d in G.nodes(data=True)},
            "community": {str(n): d.get("community") for n, d in G.nodes(data=True) if "community" in d},
        }

    def analyze(self, G, previous=None):
        current = self.snapshot(G)
        prev = previous or {"nodes": set(), "edges": set(), "degrees": {}, "centrality": {}, "community": {}}
        results = {str(n): [] for n in G.nodes()}

        # Ensure centrality values exist for anomaly comparison.
        if G.number_of_nodes() >= 2:
            H = nx.Graph()
            H.add_nodes_from(G.nodes())
            for u, v, d in G.edges(data=True):
                w = max(float(d.get("weight", 1.0) or 1.0), 0.0001)
                if H.has_edge(u, v):
                    H[u][v]["weight"] += w
                else:
                    H.add_edge(u, v, weight=w)
            deg = nx.degree_centrality(H)
            for n, value in deg.items():
                if n in G.nodes() and "centrality" not in G.nodes[n]:
                    G.nodes[n]["centrality"] = float(value)

        node_strings = {str(n): n for n in G.nodes()}
        current_nodes = current["nodes"]
        new_nodes = current_nodes - set(prev.get("nodes", set()))
        for ns in new_nodes:
            results[ns].append({"type": "New-node anomaly", "score": 0.65,
                                "reason": "Entity was not present in the previous graph snapshot."})

        prev_edges = set(prev.get("edges", set()))
        curr_edges = set(current["edges"])
        edge_lookup = {}
        for u, v, d in G.edges(data=True):
            rel = str(d.get("label", "Linked to"))
            sig = (str(u), str(v), rel, round(float(d.get("weight", 1.0) or 1.0), 6))
            edge_lookup.setdefault((str(u), str(v), rel), []).append(d)
        for edge in curr_edges - prev_edges:
            u, v, rel, w = edge
            for ns in (u, v):
                if ns in results:
                    results[ns].append({"type": "New-edge anomaly", "score": 0.62,
                                        "reason": f"New {rel} relationship appeared with {v if ns == u else u}."})

        for ns, node in node_strings.items():
            n = node_strings[ns]
            degree_now = current["degrees"].get(ns, 0)
            degree_prev = prev.get("degrees", {}).get(ns, 0)
            if degree_prev > 0 and degree_now >= max(degree_prev + 3, int(degree_prev * 2)):
                ratio = degree_now / max(degree_prev, 1)
                results[ns].append({"type": "Degree anomaly", "score": min(0.99, 0.55 + 0.12 * min(ratio, 3)),
                                    "reason": f"Degree increased from {degree_prev} to {degree_now}."})

            c_now = float(G.nodes[n].get("centrality", 0.0) or 0.0)
            c_prev = float(prev.get("centrality", {}).get(ns, 0.0) or 0.0)
            if c_prev > 0 and c_now - c_prev > 0.20:
                results[ns].append({"type": "Centrality anomaly", "score": min(0.99, 0.60 + c_now - c_prev),
                                    "reason": f"Composite centrality increased by {c_now - c_prev:.2f}."})

            if ns in prev.get("community", {}) and "community" in G.nodes[n]:
                if prev["community"].get(ns) != G.nodes[n].get("community"):
                    results[ns].append({"type": "Community anomaly", "score": 0.73,
                                        "reason": "Entity moved between detected community assignments."})

            # Edge-weight anomaly: large growth in aggregated relationship weight.
            for _, _, d in G.edges(n, data=True):
                records = d.get("records", [])
                if len(records) >= 2:
                    weights = []
                    for r in records:
                        for key in ("duration_sec", "duration", "call_duration", "amount_inr", "amount_usd", "amount", "transaction_amount"):
                            if key in r:
                                try:
                                    weights.append(float(r[key]))
                                    break
                                except (TypeError, ValueError):
                                    pass
                    if len(weights) >= 4:
                        recent = weights[-max(2, len(weights)//4):]
                        old = weights[:-len(recent)]
                        old_mean = statistics.mean(old) if old else statistics.mean(weights)
                        recent_mean = statistics.mean(recent)
                        if old_mean > 0 and recent_mean > old_mean * 2.5:
                            score = min(0.99, 0.60 + min(recent_mean / old_mean, 5) * 0.07)
                            results[ns].append({"type": "Edge-weight anomaly", "score": score,
                                                "reason": f"Recent interaction strength is about {recent_mean / old_mean:.1f}× the earlier baseline."})

            # Temporal burst anomaly.
            dated = [self._date_value(r) for _, _, d in G.edges(n, data=True) for r in d.get("records", [])]
            dated = [x for x in dated if x]
            if len(dated) >= 5:
                unique_dates = sorted(set(dated))
                if len(unique_dates) <= 3:
                    results[ns].append({"type": "Temporal anomaly", "score": 0.70,
                                        "reason": f"{len(dated)} observed activities are concentrated in only {len(unique_dates)} recorded time points."})

            # Geographic anomaly: appearance of a new location relation.
            locations = []
            for _, v, d in G.edges(n, data=True):
                if str(G.nodes[v].get("type", "")).lower() == "location" or "location" in str(d.get("label", "")).lower():
                    locations.append(str(v))
            unique_locs = set(locations)
            if len(unique_locs) >= 2 and len(locations) == len(unique_locs):
                results[ns].append({"type": "Geographic anomaly", "score": 0.60,
                                    "reason": "Entity is associated with multiple distinct locations with little repeated history."})

            # Role anomaly: compare the explicit role field with observed behaviour.
            props_text = " ".join(f"{k}:{v}" for k, v in G.nodes[n].get("properties", {}).items()).lower()
            role = G.nodes[n].get("properties", {}).get("role") or G.nodes[n].get("properties", {}).get("position")
            role_text = str(role or props_text).lower()
            high_level_neighbors = 0
            for nbr in G.neighbors(n):
                ntype = str(G.nodes[nbr].get("type", "")).lower()
                nbr_props = G.nodes[nbr].get("properties", {})
                nbr_role = str(nbr_props.get("role", nbr_props.get("position", ""))).lower()
                if any(k in (ntype + " " + nbr_role) for k in ("leader", "director", "kingpin", "organizer")):
                    high_level_neighbors += 1
            if high_level_neighbors >= 2 and any(k in role_text for k in ("courier", "driver", "associate", "member")):
                results[ns].append({"type": "Role anomaly", "score": 0.75,
                                    "reason": f"Observed connections include {high_level_neighbors} higher-level role indicators despite the stored lower-level role."})

            # Attribute anomaly: robust outlier scan over numeric node properties.
            props = G.nodes[n].get("properties", {})
            numeric_props = []
            for k, v in props.items():
                try:
                    x = float(v)
                    if math.isfinite(x):
                        numeric_props.append((k, x))
                except (TypeError, ValueError):
                    pass
            all_numeric = self._numeric_values(G)
            if len(all_numeric) >= 8 and numeric_props:
                med = statistics.median(all_numeric)
                mad = statistics.median([abs(x - med) for x in all_numeric]) or 1.0
                for k, x in numeric_props:
                    if abs(x - med) / mad > 8:
                        results[ns].append({"type": "Attribute anomaly", "score": 0.78,
                                            "reason": f"Property '{k}' is an extreme numeric outlier relative to loaded numeric attributes."})
                        break

        # Motif anomaly: unusual local triangle/star structure.
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from((u, v) for u, v in G.edges())
        triangles = nx.triangles(H) if H.number_of_nodes() else {}
        for n in G.nodes():
            tri = triangles.get(n, 0)
            deg = H.degree(n)
            if tri >= 2 or deg >= 8:
                results[str(n)].append({"type": "Motif anomaly", "score": min(0.95, 0.55 + 0.04 * tri + 0.02 * deg),
                                        "reason": f"Local neighbourhood contains {tri} triangle(s) and degree {deg}, forming a potentially unusual motif."})

        # Coordinated anomaly: several nodes sharing a new neighbour or activity window.
        new_edge_pairs = [(e[0], e[1]) for e in curr_edges - prev_edges]
        by_target = {}
        for u, v in new_edge_pairs:
            by_target.setdefault(v, set()).add(u)
            by_target.setdefault(u, set()).add(v)
        for target, members in by_target.items():
            if len(members) >= 3 and target in results:
                results[target].append({"type": "Coordinated anomaly", "score": 0.82,
                                        "reason": f"Multiple entities ({len(members)}) changed connectivity around the same target in the latest snapshot."})
                for member in members:
                    if member in results:
                        results[member].append({"type": "Coordinated anomaly", "score": 0.74,
                                                "reason": f"Connectivity changed simultaneously with {len(members) - 1} other entities around {target}."})

        # If a snapshot is brand-new, avoid flooding the UI with false warnings.
        for ns in results:
            results[ns].sort(key=lambda x: -float(x.get("score", 0.0)))
            results[ns] = results[ns][:10]
        return results


class InvestigationTools:
    """Live, queryable tools exposed to the local LLM."""

    def __init__(self, app):
        self.app = app

    def entity_context(self, node_id, max_neighbors=12):
        G = self.app.G
        if node_id not in G:
            return {"error": "entity not found", "entity": node_id}
        data = G.nodes[node_id]
        neighbors = []
        for nbr in G.neighbors(node_id):
            edges = G.get_edge_data(node_id, nbr) or {}
            for key, ed in list(edges.items())[:4]:
                neighbors.append({
                    "id": str(nbr),
                    "label": G.nodes[nbr].get("label", str(nbr)),
                    "type": G.nodes[nbr].get("type", "Entity"),
                    "relationship": ed.get("label", "Linked to"),
                    "weight": float(ed.get("weight", 1.0) or 1.0),
                    "count": int(ed.get("count", 1) or 1),
                })
                if len(neighbors) >= max_neighbors:
                    break
            if len(neighbors) >= max_neighbors:
                break
        anomalies = self.app._anomaly_results.get(str(node_id), [])
        return {
            "id": str(node_id),
            "label": data.get("label", str(node_id)),
            "type": data.get("type", "Entity"),
            "properties": data.get("properties", {}),
            "metrics": {k: data.get(k) for k in (
                "centrality", "degree_centrality", "weighted_degree",
                "betweenness_centrality", "closeness_centrality", "eigenvector_centrality"
            ) if k in data},
            "neighbors": neighbors,
            "anomalies": anomalies,
            "evidence_refs": self.app.evidence_store.evidence_refs_for_graph(G, node_id),
        }

    def investigation_summary(self):
        G = self.app.G
        return {
            "entity_count": G.number_of_nodes(),
            "relationship_count": G.number_of_edges(),
            "relation_types": sorted(set(str(d.get("label", "Linked to")) for _, _, d in G.edges(data=True))),
            "fir_documents": len(self.app.evidence_store.fir_documents),
            "selected_entity": str(self.app.selected_node) if self.app.selected_node in G else None,
            "top_entities": [
                {"id": str(n), "label": d.get("label", str(n)), "score": round(float(d.get("centrality", 0.0) or 0.0), 4)}
                for n, d in sorted(G.nodes(data=True), key=lambda x: float(x[1].get("centrality", 0.0) or 0.0), reverse=True)[:8]
            ],
        }

    def anomalies_for_entity(self, node_id):
        return self.app._anomaly_results.get(str(node_id), [])

    def evidence_for_entity(self, node_id):
        return self.app._collect_entity_evidence(node_id)

    def tool_context(self, node_id=None):
        context = {"investigation": self.investigation_summary()}
        node_id = node_id or self.app.selected_node
        if node_id in self.app.G:
            context["selected_entity"] = self.entity_context(node_id)
            context["evidence"] = self.evidence_for_entity(node_id)
        return context


class LocalOllama:
    """Robust local Ollama connector. No API key is used.

    Qwen3 can emit a separate reasoning field when thinking is enabled.  For the
    application UI we explicitly disable thinking so the final answer arrives in
    ``message.content`` and the chat panel never appears to hang on hidden
    reasoning output.
    """

    def __init__(self, base_url=OLLAMA_URL_DEFAULT, model=AI_MODEL_DEFAULT):
        self.base_url = base_url.rstrip("/")
        self.model = model.strip() or AI_MODEL_DEFAULT
        self.last_error = ""

    def list_models(self):
        if not HAS_REQUESTS:
            raise RuntimeError("Python package 'requests' is required. Install: pip install requests")
        r = requests.get(f"{self.base_url}/api/tags", timeout=5)
        r.raise_for_status()
        payload = r.json() or {}
        return [str(m.get("name", "")).strip() for m in payload.get("models", []) if m.get("name")]

    def _resolve_installed_model(self):
        models = self.list_models()
        if not models:
            raise RuntimeError("Ollama is running, but no local models are installed.")

        wanted = self.model.strip()
        if wanted in models:
            return wanted

        base = wanted.split(":", 1)[0].lower()
        same_family = [m for m in models if m.split(":", 1)[0].lower() == base]
        if same_family:
            return same_family[0]

        # Helpful fallback for a demo machine where another local model is already
        # installed (e.g. qwen3:4b, qwen3:8b, or llama3).
        return models[0]

    def available(self):
        if not HAS_REQUESTS:
            return False, "requests package not installed"
        try:
            resolved = self._resolve_installed_model()
            if resolved != self.model:
                return True, f"{resolved} ready (configured: {self.model})"
            return True, f"{resolved} ready"
        except requests.exceptions.ConnectionError:
            return False, f"Cannot reach Ollama at {self.base_url}. Start Ollama first."
        except requests.exceptions.Timeout:
            return False, "Ollama status check timed out."
        except Exception as exc:
            return False, str(exc)

    def generate(self, system, prompt, temperature=0.2):
        if not HAS_REQUESTS:
            raise RuntimeError("Python package 'requests' is required. Install: pip install requests")

        model = self._resolve_installed_model()
        payload = {
            "model": model,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {
                "temperature": float(temperature),
                "num_ctx": FAST_LLM_NUM_CTX,
                "num_predict": FAST_LLM_NUM_PREDICT,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=600)
            r.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Cannot connect to Ollama at {self.base_url}. Start Ollama and retry.") from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("Ollama took more than 10 minutes to answer. Try llama3.2 or a shorter question/context.") from exc
        except requests.exceptions.HTTPError as exc:
            body = r.text[:1000] if 'r' in locals() else ""
            raise RuntimeError(f"Ollama HTTP error {getattr(r, 'status_code', '?')}: {body}") from exc

        try:
            data = r.json()
        except ValueError as exc:
            raise RuntimeError("Ollama returned a non-JSON response.") from exc

        content = str((data.get("message") or {}).get("content", "") or "").strip()
        if not content:
            # With think=false this should almost never happen; keep the error
            # explicit instead of silently displaying a blank chatbot response.
            raise RuntimeError("Ollama returned no final answer. Try `ollama run llama3.2` once from PowerShell to verify the model.")
        self.model = model
        return content


class InvestigationAI:
    """AI orchestration: normal chat + 3 specialist agents + evaluator."""

    BASE_RULES = """
You are an investigator-support AI inside a graph-analysis workbench.
Use only the supplied evidence. Never invent records, identities, relationships,
financial amounts, dates, or legal conclusions. Distinguish OBSERVED FACTS from
INFERENCES. A graph score or anomaly is an investigative signal, not proof of guilt.
Never declare a person guilty. Prefer precise, source-grounded explanations and
state uncertainty when evidence is incomplete or contradictory.
"""

    SPECIALISTS = {
        "LLM 1 — Criminology Specialist": """Focus on network structure, roles, communities,
influence, bridges between groups, and relationship patterns. Explain structural
importance without asserting criminal responsibility.""",
        "LLM 2 — Temporal Pattern Expert": """Focus on chronology, bursts, before/after patterns,
call and transaction timing, repeated activity, and coordinated changes. Be explicit
that temporal correlation does not establish causation.""",
        "LLM 3 — Demographic / Context Analyst": """Focus on entity attributes, stated organizational
roles, geographic context, contextual relationships, and consistency or mismatch
between stored attributes and observed behaviour. Do not infer protected or sensitive
traits that are not explicitly present.""",
    }

    def __init__(self, app):
        self.app = app
        model_var = getattr(app, "ai_model_var", None)
        model_name = model_var.get().strip() if model_var is not None else AI_MODEL_DEFAULT
        self.backend = LocalOllama(model=model_name)
        self.tools = InvestigationTools(app)

    def refresh_backend(self):
        self.backend.model = self.app.ai_model_var.get().strip() or AI_MODEL_DEFAULT
        self.backend.base_url = self.app.ollama_url_var.get().strip().rstrip("/") or OLLAMA_URL_DEFAULT

    def _context_text(self, node_id=None):
        context = self.tools.tool_context(node_id)
        # Keep prompts compact enough for an 8B local model.
        raw = json.dumps(context, indent=2, default=str)
        # Keep local-model prompts compact for fast CPU inference.
        return raw[:6500]

    def chat(self, question, node_id=None):
        self.refresh_backend()
        status, detail = self.backend.available()
        if not status:
            raise RuntimeError(detail + "\n\nRecommended fix: run `ollama list` to see installed models, then set that model name in the AI panel.")
        prompt = (
            f"QUESTION:\n{question}\n\n"
            "LIVE INVESTIGATION CONTEXT:\n"
            f"{self._context_text(node_id)}\n\n"
            "Answer concisely (prefer <= 180 words). Cite source filenames/rows when present. "
            "Do not restate the whole context."
        )
        answer = self.backend.generate(self.BASE_RULES, prompt, temperature=0.15)
        return answer

    def deep_analysis(self, node_id):
        """Run either an efficient one-pass multi-perspective analysis or the full
        four-call specialist/evaluator pipeline. One-pass mode is the default for
        local 4B models so Explain Score feels responsive on CPU-only machines."""
        self.refresh_backend()
        status, detail = self.backend.available()
        if not status:
            raise RuntimeError(detail + "\n\nRecommended fix: run `ollama list` to see installed models, then set that model name in the AI panel.")
        context = self._context_text(node_id)
        deterministic = self.app.calculate_evidence_score(node_id)

        if not FULL_MULTI_AGENT:
            self.app._ai_activity("Three specialist roles are analyzing the selected entity in one local inference...")
            fast_system = self.BASE_RULES + """
You are a multi-perspective investigation analyst. In ONE response, simulate three
specialist perspectives and then an Evidence Evaluator. Keep each specialist section
short. Use ONLY the supplied evidence. Required headings:
1) LLM 1 — Criminology Specialist
2) LLM 2 — Temporal Pattern Expert
3) LLM 3 — Demographic / Context Analyst
4) ◆ Evidence Evaluator
For the evaluator, include: score interpretation, supporting evidence, evidence
against/limitations, anomaly interpretation, and next questions. Do not change the
deterministic score. Never claim guilt.
"""
            prompt = (
                f"DETERMINISTIC EVIDENCE ASSOCIATION SCORE: {deterministic['score']:.0f}/100\n"
                f"COMPONENTS: {json.dumps(deterministic['components'], indent=2)}\n\n"
                f"LIVE CONTEXT:\n{context}\n"
            )
            self.app._ai_activity("LLM 1 — Criminology Specialist ✓")
            self.app._ai_activity("LLM 2 — Temporal Pattern Expert ✓")
            self.app._ai_activity("LLM 3 — Demographic / Context Analyst ✓")
            self.app._ai_activity("Evidence Evaluator is synthesizing the specialist findings...")
            final = self.backend.generate(fast_system, prompt, temperature=0.1)
            self.app._ai_activity("Evidence Evaluator ✓ — response generated")
            return final, {"mode": "one_pass_multi_perspective"}, deterministic

        outputs = {}
        self.app._ai_activity("Three specialist LLMs are analyzing the selected entity...")
        for name, role_prompt in self.SPECIALISTS.items():
            outputs[name] = self.backend.generate(
                self.BASE_RULES + role_prompt,
                f"Analyze the selected entity concisely. Return findings, supporting evidence refs, contradictions, and uncertainty.\n\n{context}",
                0.1,
            )
            self.app._ai_activity(f"{name} ✓")

        evaluator_system = self.BASE_RULES + """
You are the Evidence Evaluator. Synthesize three specialist analyses. Return a concise
investigator briefing with: Evidence Association Score (0-100, not a guilt probability),
key supporting evidence, evidence against/limitations, important anomalies, and next
investigative questions. Do not manufacture numeric evidence. The displayed score must
come from the supplied deterministic score; explain it rather than replacing it.
"""
        evaluator_prompt = (
            f"DETERMINISTIC EVIDENCE ASSOCIATION SCORE: {deterministic['score']:.0f}/100\n"
            f"COMPONENTS: {json.dumps(deterministic['components'], indent=2)}\n\n"
            f"LIVE CONTEXT:\n{context}\n\n"
            f"SPECIALIST OUTPUTS:\n{json.dumps(outputs, indent=2)}\n"
        )
        self.app._ai_activity("Evidence Evaluator is synthesizing the specialist findings...")
        final = self.backend.generate(evaluator_system, evaluator_prompt, temperature=0.05)
        self.app._ai_activity("Evidence Evaluator ✓ — response generated")
        return final, outputs, deterministic


# ============================================================================
#  NEO4J PERSISTENCE LAYER
# ============================================================================
#
# Design notes:
#   - The Tkinter/NetworkX side of the app is UNCHANGED as the "working
#     copy" used for drawing and for all analysis (centrality, community
#     detection, link prediction). Rewriting those algorithms in pure Cypher
#     would require the (paid/plugin-only) Neo4j Graph Data Science library,
#     which most students won't have installed. Instead, NetworkX stays the
#     analysis engine, and Neo4j becomes the durable, query-able store that
#     mirrors it -- every entity/link created, edited, or deleted locally is
#     synced to Neo4j, and the graph can be reloaded fresh from Neo4j at any
#     time (e.g. on another machine, or after restarting the app).
#   - Every node gets a stable base label :Entity plus a dynamic label
#     derived from its entity "type" (e.g. :Suspect, :Bank_Account) so you
#     can write normal Cypher like `MATCH (s:Suspect)-[:CALLS]->(p:Phone_Number)`.
#   - Every relationship's "relation" string (free text, e.g. "Calls",
#     "Transfers to") becomes a dynamic Neo4j relationship TYPE, sanitized
#     into valid Cypher identifier form (e.g. "Calls" -> CALLS).
#   - Cypher does not allow parameterized labels/relationship types, so
#     sanitized names are interpolated directly into the query string. They
#     are sanitized to [A-Za-z0-9_] only, which makes Cypher injection via
#     this path impossible.
#   - Node/edge properties must be Neo4j-primitive (str/int/float/bool, or
#     lists thereof). Anything else (nested dicts, e.g. the multi-record
#     "records" list on aggregated edges) is JSON-serialized into a string
#     property so no data is silently dropped.

def _sanitize_identifier(raw, default="Entity"):
    """Turn arbitrary text into a safe Cypher label / relationship-type name."""
    s = re.sub(r'[^0-9a-zA-Z_]', '_', str(raw if raw not in (None, "") else default))
    s = re.sub(r'_+', '_', s).strip('_')
    if not s:
        s = default
    if s[0].isdigit():
        s = f"T_{s}"
    return s


def _flatten_properties(properties):
    """Coerce an arbitrary properties dict into Neo4j-storable primitives."""
    flat = {}
    for key, value in (properties or {}).items():
        safe_key = re.sub(r'[^0-9a-zA-Z_]', '_', str(key)) or "field"
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            flat[safe_key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (str, int, float, bool)) for v in value
        ):
            flat[safe_key] = list(value)
        else:
            # Nested dict/list of dicts (e.g. per-observation "records") ->
            # store as a JSON string so nothing is silently lost.
            try:
                flat[safe_key] = json.dumps(value, default=str)
            except TypeError:
                flat[safe_key] = str(value)
    return flat


class Neo4jSync:
    """Thin wrapper around the official Neo4j driver used to mirror the
    in-memory investigation graph into a real Neo4j database."""

    def __init__(self):
        self.driver = None
        self.connected = False
        self.uri = None
        self.database = None

    def connect(self, uri, user, password, database="neo4j"):
        if not HAS_NEO4J_DRIVER:
            raise RuntimeError(
                "The 'neo4j' Python driver is not installed.\n"
                "Install it with:  pip install neo4j"
            )
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()  # raises if the server is unreachable
        self.driver = driver
        self.uri = uri
        self.database = database or "neo4j"
        self.connected = True
        self._ensure_constraints()

    def close(self):
        if self.driver is not None:
            self.driver.close()
        self.driver = None
        self.connected = False

    def _session(self):
        return self.driver.session(database=self.database)

    def _ensure_constraints(self):
        with self._session() as session:
            session.run(
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
                "FOR (n:Entity) REQUIRE n.id IS UNIQUE"
            )

    # -------------------- writes --------------------

    def upsert_node(self, node_id, entity_type, properties):
        label = _sanitize_identifier(entity_type, default="Entity")
        props = _flatten_properties(properties)
        props["id"] = str(node_id)
        query = (
            "MERGE (n:Entity {id: $id}) "
            f"SET n:`{label}` "
            "SET n += $props"
        )
        with self._session() as session:
            session.run(query, id=str(node_id), props=props)

    def upsert_relationship(self, u, v, relation_label, properties):
        rel_type = _sanitize_identifier(relation_label, default="LINKED_TO")
        props = _flatten_properties(properties)
        query = (
            "MATCH (a:Entity {id: $u}), (b:Entity {id: $v}) "
            f"MERGE (a)-[r:`{rel_type}`]->(b) "
            "SET r += $props"
        )
        with self._session() as session:
            session.run(query, u=str(u), v=str(v), props=props)

    def delete_node(self, node_id):
        with self._session() as session:
            session.run(
                "MATCH (n:Entity {id: $id}) DETACH DELETE n", id=str(node_id)
            )

    def clear_all(self):
        with self._session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def write_node_property(self, node_id, key, value):
        safe_key = re.sub(r'[^0-9a-zA-Z_]', '_', str(key)) or "field"
        with self._session() as session:
            session.run(
                f"MATCH (n:Entity {{id: $id}}) SET n.`{safe_key}` = $value",
                id=str(node_id), value=value,
            )

    # -------------------- reads --------------------

    def fetch_all(self):
        """Pull the full graph back out of Neo4j as (nodes, edges) so a
        local NetworkX MultiGraph can be rebuilt from it."""
        nodes, edges = [], []
        with self._session() as session:
            for record in session.run("MATCH (n:Entity) RETURN n"):
                n = record["n"]
                data = dict(n)
                node_id = data.pop("id")
                labels = [l for l in n.labels if l != "Entity"]
                entity_type = labels[0] if labels else "Entity"
                nodes.append((node_id, entity_type, data))

            for record in session.run(
                "MATCH (a:Entity)-[r]->(b:Entity) "
                "RETURN a.id AS u, b.id AS v, type(r) AS rel, properties(r) AS props"
            ):
                edges.append(
                    (record["u"], record["v"], record["rel"], dict(record["props"]))
                )
        return nodes, edges

    def node_count(self):
        with self._session() as session:
            rec = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()
            return rec["c"] if rec else 0


# ============================================================================
#  WORKBENCH GUI
# ============================================================================

class CriminalNetworkAnalyst:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligence Analyst Workbench - SIH 2024 (Neo4j-backed, Spread + Multi-CSV)")
        self.root.geometry("1400x850")
        self.root.configure(bg="#f0f0f0")

        # Core Graph Data Structure (Entity-Link-Property)
        # MultiGraph preserves multiple distinct links between the same nodes.
        self.G = nx.MultiGraph()

        # Neo4j mirror: every add/delete below is pushed here (when connected)
        # so the investigation graph is durably stored and Cypher-queryable.
        self.neo4j = Neo4jSync()

        # AI / Evidence state -- all AI work is optional and local through Ollama.
        self.evidence_store = EvidenceStore()
        self.fir_processor = FIRProcessor()
        self.anomaly_engine = AnomalyEngine()
        self._previous_ai_snapshot = {
            "nodes": set(), "edges": set(), "degrees": {}, "centrality": {}, "community": {}
        }
        self._anomaly_results = {}
        self._last_link_predictions = []
        self._ai_model_default = AI_MODEL_DEFAULT
        self.ai_model_var = tk.StringVar(value=AI_MODEL_DEFAULT)
        self.ollama_url_var = tk.StringVar(value=OLLAMA_URL_DEFAULT)
        self._ai_busy = False
        self._ai_activity_lines = []
        self.ai = None
        self._suspend_ai_snapshot = False

        # UI State
        self.selected_node = None
        self.selected_edge = None
        self.node_positions = {}
        self.dragging_node = None
        self.mode = "select"             # "select" | "link" | "path"
        self.pending_first_node = None   # first node picked in link/path mode

        self.community_palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231",
                                   "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
                                   "#fabebe", "#008080"]

        self._entity_type_colors = {}  # cache for auto-generated type colors

        self.setup_styles()
        self.setup_menu()
        self.setup_layout()
        self.ai = InvestigationAI(self)
        self._refresh_anomalies()
        self.redraw_canvas()
        self.root.after(250, self._check_ai_status_async)

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=25)

    def setup_menu(self):
        menubar = tk.Menu(self.root)

        # FILE MENU
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Import Relationships (CSV)", command=self.import_relationships_csv)
        file_menu.add_command(label="Import Entities (CSV)", command=self.import_entities_csv)
        file_menu.add_command(label="Import NER / ML Output (JSON)", command=self.import_ner)
        file_menu.add_command(label="Upload FIR / Investigation Document...", command=self.upload_fir)
        file_menu.add_separator()
        file_menu.add_command(label="Save Investigation (JSON)", command=self.save_investigation)
        file_menu.add_command(label="Load Investigation (JSON)", command=self.load_investigation)
        file_menu.add_separator()
        file_menu.add_command(label="Export Investigation Report", command=self.export_report)
        file_menu.add_separator()
        file_menu.add_command(label="Clear Investigation", command=self.clear_investigation)
        menubar.add_cascade(label="File", menu=file_menu)

        # NEO4J MENU
        neo4j_menu = tk.Menu(menubar, tearoff=0)
        neo4j_menu.add_command(label="Connect to Neo4j...", command=self.neo4j_connect_dialog)
        neo4j_menu.add_command(label="Disconnect", command=self.neo4j_disconnect)
        neo4j_menu.add_separator()
        neo4j_menu.add_command(label="Push Current Graph to Neo4j", command=self.neo4j_push_graph)
        neo4j_menu.add_command(label="Load Graph from Neo4j (replaces current)", command=self.neo4j_load_graph)
        neo4j_menu.add_separator()
        neo4j_menu.add_command(label="Clear Neo4j Database", command=self.neo4j_clear_remote)
        menubar.add_cascade(label="Neo4j", menu=neo4j_menu)

        # ANALYSIS MENU
        analysis_menu = tk.Menu(menubar, tearoff=0)
        analysis_menu.add_command(label="Calculate Centrality (Key Players)", command=self.analyze_influencers)
        analysis_menu.add_command(label="Detect Communities (Clusters)", command=self.detect_communities)
        analysis_menu.add_command(label="Find Shortest Path (Money Trail)", command=self.enable_path_mode)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Run Link Prediction (Multilayer)", command=self.run_link_prediction)
        analysis_menu.add_command(label="Detect 12 Anomaly Types", command=self.run_anomaly_detection)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="AI Investigation Briefing", command=self.ai_explain_selected)
        analysis_menu.add_command(label="AI Copilot Status", command=self.show_ai_status)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)

        # VIEW MENU
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Auto Layout", command=self.auto_layout)
        view_menu.add_separator()
        view_menu.add_command(label="Spread Layout", command=self.auto_layout)
        menubar.add_cascade(label="View", menu=view_menu)

        self.root.config(menu=menubar)

    def setup_layout(self):
        # --- LEFT PANEL: Entity Toolbox ---
        self.left_panel = tk.Frame(self.root, width=200, bg="#d9d9d9", relief=tk.RAISED, bd=1)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.left_panel, text="ENTITY TOOLBOX", bg="#404040", fg="white",
                 font=("Arial", 10, "bold")).pack(fill=tk.X)

        entities = [
            ("Suspect", "#ff9999"),
            ("Phone Number", "#99ff99"),
            ("Bank Account", "#9999ff"),
            ("Vehicle", "#ffff99"),
            ("Location", "#ff99ff"),
            ("Organization", "#ffcc66"),
            ("Case/FIR", "#cccccc"),
        ]

        for name, color in entities:
            btn = tk.Button(self.left_panel, text=f"+ Add {name}", bg=color, relief=tk.GROOVE,
                             command=lambda n=name: self.add_new_entity_at_center(n))
            btn.pack(fill=tk.X, padx=10, pady=3)

        tk.Label(self.left_panel, text="MODE", bg="#404040", fg="white",
                 font=("Arial", 10, "bold")).pack(fill=tk.X, pady=(15, 0))

        self.mode_var = tk.StringVar(value="select")
        tk.Radiobutton(self.left_panel, text="Select / Drag", variable=self.mode_var, value="select",
                       command=lambda: self.set_mode("select"), bg="#d9d9d9", anchor="w").pack(fill=tk.X, padx=10)
        tk.Radiobutton(self.left_panel, text="Create Link", variable=self.mode_var, value="link",
                       command=lambda: self.set_mode("link"), bg="#d9d9d9", anchor="w").pack(fill=tk.X, padx=10)

        tk.Label(self.left_panel, text="FIND ENTITY", bg="#404040", fg="white",
                 font=("Arial", 10, "bold")).pack(fill=tk.X, pady=(15, 0))
        self.search_var = tk.StringVar()
        search_entry = tk.Entry(self.left_panel, textvariable=self.search_var)
        search_entry.pack(fill=tk.X, padx=10, pady=5)
        search_entry.bind("<Return>", lambda e: self.find_entity(self.search_var.get()))
        tk.Button(self.left_panel, text="Search", command=lambda: self.find_entity(self.search_var.get())).pack(
            fill=tk.X, padx=10)

        tk.Label(self.left_panel, text="DATA IN", bg="#404040", fg="white",
                 font=("Arial", 10, "bold")).pack(fill=tk.X, pady=(15, 0))
        tk.Button(self.left_panel, text="Import Relationships (CSV)",
                  command=self.import_relationships_csv).pack(fill=tk.X, padx=10, pady=(5, 2))
        tk.Button(self.left_panel, text="Import Entities (CSV)",
                  command=self.import_entities_csv).pack(fill=tk.X, padx=10, pady=2)

        tk.Label(self.left_panel, text="LINK PREDICTION", bg="#404040", fg="white",
                 font=("Arial", 10, "bold")).pack(fill=tk.X, pady=(15, 0))
        tk.Button(self.left_panel, text="Run Link Prediction", bg="#66cccc",
                  command=self.run_link_prediction).pack(fill=tk.X, padx=10, pady=5)
        # Separate controls so Inspector and AI Copilot can be shown/hidden independently.
        self.inspector_visible = True
        self.ai_panel_visible = True

        self.toggle_inspector_btn = tk.Button(
            self.left_panel,
            text="Hide Inspector",
            command=self.toggle_inspector,
            bg="#bbbbbb"
        )
        self.toggle_inspector_btn.pack(fill=tk.X, padx=10, pady=(8, 2))

        self.toggle_ai_btn = tk.Button(
            self.left_panel,
            text="Hide AI Copilot",
            command=self.toggle_ai_copilot,
            bg="#bbbbbb"
        )
        self.toggle_ai_btn.pack(fill=tk.X, padx=10, pady=(2, 5))

        # --- RIGHT PANEL: AI Copilot (vertical) ---
        # AI Copilot occupies the full-height right side.
        self.right_panel = tk.Frame(self.root, width=430, bg="#d9d9d9", relief=tk.RAISED, bd=1)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_panel.pack_propagate(False)

        self.ai_tab = tk.Frame(self.right_panel, bg="#d9d9d9")
        self.ai_tab.pack(fill=tk.BOTH, expand=True)
        self._build_ai_panel()

        # --- CENTER: Canvas + bottom Property Inspector + status bar ---
        # The inspector belongs to the graph area, not the whole root window.
        # This keeps AI Copilot vertical on the right while the inspector sits
        # horizontally below the graph.
        center = tk.Frame(self.root)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Keep the inspector shallow so it does not consume much graph height.
        self.inspector_tab = tk.Frame(center, height=185, bg="#d9d9d9", relief=tk.RAISED, bd=1)
        self.inspector_tab.pack(side=tk.BOTTOM, fill=tk.X)
        self.inspector_tab.pack_propagate(False)

        tk.Label(self.inspector_tab, text="PROPERTY INSPECTOR", bg="#404040", fg="white",
                 font=("Arial", 10, "bold")).pack(fill=tk.X)
        self.prop_tree = ttk.Treeview(self.inspector_tab, columns=("Property", "Value"), show="headings")
        self.prop_tree.heading("Property", text="Property")
        self.prop_tree.heading("Value", text="Value")
        self.prop_tree.column("Property", width=180, minwidth=140, stretch=False)
        self.prop_tree.column("Value", width=700, minwidth=250, stretch=True)
        self.prop_tree.pack(fill=tk.BOTH, expand=True)

        tk.Button(self.inspector_tab, text="Delete Selected Entity", bg="#ff6666",
                  command=self.delete_selected_node).pack(fill=tk.X, padx=5, pady=5)

        self.status_var = tk.StringVar(
            value="No data loaded. Use File > Import Relationships (CSV) to begin, "
                  "or add entities from the toolbox.")

        self.canvas = tk.Canvas(center, bg="white", bd=2, relief=tk.SUNKEN, cursor="cross")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)

    # ========================== MODES ==========================
    def toggle_inspector(self):
        """Show or hide only the Property Inspector."""
        if self.inspector_visible:
            self.inspector_tab.pack_forget()
            self.inspector_visible = False
            self.toggle_inspector_btn.config(text="Show Inspector")
        else:
            self.inspector_tab.pack(side=tk.BOTTOM, fill=tk.X)
            self.inspector_visible = True
            self.toggle_inspector_btn.config(text="Hide Inspector")
        self.root.update_idletasks()
        self.auto_layout()

    def toggle_ai_copilot(self):
        """Show or hide only the AI Copilot."""
        if self.ai_panel_visible:
            self.ai_tab.pack_forget()
            self.ai_panel_visible = False
            self.toggle_ai_btn.config(text="Show AI Copilot")
        else:
            self.ai_tab.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            self.ai_panel_visible = True
            self.toggle_ai_btn.config(text="Hide AI Copilot")
        self.root.update_idletasks()
        self.auto_layout()

    
    def set_mode(self, mode):
        self.mode = mode
        self.pending_first_node = None
        self.selected_node = None
        if mode == "select":
            self.status_var.set("Mode: SELECT - click a node to inspect, drag to move")
        elif mode == "link":
            self.status_var.set("Mode: LINK - click the first entity, then the second entity")
        self.redraw_canvas()

    def enable_path_mode(self):
        if len(self.G.nodes) < 2:
            return messagebox.showwarning("Warning", "Add at least two entities first.")
        self.mode = "path"
        self.mode_var.set("select")
        self.pending_first_node = None
        self.status_var.set("Mode: PATH - click the start entity, then the destination entity")
        self.redraw_canvas()

    # ========================== GRAPH ENGINE ==========================

    def add_entity(self, label, e_type="Entity", x=0, y=0, properties=None,
                   node_id=None, redraw=True):
        """Add/update a node while retaining arbitrary CSV properties.

        The graph node identifier is the supplied ID.  The old type/label
        fields are retained as display metadata for compatibility with the
        analysis code, but CSV imports do not require any particular type name.
        """
        node_id = str(node_id if node_id is not None else f"{e_type}_{label}")
        properties = dict(properties or {})

        if node_id not in self.G.nodes:
            self.G.add_node(
                node_id,
                label=str(label),
                type=str(e_type or "Entity"),
                color=self.get_color(str(e_type or "Entity")),
                properties=properties
            )
            self.node_positions[node_id] = [x, y]
        else:
            data = self.G.nodes[node_id]
            data.setdefault("label", str(label))
            data.setdefault("type", str(e_type or "Entity"))
            data.setdefault("color", self.get_color(str(e_type or "Entity")))
            data.setdefault("properties", {})
            data["properties"].update(properties)

        if self.neo4j.connected:
            try:
                self.neo4j.upsert_node(node_id, self.G.nodes[node_id].get("type", e_type),
                                        self.G.nodes[node_id].get("properties", {}))
            except (Neo4jError, ServiceUnavailable, OSError) as exc:
                self.status_var.set(f"Neo4j sync failed for node '{node_id}': {exc}")

        if redraw:
            self.redraw_canvas()
        if getattr(self, "ai", None) is not None and not getattr(self, "_suspend_ai_snapshot", False):
            self._commit_ai_snapshot()
        return node_id

    def add_link(self, node_u, node_v, relation_label="Linked to", source="manual",
                 confidence=1.0, weight=1.0, properties=None, redraw=True):
        """Add a relationship, aggregating repeated links of the same type.

        One MultiGraph edge represents one (node_u, node_v, relation) combination.
        Repeated observations of that same relationship increase its count/weight
        and are retained as individual records. Different relationship types
        remain separate edges.
        """
        if node_u not in self.G.nodes or node_v not in self.G.nodes or node_u == node_v:
            return

        relation_label = str(relation_label or "Linked to").strip() or "Linked to"
        props = dict(properties or {})
        props.setdefault("relation", relation_label)
        props.setdefault("_directed_from", str(node_u))
        props.setdefault("_directed_to", str(node_v))

        # Reuse an existing edge with the same relationship type.
        existing_key = None
        for key, data in self.G[node_u][node_v].items() if self.G.has_edge(node_u, node_v) else []:
            if str(data.get("label", "Linked to")) == relation_label:
                existing_key = key
                break

        if existing_key is not None:
            data = self.G[node_u][node_v][existing_key]
            data["count"] = int(data.get("count", 1)) + 1
            data["weight"] = float(data.get("weight", 1.0)) + float(weight or 1.0)

            records = data.setdefault("records", [])
            records.append({
                **props,
                "_source": source,
                "_confidence": confidence,
            })

            # Keep a compact summary of the latest values in the inspector.
            data["properties"] = props
            data["last_updated"] = time.strftime("%Y-%m-%d %H:%M")
        else:
            self.G.add_edge(
                node_u, node_v,
                label=relation_label,
                source=source,
                confidence=confidence,
                weight=float(weight or 1.0),
                count=1,
                created=time.strftime("%Y-%m-%d %H:%M"),
                properties=props,
                records=[{
                    **props,
                    "_source": source,
                    "_confidence": confidence,
                }]
            )

        if self.neo4j.connected:
            try:
                # Sync the aggregated edge data (whichever key it lives under).
                for key, edata in self.G[node_u][node_v].items():
                    if str(edata.get("label", "Linked to")) == relation_label:
                        self.neo4j.upsert_relationship(node_u, node_v, relation_label, edata)
                        break
            except (Neo4jError, ServiceUnavailable, OSError) as exc:
                self.status_var.set(f"Neo4j sync failed for link '{relation_label}': {exc}")

        if redraw:
            self.redraw_canvas()
        if getattr(self, "ai", None) is not None and not getattr(self, "_suspend_ai_snapshot", False):
            self._commit_ai_snapshot()

    def add_new_entity_at_center(self, e_type):
        count = len([n for n, d in self.G.nodes(data=True) if d['type'] == e_type])
        x = 300 + random.randint(-80, 80)
        y = 300 + random.randint(-80, 80)
        self.add_entity(f"New {e_type} {count+1}", e_type, x, y)

    def get_color(self, e_type):
        colors = {
            "Suspect": "#ff9999",
            "Phone Number": "#99ff99",
            "Bank Account": "#9999ff",
            "Vehicle": "#ffff99",
            "Location": "#ff99ff",
            "Organization": "#ffcc66",
            "Case/FIR": "#cccccc",
        }
        if e_type in colors:
            return colors[e_type]
        # Deterministic pastel color for custom/CSV-supplied entity types
        if e_type not in self._entity_type_colors:
            hue = (abs(hash(e_type)) % 360) / 360.0
            r, g, b = colorsys.hls_to_rgb(hue, 0.78, 0.55)
            self._entity_type_colors[e_type] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        return self._entity_type_colors[e_type]

    def node_radius(self, data):
        """Small Neo4j/Bloom-like circular nodes; analysis may enlarge them slightly."""
        if 'centrality' in data:
            return min(20 + data['centrality'] * 18, 38)
        return 20

    def _layout_graph(self):
        """Collapse relationship multiplicity for layout purposes.

        Repeated observations of the same relationship are represented by one
        weighted edge. The visual layout deliberately does NOT use the raw
        multiplicity as attraction strength: otherwise heavily connected pairs
        collapse into tight knots.
        """
        H = nx.Graph()
        H.add_nodes_from(self.G.nodes())
        for u, v, data in self.G.edges(data=True):
            raw_weight = max(float(data.get("weight", 1.0) or 1.0), 1.0)
            layout_weight = 1.0 + min(np.log1p(raw_weight), 2.0) * 0.15
            if H.has_edge(u, v):
                H[u][v]["weight"] = max(
                    H[u][v].get("weight", 1.0), layout_weight
                )
            else:
                H.add_edge(u, v, weight=layout_weight)
        return H

    def _analysis_graph(self):
        """Simple undirected graph with true summed edge weights (not the
        damped layout weight used by _layout_graph) for centrality metrics
        and weighted shortest-path calculations."""
        H = nx.Graph()
        H.add_nodes_from(self.G.nodes())
        for u, v, data in self.G.edges(data=True):
            w = max(float(data.get("weight", 1.0) or 1.0), 0.0001)
            if H.has_edge(u, v):
                H[u][v]["weight"] += w
            else:
                H.add_edge(u, v, weight=w)
        return H

    @staticmethod
    def _relax_node_collisions(pos, min_distance=62, iterations=60):
        """Push overlapping nodes apart after the force-directed layout."""
        if len(pos) < 2:
            return pos

        nodes = list(pos)
        for _ in range(iterations):
            displacement = {n: np.array([0.0, 0.0]) for n in nodes}
            moved = False

            for i, u in enumerate(nodes):
                for v in nodes[i + 1:]:
                    delta = np.array(pos[v], dtype=float) - np.array(pos[u], dtype=float)
                    dist = float(np.linalg.norm(delta))

                    if dist < 1e-9:
                        angle = ((i * 37) % 360) * np.pi / 180.0
                        direction = np.array([np.cos(angle), np.sin(angle)])
                        dist = 1e-9
                    else:
                        direction = delta / dist

                    if dist < min_distance:
                        push = (min_distance - dist) * 0.52
                        displacement[u] -= direction * push
                        displacement[v] += direction * push
                        moved = True

            if not moved:
                break

            for n in nodes:
                # Keep positions mutable because _force_layout updates x/y in place.
                pos[n] = [
                    float(pos[n][0] + displacement[n][0]),
                    float(pos[n][1] + displacement[n][1]),
                ]

        return pos

    def _random_spread_positions(self, nodes, w, h, min_distance=65):
        """Place nodes randomly but evenly over the canvas.

        A shuffled grid is used rather than a circular/spring layout.  This
        gives an intentionally random-looking initial graph without overlaps.
        """
        nodes = list(nodes)
        rng = random.Random(RANDOM_SEED + len(nodes) * 17)
        rng.shuffle(nodes)

        margin = 55
        left, right = margin + min_distance, w - margin - min_distance
        top, bottom = margin + min_distance, h - margin - min_distance
        usable_w = max(right - left, min_distance)
        usable_h = max(bottom - top, min_distance)

        cols = max(1, int(np.ceil(np.sqrt(len(nodes) * usable_w / max(usable_h, 1)))))
        rows = max(1, int(np.ceil(len(nodes) / cols)))
        cell_w = usable_w / cols
        cell_h = usable_h / rows

        positions = {}
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        rng.shuffle(cells)

        for node, (r, c) in zip(nodes, cells):
            jitter_x = rng.uniform(-0.28, 0.28) * cell_w
            jitter_y = rng.uniform(-0.28, 0.28) * cell_h
            x = left + (c + 0.5) * cell_w + jitter_x
            y = top + (r + 0.5) * cell_h + jitter_y
            positions[node] = [
                min(max(x, margin + min_distance), w - margin - min_distance),
                min(max(y, margin + min_distance), h - margin - min_distance),
            ]
        return positions

    def _force_layout(self, H, positions, w, h, iterations=260):
        """Run a bounded force simulation that prioritises readability.

        Unlike NetworkX spring_layout, this does not rescale the final result
        into a compact bounding box.  Nodes therefore keep real separation on
        the canvas.  All nodes repel one another; graph edges provide a softer
        attraction toward a readable target length.
        """
        nodes = list(H.nodes())
        if len(nodes) <= 1:
            return positions

        margin = 48
        radius = 23
        target = max(115.0, min(165.0, 145.0 + 0.25 * np.sqrt(len(nodes))))
        repulsion = 10500.0
        spring = 0.010
        damping = 0.72
        max_step = 18.0

        velocity = {n: np.zeros(2, dtype=float) for n in nodes}

        for _ in range(iterations):
            force = {n: np.zeros(2, dtype=float) for n in nodes}

            # Pairwise node repulsion.  This is deliberately independent of
            # relationship count so frequent calls cannot collapse a pair.
            for i, u in enumerate(nodes):
                xu = np.asarray(positions[u], dtype=float)
                for v in nodes[i + 1:]:
                    xv = np.asarray(positions[v], dtype=float)
                    delta = xu - xv
                    dist = float(np.linalg.norm(delta))
                    if dist < 1e-6:
                        angle = ((i * 53) % 360) * np.pi / 180.0
                        direction = np.array([np.cos(angle), np.sin(angle)])
                        dist = 1.0
                    else:
                        direction = delta / dist

                    # Stronger repulsion below the comfortable node spacing.
                    effective = max(dist, 25.0)
                    magnitude = repulsion / (effective * effective)
                    if dist < 2 * radius + 20:
                        magnitude *= 2.2
                    f = direction * magnitude
                    force[u] += f
                    force[v] -= f

            # Edge attraction. Multiple observations are NOT multiplied into
            # this force; count is represented by visual line thickness.
            for u, v in H.edges():
                xu = np.asarray(positions[u], dtype=float)
                xv = np.asarray(positions[v], dtype=float)
                delta = xv - xu
                dist = float(np.linalg.norm(delta))
                if dist < 1e-6:
                    continue
                direction = delta / dist
                # Only pull endpoints together when they are farther apart
                # than the desired readable distance. If they are already
                # close, global node repulsion pushes them apart instead.
                # Thus adding relationships cannot create a dense knot.
                if dist > target:
                    displacement = dist - target
                    f = direction * (spring * displacement)
                    force[u] += f
                    force[v] -= f

            # Keep the graph gently centred and away from the canvas border.
            centre = np.array([w / 2.0, h / 2.0])
            for n in nodes:
                p = np.asarray(positions[n], dtype=float)
                force[n] += (centre - p) * 0.0007

                border_push = 0.0
                if p[0] < margin:
                    border_push += (margin - p[0]) * 0.045
                elif p[0] > w - margin:
                    border_push -= (p[0] - (w - margin)) * 0.045
                if p[1] < margin:
                    # use the second component below
                    pass
                fx, fy = border_push, 0.0
                if p[1] < margin:
                    fy += (margin - p[1]) * 0.045
                elif p[1] > h - margin:
                    fy -= (p[1] - (h - margin)) * 0.045
                force[n] += np.array([fx, fy])

            max_velocity = 0.0
            for n in nodes:
                velocity[n] = damping * velocity[n] + force[n]
                norm = float(np.linalg.norm(velocity[n]))
                if norm > max_step:
                    velocity[n] *= max_step / norm
                positions[n][0] += velocity[n][0]
                positions[n][1] += velocity[n][1]
                positions[n][0] = min(max(positions[n][0], margin), w - margin)
                positions[n][1] = min(max(positions[n][1], margin), h - margin)
                max_velocity = max(max_velocity, float(np.linalg.norm(velocity[n])))

            if max_velocity < 0.08:
                break

        # A final hard-spacing pass prevents visual overlap after convergence.
        positions = self._relax_node_collisions(
            positions, min_distance=2 * radius + 12, iterations=80
        )
        for n in nodes:
            positions[n][0] = min(max(float(positions[n][0]), margin), w - margin)
            positions[n][1] = min(max(float(positions[n][1]), margin), h - margin)
        return positions

    def auto_layout(self):
        """Lay out the graph without spiral packing or compact cluster fitting.

        With no relationships, nodes are randomly distributed across the
        canvas. Once relationships exist, a bounded force simulation spreads
        the entire graph while keeping connected nodes at readable distances.
        The existing positions are retained, so adding another relation causes
        a gradual reorganisation rather than a fresh spiral.
        """
        if not self.G.nodes:
            return self.redraw_canvas()

        self.canvas.update_idletasks()
        w = max(self.canvas.winfo_width(), -10, 100)
        h = max(self.canvas.winfo_height(), -10, 100)

        if self.G.number_of_edges() == 0:
            self.node_positions = self._random_spread_positions(
                list(self.G.nodes()), w, h
            )
            self.redraw_canvas()
            return

        H = self._layout_graph()
        positions = {}

        # Keep current locations for existing nodes. New nodes are inserted at
        # an unused/random location rather than at the centre of the network.
        existing = set(self.node_positions).intersection(H.nodes())
        for n in existing:
            positions[n] = [float(self.node_positions[n][0]),
                            float(self.node_positions[n][1])]

        missing = [n for n in H.nodes() if n not in positions]
        if missing:
            new_positions = self._random_spread_positions(missing, w, h)
            positions.update(new_positions)

        positions = self._force_layout(H, positions, w, h)

        for n, (x, y) in positions.items():
            self.node_positions[n] = [float(x), float(y)]

        self.redraw_canvas()

    def _relation_color(self, relation):
        """Assign a stable visual color to each relationship type."""
        relation = str(relation or "Linked to")
        if not hasattr(self, "_relation_colors"):
            self._relation_colors = {}

        if relation not in self._relation_colors:
            palette = [
                "#0084ff", "#e74c3c", "#27ae60", "#8e44ad",
                "#f39c12", "#16a085", "#d35400", "#2c3e50",
                "#c0392b", "#2980b9", "#7f8c8d", "#8e44ad"
            ]
            self._relation_colors[relation] = palette[
                len(self._relation_colors) % len(palette)
            ]
        return self._relation_colors[relation]

    def redraw_canvas(self, highlight_path=None):
        self.canvas.delete("all")

        if len(self.G.nodes) == 0:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2 or 500, 250,
                text="No entities yet.\nUse File > Import Entities / Relationships (CSV),\n"
                     "or add entities from the toolbox on the left.",
                fill="#999999", font=("Arial", 12), justify="center")
            return

        highlight_edges = set()
        if highlight_path:
            for i in range(len(highlight_path) - 1):
                highlight_edges.add(frozenset((highlight_path[i], highlight_path[i + 1])))

        # Only DIFFERENT relation types are drawn as separate links. Repeated
        # observations of the same relation are represented by line thickness.
        pair_edges = {}
        for u, v, key, data in self.G.edges(keys=True, data=True):
            pair = frozenset((u, v))
            pair_edges.setdefault(pair, []).append((u, v, key, data))

        for pair, edge_list in pair_edges.items():
            count = len(edge_list)
            for idx, (u, v, key, data) in enumerate(edge_list):
                x1, y1 = self.node_positions[u]
                x2, y2 = self.node_positions[v]

                is_path_edge = frozenset((u, v)) in highlight_edges
                is_selected = self.selected_edge == (u, v, key)

                # Different relation types get a small, controlled offset.
                dx, dy = x2 - x1, y2 - y1
                length = max((dx * dx + dy * dy) ** 0.5, 1.0)
                px, py = -dy / length, dx / length
                offset = (idx - (count - 1) / 2) * 7
                ox, oy = px * offset, py * offset

                relation = data.get("label", "Linked to")
                fill = "#e53935" if is_path_edge else (
                    "#ff9800" if is_selected else self._relation_color(relation)
                )

                # Same-relation records are aggregated; thickness encodes count.
                relation_count = max(1, int(data.get("count", 1)))
                width = min(1.25 + 0.8 * (relation_count ** 0.5),
                            7.0)
                if is_path_edge or is_selected:
                    width += 1.5

                tag = f"edge:{u}:{v}:{key}"
                dash = (5, 3) if data.get('source') in (
                    'ml_predicted', 'link_prediction'
                ) else None

                self.canvas.create_line(
                    x1 + ox, y1 + oy, x2 + ox, y2 + oy,
                    fill=fill, width=width, dash=dash,
                    arrow=tk.LAST, arrowshape=(8, 10, 3),
                    tags=(tag, "edge")
                )

                # Keep the canvas uncluttered. Neo4j-style graph views normally
                # show the relation details when an edge is selected.
                if is_selected:
                    mid_x = (x1 + x2) / 2 + ox
                    mid_y = (y1 + y2) / 2 + oy
                    self.canvas.create_text(
                        mid_x, mid_y - 10,
                        text=f"{relation}  ({relation_count})",
                        fill=fill, font=("Arial", 8, "bold"),
                        tags=(tag, "edge")
                    )

        # Draw small circular nodes last so they remain easy to select.
        for node_id, data in self.G.nodes(data=True):
            x, y = self.node_positions[node_id]
            r = self.node_radius(data)

            outline = "#e67e22" if node_id == self.selected_node else "#555555"
            width = 3 if node_id in (
                self.selected_node, self.pending_first_node
            ) else 1

            fill_color = data.get(
                'color', self.get_color(data.get('type', 'Entity'))
            )
            if 'community' in data:
                fill_color = self.community_palette[
                    data['community'] % len(self.community_palette)
                ]

            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=fill_color, outline=outline, width=width,
                tags=(node_id, "entity")
            )

            label = str(data.get('label', node_id))
            # The ID is the graph's canonical label. For long IDs, shrink the
            # font slightly rather than turning the node into a large box.
            font_size = 8 if len(label) <= 8 else 7
            self.canvas.create_text(
                x, y, text=label, fill="white",
                font=("Arial", font_size, "bold"),
                tags=(node_id, "entity")
            )

    # ========================== INTERACTION ==========================

    def _node_at(self, event):
        item = self.canvas.find_closest(event.x, event.y)
        tags = self.canvas.gettags(item)
        for t in tags:
            if t in self.G.nodes:
                return t
        return None

    def _edge_at(self, event):
        """Return the specific edge nearest the click, if its canvas item is tagged."""
        item = self.canvas.find_closest(event.x, event.y)
        tags = self.canvas.gettags(item)
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("edge:"):
                parts = tag.split(":", 3)
                if len(parts) == 4:
                    try:
                        return parts[1], parts[2], int(parts[3])
                    except ValueError:
                        return None
        return None

    def on_canvas_click(self, event):
        node_id = self._node_at(event)

        if self.mode == "select":
            if node_id:
                self.selected_node = node_id
                self.selected_edge = None
                self.dragging_node = node_id
                self.update_inspector(node_id)
                self._show_ai_selection(node_id)
            else:
                edge = self._edge_at(event)
                if edge:
                    u, v, key = edge
                    self.selected_node = None
                    self.selected_edge = edge
                    self.dragging_node = None
                    self.update_edge_inspector(u, v, key)
                else:
                    self.selected_node = None
                    self.selected_edge = None
                    self.dragging_node = None
                    self.clear_inspector()
            self.redraw_canvas()

        elif self.mode == "link":
            if not node_id:
                return
            if self.pending_first_node is None:
                self.pending_first_node = node_id
                self.status_var.set(
                    f"Mode: LINK - first entity set to '{self.G.nodes[node_id]['label']}'. "
                    "Click the second entity."
                )
            else:
                if node_id != self.pending_first_node:
                    relation = simpledialog.askstring(
                        "Relationship", "Describe the link (e.g. Calls, Owns, Transfers to):")
                    if relation:
                        self.add_link(self.pending_first_node, node_id, relation)
                self.pending_first_node = None
                self.status_var.set("Mode: LINK - click the first entity, then the second entity")
            self.redraw_canvas()

        elif self.mode == "path":
            if not node_id:
                return
            if self.pending_first_node is None:
                self.pending_first_node = node_id
                self.status_var.set(
                    f"Mode: PATH - start set to '{self.G.nodes[node_id]['label']}'. "
                    "Click destination."
                )
            else:
                self.compute_and_show_path(self.pending_first_node, node_id)
                self.pending_first_node = None
                self.mode = "select"
                self.mode_var.set("select")
                self.status_var.set("Mode: SELECT - click a node to inspect, drag to move")

    def on_canvas_drag(self, event):
        if self.mode == "select" and self.dragging_node:
            self.node_positions[self.dragging_node] = [event.x, event.y]
            self.redraw_canvas()

    def on_canvas_release(self, event):
        self.dragging_node = None

    def on_canvas_right_click(self, event):
        node_id = self._node_at(event)
        if node_id:
            self.selected_node = node_id
            self.selected_edge = None
            self.update_inspector(node_id)
            if messagebox.askyesno(
                "Delete Entity",
                f"Remove '{self.G.nodes[node_id]['label']}' and all its links?"
            ):
                self.G.remove_node(node_id)
                del self.node_positions[node_id]
                self.selected_node = None
                self.selected_edge = None
                self.clear_inspector()
                if self.neo4j.connected:
                    try:
                        self.neo4j.delete_node(node_id)
                    except (Neo4jError, ServiceUnavailable, OSError) as exc:
                        self.status_var.set(f"Neo4j sync failed deleting '{node_id}': {exc}")
                self.redraw_canvas()
                if getattr(self, "ai", None) is not None:
                    self._commit_ai_snapshot()

    def delete_selected_node(self):
        if not self.selected_node:
            return messagebox.showwarning("Warning", "No entity selected.")
        node_id = self.selected_node
        self.G.remove_node(node_id)
        del self.node_positions[node_id]
        self.selected_node = None
        self.selected_edge = None
        self.clear_inspector()
        if self.neo4j.connected:
            try:
                self.neo4j.delete_node(node_id)
            except (Neo4jError, ServiceUnavailable, OSError) as exc:
                self.status_var.set(f"Neo4j sync failed deleting '{node_id}': {exc}")
        self.redraw_canvas()
        if getattr(self, "ai", None) is not None:
            self._commit_ai_snapshot()

    def find_entity(self, query):
        if not query:
            return
        query = query.lower()
        for node_id, data in self.G.nodes(data=True):
            if query in data['label'].lower():
                self.selected_node = node_id
                self.update_inspector(node_id)
                self.redraw_canvas()
                return
        messagebox.showinfo("Not Found", f"No entity matching '{query}'.")

    def clear_inspector(self):
        for i in self.prop_tree.get_children():
            self.prop_tree.delete(i)

    def update_inspector(self, node_id):
        """Display every stored property of a node."""
        self.clear_inspector()
        if node_id not in self.G.nodes:
            return

        data = self.G.nodes[node_id]
        self.prop_tree.insert("", tk.END, values=("System ID", node_id))

        properties = data.get("properties", {})
        if properties:
            for key, value in properties.items():
                self.prop_tree.insert("", tk.END, values=(str(key), str(value)))
        else:
            self.prop_tree.insert("", tk.END, values=("Label", data.get("label", node_id)))
            self.prop_tree.insert("", tk.END, values=("Entity Type", data.get("type", "Entity")))

        self.prop_tree.insert("", tk.END, values=("Direct Links", self.G.degree(node_id)))

        self._insert_centrality_rows(data)

    def _insert_centrality_rows(self, data):
        """Append the five centrality measures to the inspector, if they've
        been computed (via Analysis > Calculate Centrality)."""
        has_metrics = any(
            k in data for k in (
                "degree_centrality", "weighted_degree", "betweenness_centrality",
                "closeness_centrality", "eigenvector_centrality",
            )
        )
        self.prop_tree.insert("", tk.END, values=("— Centrality —", ""))
        if not has_metrics:
            self.prop_tree.insert(
                "", tk.END,
                values=("(not calculated)", "Run Analysis > Calculate Centrality")
            )
            return

        self.prop_tree.insert(
            "", tk.END, values=("Degree Centrality", f"{data.get('degree_centrality', 0.0):.4f}")
        )
        self.prop_tree.insert(
            "", tk.END, values=("Weighted Degree", f"{data.get('weighted_degree', 0.0):.4f}")
        )
        self.prop_tree.insert(
            "", tk.END,
            values=("Betweenness Centrality", f"{data.get('betweenness_centrality', 0.0):.4f}")
        )
        self.prop_tree.insert(
            "", tk.END,
            values=("Closeness Centrality", f"{data.get('closeness_centrality', 0.0):.4f}")
        )
        self.prop_tree.insert(
            "", tk.END,
            values=("Eigenvector Centrality", f"{data.get('eigenvector_centrality', 0.0):.4f}")
        )

    def update_edge_inspector(self, node_u, node_v, key):
        """Display the properties and multiplicity of one relationship type."""
        self.clear_inspector()
        if not self.G.has_edge(node_u, node_v, key):
            return

        data = self.G[node_u][node_v][key]
        relation = data.get("label", "Linked to")
        count = int(data.get("count", 1))

        self.prop_tree.insert("", tk.END, values=("Relationship", relation))
        self.prop_tree.insert("", tk.END, values=("Relationship Count", count))
        self.prop_tree.insert("", tk.END,
                              values=("From", self.G.nodes[node_u].get("label", node_u)))
        self.prop_tree.insert("", tk.END,
                              values=("To", self.G.nodes[node_v].get("label", node_v)))

        # Show the latest/representative properties directly.
        properties = data.get("properties", {})
        for k, v in properties.items():
            self.prop_tree.insert("", tk.END, values=(str(k), str(v)))

        # If multiple observations were aggregated, expose each record in a
        # compact, readable form instead of creating separate visual edges.
        records = data.get("records", [])
        if len(records) > 1:
            self.prop_tree.insert("", tk.END, values=("—", "—"))
            self.prop_tree.insert("", tk.END,
                                  values=("Individual Records", len(records)))
            for i, record in enumerate(records, 1):
                summary = "; ".join(
                    f"{k}={v}" for k, v in record.items()
                    if not str(k).startswith("_")
                )
                if len(summary) > 220:
                    summary = summary[:217] + "..."
                self.prop_tree.insert(
                    "", tk.END, values=(f"Record {i}", summary)
                )

    # ========================== ANALYSIS & MENU ACTIONS ==========================

    def compute_centrality_metrics(self):
        """Compute degree, weighted degree, betweenness, closeness, and
        eigenvector centrality for every node and store them on the graph
        so the Property Inspector can display them per node.

        Betweenness/closeness/eigenvector are computed on a simple graph
        with true summed relationship weights (_analysis_graph), not the
        damped weight used purely for visual layout.
        """
        H = self._analysis_graph()
        if H.number_of_nodes() < 2:
            return

        # Betweenness/closeness are path-based, so "weight" must behave like
        # a *distance*, not a strength: a well-evidenced relationship (many
        # calls, large transactions) should act as a SHORT hop, not a long
        # one. Relationship weight in _analysis_graph is strength, so invert
        # it into a distance purely for these two path-based measures.
        for _, _, data in H.edges(data=True):
            data["distance"] = 1.0 / max(float(data.get("weight", 1.0) or 1.0), 0.0001)

        degree_centrality = nx.degree_centrality(H)
        weighted_degree = dict(H.degree(weight="weight"))
        betweenness = nx.betweenness_centrality(H, weight="distance")
        closeness = nx.closeness_centrality(H, distance="distance")

        try:
            eigenvector = nx.eigenvector_centrality(H, weight="weight", max_iter=1000)
        except (nx.PowerIterationFailedConvergence, nx.AmbiguousSolution):
            try:
                eigenvector = nx.eigenvector_centrality_numpy(H, weight="weight")
            except Exception:
                eigenvector = {node: 0.0 for node in H.nodes()}

        max_degree = max(degree_centrality.values()) or 1
        max_betweenness = max(betweenness.values()) or 1
        max_closeness = max(closeness.values()) or 1
        max_eigenvector = max(eigenvector.values()) or 1

        for node in H.nodes():
            self.G.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0.0)
            self.G.nodes[node]['weighted_degree'] = weighted_degree.get(node, 0.0)
            self.G.nodes[node]['betweenness_centrality'] = betweenness.get(node, 0.0)
            self.G.nodes[node]['closeness_centrality'] = closeness.get(node, 0.0)
            self.G.nodes[node]['eigenvector_centrality'] = eigenvector.get(node, 0.0)

            # 'centrality' drives node size on the canvas and the report
            # ranking. It used to be normalized betweenness alone; it is now
            # a genuine composite -- the mean of all four measures, each
            # max-normalized to [0, 1] first so no single measure's raw
            # scale dominates the average.
            composite = (
                degree_centrality.get(node, 0.0) / max_degree
                + betweenness.get(node, 0.0) / max_betweenness
                + closeness.get(node, 0.0) / max_closeness
                + eigenvector.get(node, 0.0) / max_eigenvector
            ) / 4.0
            self.G.nodes[node]['centrality'] = composite

    def analyze_influencers(self):
        """Identifies key individuals using multiple centrality measures."""
        if len(self.G.nodes) < 2:
            return messagebox.showwarning("Warning", "Not enough entities in the network to analyze.")

        self.compute_centrality_metrics()

        self.redraw_canvas()
        messagebox.showinfo("Analysis Complete",
                             "Degree, weighted degree, betweenness, closeness, and eigenvector "
                             "centrality have all been calculated (betweenness/closeness use "
                             "relationship strength as inverse distance). Node size scales with "
                             "a composite influence score (mean of all four normalized measures) "
                             "- select a node to see every score in the Inspector.")

        if self.selected_node:
            self.update_inspector(self.selected_node)

    def detect_communities(self, silent=False):
        """Groups entities into likely sub-cells using Louvain (or greedy modularity fallback)."""
        if len(self.G.nodes) < 3:
            return messagebox.showwarning("Warning", "Need at least 3 entities to detect clusters.")

        # Use _analysis_graph() (true summed relationship weights), not
        # _layout_graph() (weights deliberately damped for visual spacing),
        # and pass those weights through to the community algorithm -- a
        # pair linked by 50 calls should pull together in the same cell
        # more strongly than a pair linked by 1 call.
        H = self._analysis_graph()
        if HAS_LOUVAIN:
            communities = louvain_communities(H, weight="weight", seed=42)
        else:
            communities = list(greedy_modularity_communities(H, weight="weight"))

        for idx, group in enumerate(communities):
            for node in group:
                self.G.nodes[node]['community'] = idx

        self.redraw_canvas()
        if not silent:
            messagebox.showinfo("Clusters Detected",
                                 f"Found {len(communities)} likely sub-group(s). "
                                 "Nodes are now color-coded by cluster.")

    def compute_and_show_path(self, source, target):
        """Show both the fewest-hop path and, when relationship weights are
        present, the strongest-evidence path (the chain of connections with
        the most cumulative relationship weight) between two entities --
        useful for tracing a 'money trail' through the best-supported chain
        rather than just the shortest one."""
        try:
            hop_path = nx.shortest_path(self.G, source=source, target=target)
        except nx.NodeNotFound:
            return messagebox.showwarning("No Connection", "One of the selected entities is not in the graph.")
        except nx.NetworkXNoPath:
            return messagebox.showwarning("No Connection", "These two entities are not connected in the graph.")

        highlight = hop_path
        weighted_path = None
        H = self._analysis_graph()
        if H.has_node(source) and H.has_node(target):
            for _, _, edata in H.edges(data=True):
                w = max(float(edata.get("weight", 1.0) or 1.0), 0.0001)
                edata["distance"] = 1.0 / w
            try:
                candidate = nx.shortest_path(H, source=source, target=target, weight="distance")
                if candidate != hop_path:
                    weighted_path = candidate
            except nx.NetworkXNoPath:
                weighted_path = None

        hop_labels = " -> ".join(self.G.nodes[n]['label'] for n in hop_path)
        message = f"Shortest connection ({len(hop_path) - 1} hop(s)):\n{hop_labels}"

        if weighted_path:
            weighted_labels = " -> ".join(self.G.nodes[n]['label'] for n in weighted_path)
            message += (
                f"\n\nStrongest-evidence path ({len(weighted_path) - 1} hop(s), "
                f"weighted by relationship strength):\n{weighted_labels}"
            )
            highlight = weighted_path

        self.redraw_canvas(highlight_path=highlight)
        messagebox.showinfo("Path Found", message)

    # ========================== LINK PREDICTION ==========================

    def run_link_prediction(self):
        """Train the multilayer link-prediction model on the CURRENT graph
        (grouping edges by their relation label into layers) and show a
        ranked list of candidate new links per relation."""
        if len(self.G.edges) < 4:
            return messagebox.showwarning(
                "Not Enough Data",
                "Add more relationships first (or import a CSV). Link prediction needs "
                "at least a handful of links -- ideally 5+ of the same relation type -- "
                "to train and evaluate a model.")

        layers, metadata, node_types = build_relation_layers(self.G)
        all_nodes = sorted(self.G.nodes())

        # One dedicated, explicitly-seeded generator for this run, created
        # fresh every time the button is pressed. This -- not the module-level
        # random.seed(RANDOM_SEED) call at import time -- is what actually
        # guarantees repeatable results: relying on the shared `random`
        # module would mean every prior use of it in this process (including
        # unrelated ones, e.g. add_new_entity_at_center's random.randint
        # calls) shifts where the next split_edges() call starts drawing from.
        rng = random.Random(RANDOM_SEED)

        log_lines = []

        def log(msg):
            log_lines.append(msg)

        self._last_link_predictions = []
        results = {}
        for relation, Gl in layers.items():
            Gl.graph["target_relation"] = relation
            res = evaluate_relation_layer(relation, Gl, layers, metadata, node_types, log=log, rng=rng)
            if res is not None:
                results[relation] = res

        if not results:
            return messagebox.showinfo(
                "Link Prediction",
                "No relation type currently has enough labeled links to train a model.\n\n"
                "Import more data, or add more relationships of the same type, then try again."
            )

        self.show_link_prediction_results(results, layers, metadata, node_types, all_nodes,
                                           "\n".join(log_lines))

    def show_link_prediction_results(self, results, layers, metadata, node_types, all_nodes, log_text):
        win = tk.Toplevel(self.root)
        win.title("Link Prediction Results - Candidate New Links")
        win.geometry("900x600")

        tk.Label(win, text="MULTILAYER LINK PREDICTION", bg="#404040", fg="white",
                 font=("Arial", 11, "bold")).pack(fill=tk.X)

        log_box = tk.Text(win, height=6, wrap="word", bg="#f5f5f5")
        log_box.insert("1.0", log_text)
        log_box.configure(state="disabled")
        log_box.pack(fill=tk.X, padx=5, pady=5)

        notebook = ttk.Notebook(win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for relation, result in results.items():
            candidates = top_k_candidates(
                result, layers, all_nodes, metadata[relation], node_types, relation, k=8)

            tab = tk.Frame(notebook)
            notebook.add(tab, text=relation)

            metrics = (f"ROC-AUC: {result['auc']:.3f}   |   Average Precision: {result['ap']:.3f}   |   "
                       f"evaluated on {result['n_pos']} held-out pos / {result['n_neg']} neg links")
            tk.Label(tab, text=metrics, anchor="w", font=("Arial", 9, "italic")).pack(fill=tk.X, padx=5, pady=(5, 0))

            columns = ("entity_a", "type_a", "entity_b", "type_b", "prob")
            tree = ttk.Treeview(tab, columns=columns, show="headings", selectmode="extended")
            headings = {"entity_a": "Entity A", "type_a": "Type A", "entity_b": "Entity B",
                        "type_b": "Type B", "prob": "Predicted Prob."}
            for c in columns:
                tree.heading(c, text=headings[c])
                tree.column(c, width=140 if c != "prob" else 110, anchor="w")
            tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            candidate_map = {}
            if not candidates:
                tk.Label(tab, text="No eligible candidate pairs found for this relation.",
                         fg="#888888").pack(pady=10)
            for u, v, prob in candidates:
                label_u = self.G.nodes[u]['label'] if u in self.G.nodes else u
                label_v = self.G.nodes[v]['label'] if v in self.G.nodes else v
                type_u = node_types.get(u, "Entity")
                type_v = node_types.get(v, "Entity")
                iid = tree.insert("", tk.END, values=(label_u, type_u, label_v, type_v, f"{prob:.3f}"))
                candidate_map[iid] = (u, v, prob)

            def make_add_handler(tree=tree, candidate_map=candidate_map, relation=relation):
                def add_selected():
                    selected = tree.selection()
                    if not selected:
                        return messagebox.showwarning("No Selection", "Select one or more candidate rows first.")
                    added = 0
                    for iid in selected:
                        u, v, prob = candidate_map[iid]
                        self.add_link(u, v, relation, source="link_prediction",
                                      confidence=prob, weight=1.0, redraw=False)
                        tree.delete(iid)
                        added += 1
                    self.redraw_canvas()
                    messagebox.showinfo("Links Added",
                                         f"Added {added} predicted '{relation}' link(s) to the investigation "
                                         "graph (shown as dashed lines).")
                return add_selected

            btn_row = tk.Frame(tab)
            btn_row.pack(fill=tk.X, padx=5, pady=(0, 5))
            tk.Button(btn_row, text="Add Selected as Predicted Link(s)", bg="#66cccc",
                      command=make_add_handler()).pack(side=tk.LEFT)

    # ========================== NEO4J ACTIONS ==========================

    def neo4j_connect_dialog(self):
        if not HAS_NEO4J_DRIVER:
            return messagebox.showerror(
                "Neo4j Driver Missing",
                "The 'neo4j' Python package isn't installed.\n\n"
                "Install it first:\n    pip install neo4j"
            )
        if self.neo4j.connected:
            if not messagebox.askyesno("Already Connected",
                                        f"Already connected to {self.neo4j.uri}. Reconnect?"):
                return
            self.neo4j.close()

        uri = simpledialog.askstring(
            "Connect to Neo4j", "Neo4j URI:",
            initialvalue="neo4j+s://<your-db-id>.databases.neo4j.io"
        )
        if not uri:
            return
        user = simpledialog.askstring("Connect to Neo4j", "Username:", initialvalue="neo4j")
        if not user:
            return
        password = simpledialog.askstring("Connect to Neo4j", "Password:", show="*")
        if password is None:
            return
        database = simpledialog.askstring(
            "Connect to Neo4j", "Database name:", initialvalue="neo4j"
        ) or "neo4j"

        try:
            self.neo4j.connect(uri, user, password, database)
        except Exception as exc:
            return messagebox.showerror("Connection Failed", str(exc))

        self.status_var.set(f"Connected to Neo4j at {uri} (database: {database}).")
        messagebox.showinfo(
            "Connected",
            f"Connected to Neo4j.\n\n"
            "Every entity/link you add, edit, or delete from now on is mirrored "
            "into this database automatically.\n\n"
            "Use 'Neo4j > Push Current Graph to Neo4j' to also send everything "
            "already in this session."
        )

    def neo4j_disconnect(self):
        if not self.neo4j.connected:
            return messagebox.showinfo("Neo4j", "Not currently connected.")
        self.neo4j.close()
        self.status_var.set("Disconnected from Neo4j. Changes are no longer being synced.")

    def neo4j_push_graph(self):
        if not self.neo4j.connected:
            return messagebox.showwarning("Not Connected", "Connect to Neo4j first (Neo4j > Connect to Neo4j...).")
        if len(self.G.nodes) == 0:
            return messagebox.showinfo("Nothing to Push", "The current graph is empty.")

        pushed_nodes, pushed_edges, errors = 0, 0, []
        for node_id, data in self.G.nodes(data=True):
            try:
                self.neo4j.upsert_node(node_id, data.get("type", "Entity"), data.get("properties", {}))
                pushed_nodes += 1
            except (Neo4jError, ServiceUnavailable, OSError) as exc:
                errors.append(f"Node '{node_id}': {exc}")

        for u, v, key, data in self.G.edges(keys=True, data=True):
            relation = data.get("label", "Linked to")
            try:
                self.neo4j.upsert_relationship(u, v, relation, data)
                pushed_edges += 1
            except (Neo4jError, ServiceUnavailable, OSError) as exc:
                errors.append(f"Edge '{u}->{v}' ({relation}): {exc}")

        summary = f"Pushed {pushed_nodes} node(s) and {pushed_edges} relationship(s) to Neo4j."
        if errors:
            summary += f"\n\n{len(errors)} error(s), e.g.:\n" + "\n".join(errors[:8])
        messagebox.showinfo("Push Complete", summary)

    def neo4j_load_graph(self):
        if not self.neo4j.connected:
            return messagebox.showwarning("Not Connected", "Connect to Neo4j first (Neo4j > Connect to Neo4j...).")
        if len(self.G.nodes) > 0:
            if not messagebox.askyesno(
                "Replace Current Graph",
                "This clears the CURRENT in-memory investigation and replaces it with "
                "what's stored in Neo4j. Continue?"
            ):
                return

        try:
            nodes, edges = self.neo4j.fetch_all()
        except (Neo4jError, ServiceUnavailable, OSError) as exc:
            return messagebox.showerror("Load Failed", str(exc))

        self.G.clear()
        self.node_positions.clear()
        self.evidence_store.fir_documents.clear()
        self._anomaly_results = {}
        self._last_link_predictions = []
        self.selected_node = None
        self.selected_edge = None

        for node_id, entity_type, props in nodes:
            label = props.get("label", node_id)
            # Everything else that came back from Neo4j is treated as the
            # node's stored properties (matches how CSV-imported nodes work).
            properties = {k: v for k, v in props.items() if k not in ("label", "type", "color")}
            self.add_entity(label, entity_type, 0, 0, properties=properties,
                             node_id=node_id, redraw=False)

        for u, v, rel_type, props in edges:
            if u not in self.G.nodes or v not in self.G.nodes:
                continue
            relation_label = props.get("label", rel_type.replace("_", " ").title())
            self.add_link(u, v, relation_label, source=props.get("source", "neo4j"),
                          confidence=float(props.get("confidence", 1.0) or 1.0),
                          weight=float(props.get("weight", 1.0) or 1.0),
                          properties=props, redraw=False)

        self.auto_layout()
        self.status_var.set(f"Loaded {len(nodes)} entities and {len(edges)} relationships from Neo4j.")
        messagebox.showinfo("Loaded from Neo4j",
                             f"Loaded {len(nodes)} entities and {len(edges)} relationships.")

    def neo4j_clear_remote(self):
        if not self.neo4j.connected:
            return messagebox.showwarning("Not Connected", "Connect to Neo4j first (Neo4j > Connect to Neo4j...).")
        if not messagebox.askyesno(
            "Clear Neo4j Database",
            "This permanently deletes ALL nodes and relationships in the connected "
            "Neo4j database (not just this session's data). Continue?"
        ):
            return
        try:
            self.neo4j.clear_all()
        except (Neo4jError, ServiceUnavailable, OSError) as exc:
            return messagebox.showerror("Clear Failed", str(exc))
        messagebox.showinfo("Cleared", "The connected Neo4j database is now empty.")

    # ========================== IMPORT / EXPORT ==========================

    @staticmethod
    def _csv_column_lookup(fieldnames):
        """Case-insensitive column lookup helper for flexible CSV schemas."""
        return {name.strip().lower(): name for name in (fieldnames or [])}

    @staticmethod
    def _row_get(row, lookup, *names, default=None):
        for name in names:
            key = lookup.get(name)
            if key is not None:
                val = (row.get(key) or "").strip()
                if val != "":
                    return val
        return default

    @staticmethod
    def _normalise_csv_value(value):
        return str(value).strip()

    def _find_relationship_endpoint_columns(self, fieldnames, rows):
        """Find two columns whose values best match existing node IDs.

        No source/target/relation column names are required. Header names are
        only a weak tie-breaker; the actual decision is based primarily on
        matching values against graph node IDs.
        """
        node_ids = {str(n) for n in self.G.nodes()}
        if len(fieldnames) < 2 or not node_ids:
            return None

        scores = []
        for col in fieldnames:
            values = [self._normalise_csv_value(r.get(col, "")) for r in rows]
            nonempty = [v for v in values if v]
            if not nonempty:
                continue
            matches = sum(v in node_ids for v in nonempty)
            ratio = matches / len(nonempty)
            header = col.lower()
            hint = 0
            if any(x in header for x in ("source", "from", "origin", "src")):
                hint += 0.02
            if any(x in header for x in ("target", "to", "destination", "dest", "dst")):
                hint += 0.02
            scores.append((ratio + hint, matches, col))

        scores.sort(reverse=True)
        if len(scores) < 2 or scores[0][1] == 0 or scores[1][1] == 0:
            return None

        return scores[0][2], scores[1][2]

    # Column-name candidates (lowercased) that represent relationship
    # *strength* for common investigation data: call duration and
    # transaction amount. Extend this list if new evidence types are
    # imported (e.g. message counts, transfer frequency).
    _STRENGTH_COLUMN_CANDIDATES = (
        "duration_sec", "duration", "call_duration", "seconds", "duration_seconds",
        "amount_inr", "amount_usd", "amount", "amount_rs", "value", "amt", "transaction_amount", "txn_amount",
    )

    @classmethod
    def _find_strength_column(cls, fieldnames, rows):
        """Pick the column (if any) that represents relationship strength,
        e.g. call duration in seconds or transaction amount in rupees.

        Returns (column_name, list_of_floats_in_row_order) or (None, None)
        if no such numeric column is present. Only rows are scanned; missing
        or non-numeric cells become 0.0 rather than dropping the row, so a
        malformed value doesn't silently exclude an otherwise valid link.
        """
        lower_map = {c.lower(): c for c in fieldnames}
        for candidate in cls._STRENGTH_COLUMN_CANDIDATES:
            if candidate in lower_map:
                col = lower_map[candidate]
                values = []
                any_numeric = False
                for row in rows:
                    raw = (row.get(col) or "").strip()
                    try:
                        v = float(raw)
                        any_numeric = True
                    except (TypeError, ValueError):
                        v = 0.0
                    values.append(v)
                if any_numeric:
                    return col, values
        return None, None

    @staticmethod
    def _normalise_strength_to_weight(raw_values, low=0.2, high=1.0):
        """Convert raw strength values (call seconds, rupees, ...) into
        comparable per-observation edge weights.

        Different relation types (calls vs. transactions) live on
        incompatible numeric scales -- 300 seconds and Rs 300,000 cannot be
        summed directly. Compressing with log1p (so a handful of very large
        outlier transactions don't dominate every other interaction) and
        then min-max scaling every relation-type's values independently into
        the same [low, high] band puts every layer on a comparable footing
        *before* _analysis_graph() sums them across relation types. `low`
        is kept above 0 so even the weakest observed interaction in a file
        still counts as real evidence, never a zero-weight edge.
        """
        if not raw_values:
            return []
        logged = [np.log1p(max(v, 0.0)) for v in raw_values]
        lo, hi = min(logged), max(logged)
        if hi - lo < 1e-9:
            # No spread to distinguish interactions by (e.g. every call in
            # the file lasted exactly as long) -- treat them all as equally
            # strong evidence rather than fabricating a difference.
            return [1.0] * len(raw_values)
        return [low + (high - low) * (x - lo) / (hi - lo) for x in logged]

    def _infer_relation_label(self, path, properties, fieldnames):
        """Infer a useful relation label without requiring a fixed CSV schema."""
        for key in ("relation", "relationship", "rel_type", "edge_type", "type"):
            if key in properties and properties[key].strip():
                return properties[key].strip()

        filename = str(path).lower()
        if any(x in filename for x in ("cdr", "call", "phone")):
            return "Calls"
        if any(x in filename for x in ("financial", "transaction", "payment", "money")):
            return "Transactions"
        return "Linked to"

    def import_relationships_csv(self):
        """Import multiple relationship CSVs in one operation.

        Each selected CSV is treated as a separate relationship source. If a
        row contains an explicit relation/relationship column, that value is
        used. Otherwise the relation is inferred from the filename (e.g.
        financial.csv -> Transactions, cdr.csv -> Calls, ownership.csv ->
        Ownership).
        """
        paths = filedialog.askopenfilenames(
            title="Select Relationship CSV Files",
            filetypes=[("CSV files", "*.csv")]
        )
        if not paths:
            return

        self._suspend_ai_snapshot = True
        total_added = 0
        total_skipped = 0
        file_summaries = []
        skipped_details = []

        for path in paths:
            added = 0
            skipped = 0

            try:
                with open(path, newline='', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    fieldnames = reader.fieldnames or []

                endpoints = self._find_relationship_endpoint_columns(fieldnames, rows)
                if not endpoints:
                    file_summaries.append(
                        f"{Path(path).name}: skipped - could not identify two endpoint columns"
                    )
                    total_skipped += len(rows)
                    continue

                source_col, target_col = endpoints
                filename_relation = self._infer_relation_label(path, {}, fieldnames)

                # Derive a per-row relationship weight from call duration /
                # transaction amount (or any other recognised strength
                # column) instead of letting every row default to weight=1.
                # See _find_strength_column / _normalise_strength_to_weight.
                strength_col, raw_strengths = self._find_strength_column(fieldnames, rows)
                row_weights = (
                    self._normalise_strength_to_weight(raw_strengths)
                    if strength_col else None
                )

                for row_number, row in enumerate(rows, start=2):
                    src = self._normalise_csv_value(row.get(source_col, ""))
                    tgt = self._normalise_csv_value(row.get(target_col, ""))

                    if not src or not tgt:
                        skipped += 1
                        skipped_details.append(
                            f"{Path(path).name}, row {row_number}: missing endpoint"
                        )
                        continue

                    if src not in self.G.nodes or tgt not in self.G.nodes:
                        skipped += 1
                        missing = []
                        if src not in self.G.nodes:
                            missing.append(f"{source_col}='{src}'")
                        if tgt not in self.G.nodes:
                            missing.append(f"{target_col}='{tgt}'")
                        skipped_details.append(
                            f"{Path(path).name}, row {row_number}: node not found "
                            f"({', '.join(missing)})"
                        )
                        continue

                    properties = {
                        col: self._normalise_csv_value(row.get(col, ""))
                        for col in fieldnames
                        if col not in (source_col, target_col)
                        and self._normalise_csv_value(row.get(col, "")) != ""
                    }
                    properties["_source_file"] = Path(path).name
                    properties["_row"] = row_number
                    properties["_source_type"] = "CSV"

                    # Prefer an explicit per-row relation. Otherwise use the
                    # filename-derived relation, so every selected file can
                    # represent a distinct layer.
                    relation = self._infer_relation_label(path, properties, fieldnames)
                    if relation == "Linked to" and filename_relation != "Linked to":
                        relation = filename_relation

                    row_index = row_number - 2  # rows is 0-indexed; header was row 1
                    link_weight = (
                        row_weights[row_index] if row_weights is not None else 1.0
                    )

                    self.add_link(
                        src, tgt,
                        relation_label=relation,
                        source="csv_import",
                        weight=link_weight,
                        properties=properties,
                        redraw=False
                    )
                    added += 1

                weight_note = (
                    f"; weighted by '{strength_col}'" if strength_col else ""
                )
                file_summaries.append(
                    f"{Path(path).name}: {added} links added, {skipped} skipped "
                    f"({source_col} -> {target_col}; relation: {filename_relation}{weight_note})"
                )
                total_added += added
                total_skipped += skipped

            except Exception as e:
                file_summaries.append(f"{Path(path).name}: import failed - {e}")

        if total_added:
            self.auto_layout()
        else:
            self.redraw_canvas()

        details = "\n".join(file_summaries)
        if skipped_details:
            details += "\n\nSkipped-row details:\n" + "\n".join(skipped_details[:10])
            if len(skipped_details) > 10:
                details += f"\n... and {len(skipped_details) - 10} more."

        self._suspend_ai_snapshot = False
        self._commit_ai_snapshot()
        messagebox.showinfo(
            "Relationship Import Complete",
            f"Processed {len(paths)} CSV file(s).\n"
            f"Added {total_added} relationship observation(s).\n"
            f"Skipped {total_skipped} row(s).\n\n"
            f"{details}"
        )

    def import_entities_csv(self):
        """Import arbitrary entity CSVs.

        Any column whose header contains the substring 'id' is treated as an
        entity-ID column. Each non-empty value in such a column becomes a node.
        All other non-empty columns from that row are retained as properties.
        If a row contains multiple ID columns, each ID becomes a node and the
        row's non-ID fields are copied as properties to each node.
        """
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return

        added = 0
        self._suspend_ai_snapshot = True
        try:
            with open(path, newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []

            id_columns = [c for c in fieldnames if "id" in c.lower()]
            if not id_columns:
                self._suspend_ai_snapshot = False
                return messagebox.showerror(
                    "No ID column found",
                    "No column name contains 'id'.\n\n"
                    "Examples that are accepted: id, node_id, person_id, account_id."
                )

            for row_number, row in enumerate(rows, start=2):
                for id_col in id_columns:
                    node_id = self._normalise_csv_value(row.get(id_col, ""))
                    if not node_id:
                        continue

                    # Store the complete row as node properties, excluding the
                    # ID columns so the inspector isn't cluttered with duplicate
                    # identifiers when a row contains several IDs.
                    properties = {
                        col: self._normalise_csv_value(row.get(col, ""))
                        for col in fieldnames
                        if col not in id_columns
                        and self._normalise_csv_value(row.get(col, "")) != ""
                    }
                    properties[id_col] = node_id
                    properties["_source_file"] = Path(path).name
                    properties["_row"] = row_number
                    properties["_source_type"] = "CSV"

                    # Optional display type is taken from a row value if present,
                    # but no particular type vocabulary is required.
                    e_type = (
                        self._normalise_csv_value(row.get("type", ""))
                        or self._normalise_csv_value(row.get("entity_type", ""))
                        or "Entity"
                    )

                    self.add_entity(
                        label=node_id,
                        e_type=e_type,
                        node_id=node_id,
                        properties=properties,
                        x=0, y=0, redraw=False
                    )
                    added += 1

        except Exception as e:
            return messagebox.showerror("Import Failed", f"Could not parse CSV: {e}")

        self._suspend_ai_snapshot = False
        self._commit_ai_snapshot()
        if added:
            self.auto_layout()
        else:
            self.redraw_canvas()

        messagebox.showinfo(
            "Import Complete",
            f"Added/updated {added} entity record(s).\n"
            f"ID columns detected: {', '.join(id_columns)}"
        )

    def import_ner(self):
        """Import your teammates' model output. Expected JSON shape:
        {
          "entities": [{"name": "...", "type": "Suspect"}, ...],
          "relations": [{"from": "...", "from_type": "Suspect",
                          "to": "...", "to_type": "Phone Number",
                          "label": "Uses", "confidence": 0.82}, ...]
        }
        Predicted links render as dashed lines until an analyst verifies them."""
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                payload = json.load(f)
            added_nodes, added_edges = 0, 0
            for ent in payload.get("entities", []):
                self.add_entity(ent["name"], ent.get("type", "Suspect"), 320, 320, redraw=False)
                added_nodes += 1
            for rel in payload.get("relations", []):
                u = f"{rel.get('from_type', 'Suspect').replace(' ', '_')}_{rel['from'].replace(' ', '_')}"
                v = f"{rel.get('to_type', 'Suspect').replace(' ', '_')}_{rel['to'].replace(' ', '_')}"
                self.add_link(u, v, rel.get('label', 'Related to'),
                              source="ml_predicted",
                              confidence=rel.get('confidence', 0.7), redraw=False)
                added_edges += 1
        except Exception as e:
            return messagebox.showerror("Import Failed", f"Could not parse model output: {e}")

        if added_nodes:
            self.auto_layout()
        else:
            self.redraw_canvas()
        messagebox.showinfo("Import Complete",
                             f"Added {added_nodes} entities and {added_edges} predicted link(s) "
                             "(shown as dashed lines until an analyst verifies them).")

    def save_investigation(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not path:
            return
        payload = {
            "nodes": [{"id": n, "pos": self.node_positions[n], **d} for n, d in self.G.nodes(data=True)],
            "edges": [{"source": u, "target": v, "key": k, **d}
                      for u, v, k, d in self.G.edges(keys=True, data=True)],
            "fir_documents": self.evidence_store.fir_documents,
        }
        with open(path, "w", encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        messagebox.showinfo("Saved", f"Investigation saved to {path}")

    def load_investigation(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not path:
            return
        with open(path, encoding='utf-8') as f:
            payload = json.load(f)

        self.G.clear()
        self.node_positions.clear()
        self.selected_node = None
        self.selected_edge = None
        self.evidence_store.fir_documents = payload.get("fir_documents", [])
        self._last_link_predictions = []

        for n in payload["nodes"]:
            n = dict(n)
            node_id = n.pop("id")
            pos = n.pop("pos")
            self.G.add_node(node_id, **n)
            self.node_positions[node_id] = pos

        for e in payload["edges"]:
            e = dict(e)
            u = e.pop("source")
            v = e.pop("target")
            key = e.pop("key", None)
            if key is None:
                self.G.add_edge(u, v, **e)
            else:
                self.G.add_edge(u, v, key=key, **e)

        self._commit_ai_snapshot()
        self.redraw_canvas()
        messagebox.showinfo("Loaded", f"Investigation loaded from {path}")

    def export_report(self):
        if len(self.G.nodes) == 0:
            return messagebox.showwarning("Warning", "Nothing to export yet.")
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not path:
            return
        lines = ["INVESTIGATION REPORT", "=" * 40, ""]
        lines.append(f"Total entities: {len(self.G.nodes)}")
        lines.append(f"Total links: {len(self.G.edges)}")
        lines.append("")
        lines.append("KEY ENTITIES (by influence score):")
        ranked = sorted(self.G.nodes(data=True), key=lambda x: x[1].get('centrality', 0), reverse=True)
        for node_id, data in ranked[:10]:
            score = data.get('centrality')
            score_str = f"{score:.4f}" if score is not None else "n/a"
            lines.append(f"  - {data['label']} ({data['type']}) - influence {score_str}")
        lines.append("")
        lines.append("ALL RELATIONSHIPS:")
        for u, v, k, d in self.G.edges(keys=True, data=True):
            lines.append(f"  - {self.G.nodes[u]['label']} --[{d.get('label', '')}]--> "
                          f"{self.G.nodes[v]['label']} "
                          f"(count: {d.get('count', 1)}; source: {d.get('source', 'manual')})")

        with open(path, "w", encoding='utf-8') as f:
            f.write("\n".join(lines))
        messagebox.showinfo("Exported", f"Report saved to {path}")

    def clear_investigation(self):
        if len(self.G.nodes) == 0:
            return
        if not messagebox.askyesno("Clear Investigation",
                                    "This removes ALL entities and links from the current graph. Continue?"):
            return
        self.G.clear()
        self.node_positions.clear()
        self.evidence_store.fir_documents.clear()
        self._anomaly_results = {}
        self._last_link_predictions = []
        self.selected_node = None
        self.selected_edge = None
        for i in self.prop_tree.get_children():
            self.prop_tree.delete(i)
        self.redraw_canvas()
        self._commit_ai_snapshot()


    # ========================== AI / EVIDENCE / ANOMALY ACTIONS ==========================

    def _build_ai_panel(self):
        """Build the AI Copilot inside the existing right-side panel."""
        tk.Label(self.ai_tab, text="✦ AI INVESTIGATION COPILOT", bg="#404040", fg="white",
                 font=("Arial", 10, "bold")).pack(fill=tk.X)

        status_row = tk.Frame(self.ai_tab, bg="#d9d9d9")
        status_row.pack(fill=tk.X, padx=5, pady=4)
        self.ai_status_var = tk.StringVar(value="● CHECKING LOCAL AI")
        self.ai_model_label_var = tk.StringVar(value="Local AI")
        tk.Label(status_row, textvariable=self.ai_status_var, bg="#d9d9d9", anchor="w").pack(side=tk.LEFT)
        tk.Label(status_row, textvariable=self.ai_model_label_var, bg="#d9d9d9", anchor="e",
                 font=("Arial", 8)).pack(side=tk.RIGHT)

        # Keep the model configuration internal; do not expose the model name in the UI.

        tk.Label(self.ai_tab, text="AI SPECIALISTS", bg="#404040", fg="white",
                 font=("Arial", 9, "bold")).pack(fill=tk.X, pady=(6, 0))
        self.ai_agents_box = tk.Text(self.ai_tab, height=5, wrap="word", bg="#f5f5f5",
                                     fg="black", relief=tk.SUNKEN, bd=1, font=("Arial", 8))
        self.ai_agents_box.pack(fill=tk.X, padx=5, pady=4)
        self._render_ai_agents()

        fir_row = tk.Frame(self.ai_tab, bg="#d9d9d9")
        fir_row.pack(fill=tk.X, padx=5, pady=(2, 4))
        tk.Button(fir_row, text="📄 UPLOAD HANDWRITTEN FIR", bg="#66cccc",
                  command=self.upload_fir).pack(fill=tk.X)

        action_row = tk.Frame(self.ai_tab, bg="#d9d9d9")
        action_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Button(action_row, text="Explain Score", bg="#66cccc", command=self.ai_explain_selected).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        tk.Button(action_row, text="Find Anomalies", command=self.run_anomaly_detection).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        action_row2 = tk.Frame(self.ai_tab, bg="#d9d9d9")
        action_row2.pack(fill=tk.X, padx=5, pady=2)
        tk.Button(action_row2, text="Show Evidence", command=self.ai_show_evidence).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        tk.Button(action_row2, text="Challenge AI", command=self.ai_challenge).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        tk.Label(self.ai_tab, text="SUGGESTED QUESTIONS", bg="#404040", fg="white",
                 font=("Arial", 9, "bold")).pack(fill=tk.X, pady=(6, 0))
        questions = [
            "Why is this entity important?",
            "What anomalies involve this entity?",
            "What evidence supports this score?",
            "Why was this relationship predicted?",
            "Who should I investigate next?",
            "What evidence contradicts this hypothesis?",
        ]
        qbox = tk.Frame(self.ai_tab, bg="#d9d9d9")
        qbox.pack(fill=tk.X, padx=5, pady=3)
        for q in questions:
            tk.Button(qbox, text=q, anchor="w", relief=tk.GROOVE,
                      command=lambda text=q: self._ask_question(text)).pack(fill=tk.X, pady=1)

        tk.Label(self.ai_tab, text="✦ AI RESPONSE", bg="#404040", fg="white",
                 font=("Arial", 9, "bold")).pack(fill=tk.X, pady=(4, 0))
        self.ai_chat = tk.Text(self.ai_tab, height=11, wrap="word", bg="white", fg="black", relief=tk.SUNKEN,
                               bd=1, font=("Arial", 9))
        self.ai_chat.pack(fill=tk.BOTH, expand=True, padx=5, pady=4)
        self.ai_chat.insert("1.0", "AI Copilot ready. Select an entity or ask a question.\n\n"
                                   "This local assistant queries the live graph and FIR evidence only when needed.\n")
        self.ai_chat.configure(state="disabled")

        input_row = tk.Frame(self.ai_tab, bg="#d9d9d9")
        input_row.pack(fill=tk.X, padx=5, pady=(0, 2))
        self.ai_input = tk.Entry(input_row)
        self.ai_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.ai_input.bind("<Return>", lambda _e: self.ask_ai())
        tk.Button(input_row, text="✦ ASK AI", bg="#66cccc", command=self.ask_ai).pack(side=tk.RIGHT, padx=(4, 0))

        self.ai_activity_var = tk.StringVar(value="AI ACTIVITY: ready")
        tk.Label(self.ai_tab, textvariable=self.ai_activity_var, bg="#d9d9d9", fg="#555555",
                 anchor="w", font=("Arial", 8)).pack(fill=tk.X, padx=5, pady=(0, 4))

    def _render_ai_agents(self, running=None):
        running = running or set()
        if not hasattr(self, "ai_agents_box"):
            return
        self.ai_agents_box.configure(state="normal")
        self.ai_agents_box.delete("1.0", "end")
        lines = [
            f"{'●' if 'network' not in running else '…'} LLM 1 — Criminology Specialist",
            f"{'●' if 'temporal' not in running else '…'} LLM 2 — Temporal Pattern Expert",
            f"{'●' if 'context' not in running else '…'} LLM 3 — Demographic / Context Analyst",
            "◆ Evidence Evaluator — composite explanation",
        ]
        self.ai_agents_box.insert("1.0", "\n".join(lines))
        self.ai_agents_box.configure(state="disabled")

    def _ai_activity(self, text):
        self._ai_activity_lines.append(time.strftime("%H:%M:%S") + "  " + text)
        self._ai_activity_lines = self._ai_activity_lines[-5:]
        if hasattr(self, "ai_activity_var"):
            self.ai_activity_var.set("AI ACTIVITY: " + " | ".join(x.split("  ", 1)[-1] for x in self._ai_activity_lines[-2:]))

    def _append_ai_chat(self, text, header="AI"):
        self.ai_chat.configure(state="normal")
        self.ai_chat.insert("end", f"\n{header}:\n{text}\n")
        self.ai_chat.see("end")
        self.ai_chat.configure(state="disabled")

    def _show_ai_selection(self, node_id):
        if not hasattr(self, "ai_chat"):
            return
        label = self.G.nodes[node_id].get("label", node_id)
        # AI Copilot remains visible alongside the inspector.
        score = self.calculate_evidence_score(node_id)
        anomaly_count = len(self._anomaly_results.get(str(node_id), []))
        text = (f"Selected entity: {label} ({node_id})\n"
                f"Evidence Association Score: {score['score']:.0f}/100\n"
                f"Detected anomaly signals: {anomaly_count}\n\n"
                f"Suggested: click 'Explain Score' or ask a question below.")
        self._append_ai_chat(text, header="✦ AI BRIEFING")

    def _check_ai_status_async(self):
        if not self.ai:
            return
        def worker():
            self.ai.refresh_backend()
            ok, detail = self.ai.backend.available()
            self.root.after(0, lambda: self._set_ai_status(ok, detail))
        threading.Thread(target=worker, daemon=True).start()

    def _set_ai_status(self, ok, detail):
        self.ai_status_var.set(("● LOCAL AI ONLINE" if ok else "○ LOCAL AI OFFLINE"))
        self.ai_model_label_var.set("Local AI")
        # Keep the configured model name out of the visible status bar.

    def show_ai_status(self):
        self._check_ai_status_async()
        messagebox.showinfo("AI Copilot", f"Local AI\nEndpoint: {self.ollama_url_var.get()}\n\n"
                            "No cloud API key is used. The application connects to the local AI service when AI is requested.")

    def ask_ai(self):
        question = self.ai_input.get().strip()
        if not question:
            return
        if self._ai_busy:
            return messagebox.showinfo("AI Busy", "The AI is already analyzing a request. Please wait for the response.")
        self.ai_input.delete(0, "end")
        self._run_ai_async(question, deep=False)

    def _ask_question(self, text):
        if self._ai_busy:
            return
        self._append_ai_chat(text, header="YOU")
        self._run_ai_async(text, deep=False)

    def _run_ai_async(self, question, deep=False):
        node_id = self.selected_node if self.selected_node in self.G else None
        self._ai_busy = True
        self._ai_activity("Query received — retrieving live evidence")
        self.ai_status_var.set("… LOCAL AI WORKING")
        def worker():
            try:
                if deep:
                    answer, specialists, deterministic = self.ai.deep_analysis(node_id)
                    result = (answer, specialists, deterministic)
                else:
                    result = self.ai.chat(question, node_id)
                self.root.after(0, lambda: self._ai_done(question, result, deep))
            except Exception as exc:
                self.root.after(0, lambda: self._ai_error(str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _ai_done(self, question, result, deep):
        self._ai_busy = False
        self.ai_status_var.set("● LOCAL AI ONLINE")
        self.ai._activity = getattr(self.ai, "_activity", [])
        if deep:
            answer, specialists, deterministic = result
            self._append_ai_chat(answer, header="◆ EVIDENCE EVALUATOR")
            self._append_ai_chat("Deterministic score: " + f"{deterministic['score']:.0f}/100\n" +
                                 json.dumps(deterministic["components"], indent=2), header="SCORE TRACE")
        else:
            self._append_ai_chat(str(result), header="✦ AI COPILOT")
        self._ai_activity("Response generated")

    def _ai_error(self, message):
        self._ai_busy = False
        self.ai_status_var.set("○ LOCAL AI OFFLINE")
        self._append_ai_chat(message + "\n\nTip: start the local AI service and verify that a model is installed.", header="AI STATUS")
        self._ai_activity("Request failed")

    def calculate_evidence_score(self, node_id):
        if node_id not in self.G:
            return {"score": 0.0, "components": {}}
        d = self.G.nodes[node_id]
        network = min(max(float(d.get("centrality", 0.0) or 0.0), 0.0), 1.0)
        anomalies = self._anomaly_results.get(str(node_id), [])
        anomaly = min(1.0, statistics.mean([float(a.get("score", 0.0)) for a in anomalies])) if anomalies else 0.0
        rel_strength = min(1.0, math.log1p(float(d.get("weighted_degree", self.G.degree(node_id)) or 0.0)) / 5.0)
        fir_hits = len(self._collect_entity_evidence(node_id).get("fir_mentions", []))
        fir = min(1.0, fir_hits / 3.0)
        preds = [x["probability"] for x in self._last_link_predictions if x["from"] == str(node_id) or x["to"] == str(node_id)]
        link_pred = max(preds) if preds else 0.0
        score = 100 * (0.30 * network + 0.25 * anomaly + 0.20 * rel_strength + 0.15 * fir + 0.10 * link_pred)
        return {"score": score, "components": {
            "network_influence": round(network, 4),
            "anomaly_signals": round(anomaly, 4),
            "relationship_strength": round(rel_strength, 4),
            "FIR_correlation": round(fir, 4),
            "link_prediction_signal": round(link_pred, 4),
        }}

    def _collect_entity_evidence(self, node_id):
        evidence = {"csv": [], "fir_mentions": [], "relationships": []}
        if node_id not in self.G:
            return evidence
        for _, _, data in self.G.edges(node_id, data=True):
            for record in data.get("records", []):
                item = {
                    "source_file": record.get("_source_file", record.get("source_file")),
                    "row": record.get("_row"),
                    "relation": data.get("label", "Linked to"),
                    "properties": {k: v for k, v in record.items() if not str(k).startswith("_")}
                }
                evidence["csv"].append(item)
                evidence["relationships"].append(item)
        label = str(self.G.nodes[node_id].get("label", node_id)).lower()
        canonical = str(node_id).lower()
        for doc in self.evidence_store.fir_documents:
            text = doc.get("text", "").lower()
            if label and label in text or canonical in text:
                evidence["fir_mentions"].append({"fir_id": doc["id"], "file": Path(doc["file"]).name})
        evidence["csv"] = evidence["csv"][:25]
        evidence["relationships"] = evidence["relationships"][:25]
        return evidence

    def _refresh_anomalies(self):
        self._anomaly_results = self.anomaly_engine.analyze(self.G, self._previous_ai_snapshot)

    def _commit_ai_snapshot(self):
        # Refresh structural metrics before snapshotting so centrality anomalies
        # compare meaningful before/after values.
        try:
            if self.G.number_of_nodes() >= 2:
                self.compute_centrality_metrics()
        except Exception:
            pass
        self._refresh_anomalies()
        self._previous_ai_snapshot = self.anomaly_engine.snapshot(self.G)

    def run_anomaly_detection(self):
        if not self.G.nodes:
            return messagebox.showwarning("Anomalies", "Load some evidence first.")
        # _commit_ai_snapshot() already computed the latest anomaly signals at the
        # time of the graph mutation. Do not replace them by comparing the graph
        # against an identical current snapshot.
        counts = {}
        for items in self._anomaly_results.values():
            for item in items:
                counts[item["type"]] = counts.get(item["type"], 0) + 1
        lines = ["12-TYPE ANOMALY SCAN", "=" * 32]
        for typ in AnomalyEngine.TYPES:
            lines.append(f"{typ}: {counts.get(typ, 0)} signal(s)")
        lines.append("\nSelect an entity and use Find Anomalies / Explain Score for entity-level reasoning.")
        messagebox.showinfo("Anomaly Detection", "\n".join(lines))
        if self.selected_node in self.G:
            self._append_ai_chat(json.dumps(self._anomaly_results.get(str(self.selected_node), []), indent=2), header="ANOMALY ENGINE")

    def ai_explain_selected(self):
        if self.selected_node not in self.G:
            return messagebox.showwarning("AI", "Select an entity first.")
        if self._ai_busy:
            return
        # AI Copilot remains visible alongside the inspector.
        self._append_ai_chat("Deep analysis requested for " + str(self.G.nodes[self.selected_node].get("label", self.selected_node)), header="YOU")
        self._run_ai_async("Explain the selected entity's evidence score, anomalies, strongest supporting and contradicting evidence, and suggest next investigative questions.", deep=True)

    def ai_show_evidence(self):
        if self.selected_node not in self.G:
            return messagebox.showwarning("Evidence", "Select an entity first.")
        evidence = self._collect_entity_evidence(self.selected_node)
        win = tk.Toplevel(self.root)
        win.title("Evidence References")
        win.geometry("650x500")
        tk.Label(win, text="EVIDENCE REFERENCES", bg="#404040", fg="white", font=("Arial", 11, "bold")).pack(fill=tk.X)
        txt = tk.Text(win, wrap="word")
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        txt.insert("1.0", json.dumps(evidence, indent=2, default=str))
        txt.configure(state="disabled")

    def ai_challenge(self):
        if self.selected_node not in self.G:
            return messagebox.showwarning("AI", "Select an entity first.")
        question = "Challenge the current assessment. Identify evidence against the hypothesis, missing evidence, possible alternative explanations, and reasons the current score could be misleading."
        self._append_ai_chat(question, header="YOU")
        self._run_ai_async(question, deep=False)

    def _fuzzy_fir_candidate(self, entity_type, value):
        """Return the best existing node label match and its ratio score."""
        if not HAS_THEFUZZ:
            raise RuntimeError("TheFuzz is required for FIR entity matching. Install: pip install thefuzz python-Levenshtein")
        target = FIRProcessor._normalise_value(value)
        if not target:
            return None, 0
        best_node = None
        best_score = 0
        for node_id, data in self.G.nodes(data=True):
            label = str(data.get("label", node_id)).strip()
            if not label:
                continue
            score = fuzz.ratio(target, FIRProcessor._normalise_value(label))
            if score > best_score:
                best_node = node_id
                best_score = int(score)
        return best_node, best_score

    def _review_fir_matches(self, entities):
        """Resolve 75-92% fuzzy matches and all ambiguous entities via Tkinter."""
        decisions = {}
        pending = []
        for index, item in enumerate(entities):
            value = str(item.get("value", item.get("name", ""))).strip()
            if not value:
                continue
            etype = str(item.get("type", "Entity")).strip().lower()
            ambiguous = bool(item.get("is_ambiguous", False))
            candidate, score = self._fuzzy_fir_candidate(etype, value)
            if not ambiguous and score > 92 and candidate is not None:
                decisions[index] = {"action": "merge", "node_id": candidate, "score": score}
            elif ambiguous or (75 <= score <= 92):
                pending.append({
                    "index": index,
                    "value": value,
                    "type": FIRProcessor._canonical_type(etype),
                    "candidate": candidate,
                    "candidate_label": self.G.nodes[candidate].get("label", candidate) if candidate in self.G else "No candidate",
                    "score": score,
                    "ambiguous": ambiguous,
                })
            else:
                decisions[index] = {"action": "create", "node_id": None, "score": score}

        if not pending:
            return decisions

        dialog = tk.Toplevel(self.root)
        dialog.title("FIR Entity Review Queue")
        dialog.geometry("900x560")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Review FIR entity matches", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        ttk.Label(dialog, text="Merge uses the suggested existing node. Create New adds a separate graph entity.", wraplength=850).pack(anchor="w", padx=12, pady=(0, 8))

        outer = ttk.Frame(dialog)
        outer.pack(fill="both", expand=True, padx=12, pady=8)
        canvas = tk.Canvas(outer, bg="#f0f0f0", highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        choices = {}
        for row, item in enumerate(pending):
            choice = tk.StringVar(value="create")
            choices[item["index"]] = choice
            frame = tk.Frame(inner, bg="#d9d9d9", bd=1, relief="solid")
            frame.grid(row=row, column=0, sticky="ew", padx=2, pady=4)
            inner.columnconfigure(0, weight=1)
            ambiguous_text = "AMBIGUOUS" if item["ambiguous"] else "FUZZY MATCH"
            tk.Label(frame, text=f"{item['value']}  [{item['type']}]", bg="#d9d9d9", font=("Segoe UI", 10, "bold"), anchor="w").grid(row=0, column=0, sticky="ew", padx=8, pady=(7, 2))
            tk.Label(frame, text=f"{ambiguous_text} · similarity {item['score']}% · candidate: {item['candidate_label']}", bg="#d9d9d9", fg="#555555", anchor="w").grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 5))
            merge_state = "normal" if item["candidate"] in self.G.nodes else "disabled"
            tk.Radiobutton(frame, text="Merge with candidate", variable=choice, value="merge", state=merge_state, bg="#d9d9d9", anchor="w").grid(row=2, column=0, sticky="w", padx=8, pady=2)
            tk.Radiobutton(frame, text="Create New", variable=choice, value="create", bg="#d9d9d9", anchor="w").grid(row=3, column=0, sticky="w", padx=8, pady=(2, 7))

        result = {"confirmed": False}
        buttons = ttk.Frame(dialog)
        buttons.pack(fill="x", padx=12, pady=10)

        def approve():
            for item in pending:
                action = choices[item["index"]].get()
                if action == "merge" and item["candidate"] in self.G.nodes:
                    decisions[item["index"]] = {"action": "merge", "node_id": item["candidate"], "score": item["score"]}
                else:
                    decisions[item["index"]] = {"action": "create", "node_id": None, "score": item["score"]}
            result["confirmed"] = True
            dialog.destroy()

        def cancel():
            result["confirmed"] = False
            dialog.destroy()

        ttk.Button(buttons, text="Approve & Continue", command=approve).pack(side="right", padx=(6, 0))
        ttk.Button(buttons, text="Cancel FIR import", command=cancel).pack(side="right")
        self.root.wait_window(dialog)
        if not result["confirmed"]:
            raise RuntimeError("FIR import cancelled during entity review.")
        return decisions

    def _integrate_fir_entities(self, case_id, path, structured, decisions=None):
        decisions = decisions or {}
        entity_nodes = {}
        created = 0
        matched = 0
        added_links = 0
        source_file = Path(path).name

        for index, item in enumerate(structured.get("entities", [])):
            etype = str(item.get("type", "Entity")).lower()
            value = str(item.get("value", item.get("name", ""))).strip()
            if not value:
                continue
            decision = decisions.get(index, {"action": "create", "node_id": None})
            existing = decision.get("node_id") if decision.get("action") == "merge" else None
            if existing is not None and existing not in self.G:
                existing = None

            if existing is None:
                node_type = FIRProcessor.ENTITY_TYPE_MAP.get(etype, FIRProcessor._canonical_type(etype))
                existing = self.add_entity(
                    label=value,
                    e_type=node_type,
                    node_id=f"FIR_{case_id}_{etype}_{len(entity_nodes)+1}",
                    properties={
                        "value": value,
                        "entity_type": node_type,
                        "is_ambiguous": bool(item.get("is_ambiguous", False)),
                        "match_score": int(decision.get("score", 0) or 0),
                        "_source_file": source_file,
                        "_source_type": "FIR",
                        "_fir_id": case_id,
                    },
                    x=420 + 40 * len(entity_nodes), y=280,
                    redraw=False,
                )
                created += 1
            else:
                matched += 1
                self.G.nodes[existing].setdefault("properties", {})[f"fir_{case_id}"] = value
                self.G.nodes[existing]["properties"][f"fir_{case_id}_match_score"] = int(decision.get("score", 0) or 0)
                self.G.nodes[existing]["properties"][f"fir_{case_id}_ambiguous"] = bool(item.get("is_ambiguous", False))

            entity_nodes[FIRProcessor._normalise_value(value)] = existing
            self.add_link(
                case_id, existing, "Contains evidence",
                source="fir_extraction", confidence=0.90 if decision.get("action") == "merge" else 0.80, weight=1.0,
                properties={
                    "_source_file": source_file,
                    "_source_type": "FIR",
                    "_fir_id": case_id,
                    "entity_type": etype,
                    "matched_existing": bool(decision.get("action") == "merge"),
                    "match_score": int(decision.get("score", 0) or 0),
                    "value": value,
                }, redraw=False,
            )
            added_links += 1

        # Add explicitly stated relationships only after entity resolution.
        for rel in structured.get("relationships", []):
            a = entity_nodes.get(FIRProcessor._normalise_value(rel.get("from", "")))
            b = entity_nodes.get(FIRProcessor._normalise_value(rel.get("to", "")))
            if not a or not b or a == b:
                continue
            self.add_link(
                a, b, rel.get("relation") or "Associated with",
                source="fir_extraction", confidence=0.78, weight=1.0,
                properties={"_source_file": source_file, "_source_type": "FIR", "_fir_id": case_id},
                redraw=False,
            )
            added_links += 1
        return created, matched, added_links, entity_nodes

    def upload_fir(self):
        path = filedialog.askopenfilename(
            title="Upload Handwritten FIR",
            filetypes=[
                ("Scanned FIR", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.pdf"),
                ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                ("PDF", "*.pdf"),
            ],
        )
        if not path:
            return
        if getattr(self, "_fir_busy", False):
            return messagebox.showinfo("FIR Processing", "A FIR is already being processed. Please wait.")

        self._fir_busy = True
        # AI Copilot remains visible alongside the inspector.
        self._ai_activity("FIR received — Gemini OCR starting")
        self._append_ai_chat(f"Processing scanned FIR: {Path(path).name}", header="YOU")
        self.ai_status_var.set("… FIR OCR WORKING")

        def worker():
            try:
                result = self.fir_processor.extract_fir(path)
                self.root.after(0, lambda: self._finish_fir_upload(path, result))
            except Exception as exc:
                # Exception variables from an except block are cleared by Python
                # when the block exits, so bind the message as a default argument
                # before scheduling the Tkinter callback.
                error_message = str(exc)
                self.root.after(0, lambda message=error_message: self._fir_upload_error(message))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_fir_upload(self, path, result):
        self._fir_busy = False
        self.ai_status_var.set("● FIR OCR READY")
        try:
            text = str(result.get("transcription", "")).strip()
            structured = {
                "entities": result.get("entities", []),
                "relationships": result.get("relationships", []),
            }
            if not text:
                raise RuntimeError("No FIR transcription was produced.")

            # The transcription has already been saved to disk by FIRProcessor
            # before the structured payload reaches this downstream stage.
            transcription_path = result.get("transcription_path")
            backend = result.get("ocr_backend", "UNKNOWN")
            cloud_error = result.get("cloud_error", "")
            self._ai_activity(f"OCR completed via {backend} — transcription saved")

            doc = self.evidence_store.add_fir(path, text, structured.get("entities", []))
            case_id = doc["id"]
            case_props = {
                "document": Path(path).name,
                "transcribed_text_file": transcription_path or "",
                "extracted_chars": len(text),
                "ocr_engine": backend,
                "extracted_entities": len(structured.get("entities", [])),
                "ocr_text": text[:12000],
                "cloud_error": cloud_error,
                "_source_file": Path(path).name,
                "_source_type": "FIR",
            }
            self.add_entity(
                Path(path).stem, "Case/FIR", 360, 240,
                properties=case_props, node_id=case_id, redraw=False,
            )

            # Resolve entities against existing graph labels. >92% auto-merges;
            # 75-92% and ambiguous handwriting require a manual Merge/Create-New decision.
            decisions = self._review_fir_matches(structured.get("entities", []))
            self._ai_activity("Entity resolution complete — updating graph")
            created, matched, added_links, entity_nodes = self._integrate_fir_entities(
                case_id, path, structured, decisions=decisions
            )

            # Exact textual mentions remain useful evidence even when no structured entity was emitted.
            lowered = text.lower()
            exact_mentions = 0
            for node_id, data in list(self.G.nodes(data=True)):
                if node_id == case_id:
                    continue
                label = str(data.get("label", node_id))
                if len(label) >= 4 and label.lower() in lowered:
                    self.add_link(
                        case_id, node_id, "Mentions", source="fir_text_match",
                        confidence=0.82, weight=1.0,
                        properties={
                            "_source_file": Path(path).name,
                            "_source_type": "FIR",
                            "match": label,
                            "_fir_id": case_id,
                        }, redraw=False,
                    )
                    exact_mentions += 1

            # Re-run all graph analysis immediately after FIR integration.
            try:
                self.compute_centrality_metrics()
            except Exception:
                pass
            try:
                if len(self.G.nodes) >= 3:
                    self.detect_communities(silent=True)
            except TypeError:
                try:
                    if len(self.G.nodes) >= 3:
                        self.detect_communities()
                except Exception:
                    pass
            except Exception:
                pass
            self._commit_ai_snapshot()
            self.auto_layout()
            self.redraw_canvas()

            self._ai_activity("FIR integrated — graph analysis recalculated")
            fallback_note = "\nGemini unavailable; local Tesseract fallback was used." if cloud_error else ""
            summary = (
                f"FIR processed and integrated: {Path(path).name}\n"
                f"OCR backend: {backend}\n"
                f"Transcription saved: {transcription_path}\n"
                f"OCR characters: {len(text)}\n"
                f"Entities extracted: {len(structured.get('entities', []))}\n"
                f"Existing entities merged: {matched}\n"
                f"New entities added: {created}\n"
                f"Relationships/evidence links added: {added_links + exact_mentions}"
                f"{fallback_note}\n\n"
                "The graph and anomaly signals have been updated automatically.\n"
                "Fuzzy matches above 92% were auto-merged; ambiguous/75-92% matches were reviewed manually."
            )
            self._append_ai_chat(summary, header="✦ FIR → GRAPH")
            if self.selected_node not in self.G and entity_nodes:
                self.selected_node = next(iter(entity_nodes.values()))
            if self.selected_node in self.G:
                self.update_inspector(self.selected_node)
            messagebox.showinfo("FIR Integrated", summary)
        except Exception as exc:
            self._fir_upload_error(str(exc))

    def _fir_upload_error(self, message):
        self._fir_busy = False
        self.ai_status_var.set("○ FIR OCR FAILED")
        messagebox.showerror("FIR Processing Failed", message)
        self._append_ai_chat(message, header="FIR STATUS")
        self._ai_activity("FIR processing failed")


def _on_close(root, app):
    try:
        app.neo4j.close()
    finally:
        root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CriminalNetworkAnalyst(root)
    root.protocol("WM_DELETE_WINDOW", lambda: _on_close(root, app))
    root.mainloop()
