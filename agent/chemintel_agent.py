import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import nfp
import requests
import json
import re
import time
import warnings
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import quote

# 0. System configuration (suppress warnings and clean up logs)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama server
OLLAMA_MODEL_NAME = "llama3"
LLM_OPTIONS = {"temperature": 0.0}                  # deterministic decoding for every LLM call
PUBCHEM_CACHE = {}
VERBOSE = True

# Base path for GNN model artifacts
GNN_BASE_PATH = os.path.join('..', 'gnn', 'models')  # path to the pretrained GNN artifacts

# Target properties and the exact PubChem PUG-View TOC headings used to retrieve them
HEADING_MAP = {
    "Molecular Formula": ["Molecular Formula"],
    "Molecular Weight": ["Molecular Weight"],
    "Canonical SMILES": ["Canonical SMILES", "SMILES"],
    "Isomeric SMILES": ["Isomeric SMILES", "SMILES"],
    "InChI": ["InChI"],
    "InChIKey": ["InChIKey"],
    "IUPAC Name": ["IUPAC Name"],
    "XLogP3": ["XLogP3", "XLogP3-AA"],
    "TPSA": ["Topological Polar Surface Area"],
    "Complexity": ["Complexity"],
    "Charge": ["Formal Charge"],
    "LogP": ["LogP"],
    "Boiling Point": ["Boiling Point"],
    "Melting Point": ["Melting Point"],
    "Flash Point": ["Flash Point"],
    "Solubility": ["Solubility"],
    "Density": ["Density"],
    "Vapor Density": ["Vapor Density"],
    "Vapor Pressure": ["Vapor Pressure"],
    "Viscosity": ["Viscosity"],
    "Stability": ["Stability/Shelf Life"],
    "Auto-Ignition": ["Autoignition Temperature"],
    "Heat of Combustion": ["Heat of Combustion"],
    "Heat of Vaporization": ["Heat of Vaporization"],
    "Record Description": ["Record Description"],
}
ALL_TARGET_PROPS = list(HEADING_MAP.keys())
GNN_TARGETS = ["Boiling Point", "Vapor Pressure", "Solubility", "Flash Point", "Heat of Combustion"]

# ==========================================
# 1. Required functions for running the GNN models (NFP)
# ==========================================
def atom_featurizer(atom):
    return (atom.GetSymbol(), atom.GetIsAromatic(), nfp.preprocessing.features.get_ring_size(atom, max_size=6),
            atom.GetDegree(), atom.GetTotalNumHs(includeNeighbors=True))

def bond_featurizer(bond, flipped=False):
    atoms = "{}-{}".format(bond.GetBeginAtom().GetSymbol() if not flipped else bond.GetEndAtom().GetSymbol(),
                            bond.GetEndAtom().GetSymbol() if not flipped else bond.GetBeginAtom().GetSymbol())
    btype = str(bond.GetBondType())
    ring = nfp.preprocessing.features.get_ring_size(bond, max_size=6) if bond.IsInRing() else None
    return (atoms, btype, ring)

# The fitted preprocessors were pickled in a notebook, so they look up the featurizers in __main__.
# Registering them there lets this file be imported from any script (e.g. `import chemintel_agent`).
import __main__
for _f in (atom_featurizer, bond_featurizer):
    if not hasattr(__main__, _f.__name__):
        setattr(__main__, _f.__name__, _f)

def build_gnn_model(preprocessor, model_type="simple"):
    atom = layers.Input(shape=[None], dtype=tf.int32, name='atom')
    bond = layers.Input(shape=[None], dtype=tf.int32, name='bond')
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int32, name='connectivity')
    num_features = 128
    atom_state = layers.Embedding(preprocessor.atom_classes, num_features, mask_zero=True)(atom)
    bond_state = layers.Embedding(preprocessor.bond_classes, num_features, mask_zero=True)(bond)
    atom_mean = layers.Embedding(preprocessor.atom_classes, 1, mask_zero=True)(atom)
    for _ in range(8):
        new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity])
        bond_state = layers.Add()([bond_state, new_bond_state])
        new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity])
        atom_state = layers.Add()([atom_state, new_atom_state])
        atom_state = layers.LayerNormalization()(atom_state)
    if model_type == "simple":
        atomwise_prediction = layers.Dense(1)(atom_state)
        atomwise_prediction = layers.Add()([atomwise_prediction, atom_mean])
        mol_prediction = layers.GlobalAveragePooling1D()(atomwise_prediction)
    else:
        atomwise_prediction = layers.Dense(64, activation='relu')(atom_state)
        atomwise_prediction = layers.Dropout(0.1)(atomwise_prediction)
        atomwise_prediction = layers.Dense(1)(atomwise_prediction)
        atomwise_prediction = layers.Add()([atomwise_prediction, atom_mean])
        mol_prediction = layers.GlobalAveragePooling1D()(atomwise_prediction)
        mol_prediction = layers.Dense(64, activation='relu')(mol_prediction)
        mol_prediction = layers.Dropout(0.1)(mol_prediction)
        mol_prediction = layers.Dense(1)(mol_prediction)
    return tf.keras.Model([atom, bond, connectivity], mol_prediction)

# ==========================================
# 2. Utilities, LLM calls, and chemical resolver (Steps 1-2)
# ==========================================
def print_step(title: str, content: str = ""):
    if not VERBOSE: return
    print(f"\n🔵 [{title}]")
    if content: print(f"   {content}")

def print_sub(content: str):
    if VERBOSE: print(f"   └─ {content}")

def clean_text(text: str) -> str:
    return text.lower().replace(" ", "").replace("-", "")

def query_llm(prompt: str, timeout: int = 120) -> str:
    payload = {"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False, "options": LLM_OPTIONS}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        return r.json().get("response", "").strip() if r.status_code == 200 else ""
    except Exception:
        return ""

def clean_extracted_term(raw: str) -> str:
    """Strip LLM preambles such as 'The chemical compound name is:' and surrounding quotes/punctuation."""
    lines = [l.strip() for l in raw.replace('"', '').replace('`', '').splitlines() if l.strip()]
    if not lines:
        return ""
    term = lines[-1]
    m = re.match(r"^(?:the\s+)?(?:extracted\s+)?(?:chemical\s+)?(?:compound|term|name|smiles)[^:]*:\s*(.+)$", term, re.IGNORECASE)
    if m:
        term = m.group(1)
    return term.strip().rstrip('.?!').strip("'").strip()

def rule_based_candidates(query: str) -> List[str]:
    """Deterministic fallback candidates for the compound term (used when the LLM output is not grounded)."""
    q = query.strip().rstrip('?.! ')
    out = []
    m = re.search(r"\b(?:of|for)\s+(?:an?\s+|the\s+)?(.+)$", q, re.IGNORECASE)
    if m: out.append(m.group(1).strip())
    for sep in [':', ',', '?', '(']:
        if sep in q:
            out.append(q.split(sep)[0].strip())
    m = re.search(r"\(([^)]+)\)", q)
    if m: out.append(m.group(1).strip())
    out += [tok for tok in q.split() if re.fullmatch(r"[A-Za-z0-9@+\-\[\]\(\)=#$/\\%.]{3,}", tok) and re.search(r"[=#\[\]()]", tok)]
    m = re.search(r"\b(?:describes?|does|is|are|about)\s+(?:an?\s+|the\s+)?(.+?)(?:\s+(?:boil|melt|weigh|have|has))?$", q, re.I)
    if m: out.append(m.group(1).strip())
    words = re.findall(r"[\w\-,()\[\]]+", q)
    for n in (2, 1, 3):                      # trailing word spans, e.g. '... describes diethyl ether'
        if len(words) >= n: out.append(" ".join(words[-n:]))
    seen, res = set(), []
    for c in out:
        c = re.sub(r"\s+(molecule|compound)$", "", c, flags=re.IGNORECASE).strip()
        if c and c.lower() not in seen:
            seen.add(c.lower()); res.append(c)
    return res

def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def add_celsius(value: str) -> str:
    """Append deterministic °C conversions to °F values so that the LLM never performs arithmetic."""
    def conv(m):
        f = float(m.group(1)); return f"{(f - 32) * 5 / 9:.1f} °C ({m.group(0)})"
    return re.sub(r"(-?\d+(?:\.\d+)?)\s*°F", conv, str(value))

def extract_intent(query: str) -> str:
    """LLM-based identification of the requested property (one of ALL_TARGET_PROPS, or ALL for a full report)."""
    options = ", ".join(ALL_TARGET_PROPS)
    prompt = (f"A user asked: '{query}'\n"
              f"Which ONE chemical property is requested? Choose exactly one item from this list: {options}.\n"
              "Spelling mistakes, abbreviations, and synonyms may occur (e.g., 'mol. wt.' = Molecular Weight, "
              "'boiling temp' = Boiling Point, 'how heavy' = Molecular Weight). "
              "If the question does NOT mention any specific property (for example 'What is CID 3283?', 'Tell me about "
              "aspirin', or a request for information or a report in any language), answer ALL. "
              "Reply with the item only.")
    ans = query_llm(prompt).strip().strip('."\'')
    chosen = "ALL"
    for prop in ALL_TARGET_PROPS:
        if clean_text(prop) == clean_text(ans):
            chosen = prop; break
    else:
        for prop in ALL_TARGET_PROPS:
            if clean_text(prop) in clean_text(ans):
                chosen = prop; break
    # Grounding check: a specific property is accepted only if the query contains a matching cue.
    if chosen != "ALL" and not property_mentioned(chosen, query):
        return "ALL"
    return chosen

PROPERTY_CUES = {
    "Molecular Formula": ["formula", "분자식", "fórmula", "分子式"],
    "Molecular Weight": ["molecular weight", "molecularweight", "mol wt", "mw", "weight", "heavy", "molar mass",
                         "분자량", "peso molecular", "masa molar", "分子量"],
    "Canonical SMILES": ["smiles"], "Isomeric SMILES": ["isomeric"],
    "InChI": ["inchi"], "InChIKey": ["inchikey", "inchi key"], "IUPAC Name": ["iupac"],
    "XLogP3": ["xlogp", "xlogp3"], "TPSA": ["tpsa", "polar surface"], "Complexity": ["complexity"],
    "Charge": ["charge"], "LogP": ["logp", "log p", "kow", "partition"],
    "Boiling Point": ["boiling", "boil", "끓는점", "ebullición", "沸点"],
    "Melting Point": ["melting", "melt", "녹는점", "fusión", "熔点"],
    "Flash Point": ["flash", "인화점", "inflamación", "闪点"],
    "Solubility": ["solubility", "soluble", "dissolve", "용해도", "solubilidad", "溶解度"],
    "Density": ["density", "밀도", "densidad", "密度"],
    "Vapor Density": ["vapor density", "vapour density", "증기밀도"],
    "Vapor Pressure": ["vapor pressure", "vapour pressure", "pressure", "증기압", "presión de vapor", "蒸气压"],
    "Viscosity": ["viscosity", "점도", "viscosidad", "粘度"], "Stability": ["stability", "stable", "안정성"],
    "Auto-Ignition": ["ignition", "autoignition", "자연발화"],
    "Heat of Combustion": ["combustion", "연소열", "combustión", "燃烧热"],
    "Heat of Vaporization": ["vaporization", "vaporisation", "증발열", "汽化热"],
    "Record Description": ["description", "describe"],
}

def property_mentioned(prop: str, query: str) -> bool:
    """True if the query contains a cue for `prop`, allowing small spelling errors (e.g., 'boilling', 'presure')."""
    import difflib
    q = query.lower()
    q_compact = re.sub(r"[^\w]", "", q)
    tokens = re.findall(r"[\w]+", q)
    for cue in PROPERTY_CUES.get(prop, []):
        if cue in q or cue.replace(" ", "") in q_compact:
            return True
        for cw in cue.split():
            if len(cw) >= 5 and any(difflib.SequenceMatcher(None, cw, t).ratio() >= 0.8 for t in tokens):
                return True
    return False

class ChemicalResolver:
    def __init__(self):
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"

    def resolve(self, query: str) -> Tuple[Optional[str], str, str]:
        print_step("Step 1: Query Analysis and Entity Extraction", f"User query: \"{query}\"")
        cid_match = re.search(r"CID\s*[:#]?\s*(\d+)", query, re.IGNORECASE)
        if cid_match: return cid_match.group(1), "CID", cid_match.group(1)

        prompt = (f"Extract chemical term. USER: '{query}' RULES: Copy the compound name or SMILES string exactly as "
                  "written in the USER text. Do not convert names to SMILES. Output only the term, no extra words.")
        llm_term = clean_extracted_term(query_llm(prompt))
        # Grounding check: an extracted term must literally occur in the query; otherwise it is discarded as
        # hallucinated and rule-based candidates are used instead.
        candidates = []
        if llm_term and llm_term.lower() in query.lower():
            candidates.append(llm_term)
        candidates += [c for c in rule_based_candidates(query) if c not in candidates]
        if not candidates: return None, "Fail", llm_term

        for term in candidates:
            print_step("Step 2: CID Conversion and Validation", f"Converting the term '{term}' to a PubChem CID.")
            cid, itype = self.lookup(term)
            if cid: return cid, itype, term

        # Spelling correction: accepted only if the corrected name is within a small edit distance of the typed term
        # and resolves in PubChem, so the LLM cannot substitute a different compound.
        for term in candidates[:2]:
            fixed = clean_extracted_term(query_llm(
                f"Correct the spelling of this chemical compound name: '{term}'. Output only the corrected name."))
            if fixed and fixed.lower() != term.lower() and edit_distance(fixed.lower(), term.lower()) <= 2:
                cid, itype = self.lookup(fixed)
                if cid:
                    print_sub(f"Spelling corrected: '{term}' -> '{fixed}'")
                    return cid, "NAME (spelling-corrected)", fixed
        return None, "Fail", candidates[0]

    def lookup(self, term: str):
        for itype in ["name", "smiles"]:
            try:
                if itype == "smiles":
                    r = requests.post(f"{self.base_url}/smiles/cids/JSON", data={"smiles": term}, timeout=10)
                else:
                    r = requests.get(f"{self.base_url}/name/{quote(term)}/cids/JSON", timeout=10)
                if r.status_code == 200:
                    cid = r.json()["IdentifierList"]["CID"][0]
                    if cid: return str(cid), itype.upper()
            except Exception: continue
        return None, None

# ==========================================
# 3. Data collection (Step 3)
# ==========================================
def fetch_pubchem_data(cid: str) -> Optional[Dict]:
    if cid in PUBCHEM_CACHE: return PUBCHEM_CACHE[cid]
    try:
        r = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON", timeout=30)
        if r.status_code == 200:
            PUBCHEM_CACHE[cid] = r.json()
            return PUBCHEM_CACHE[cid]
    except Exception: return None

def find_section_recursive(sections: List[Dict], target_heading: str) -> Optional[str]:
    """Return the first value under a PUG-View section whose TOC heading matches target_heading exactly."""
    clean_target = clean_text(target_heading)
    for section in sections:
        if clean_text(section.get("TOCHeading", "")) == clean_target:
            for info in section.get("Information", []):
                val = info.get("Value", {})
                for k in ["StringWithMarkup", "Number", "String"]:
                    v = val.get(k)
                    if v:
                        res = v[0].get('String') if k == "StringWithMarkup" else (v[0] if isinstance(v, list) else v)
                        return f"{res} {val.get('Unit', '')}".strip()
        found = find_section_recursive(section.get("Section", []), target_heading)
        if found: return found
    return None

def execute_full_scan(cid: str) -> Dict[str, Any]:
    print_step("Step 3: Detailed PubChem Data Scan", f"Retrieving information for CID {cid}.")
    data = fetch_pubchem_data(cid)
    results = {}
    if not data: return results

    results["Common Name"] = data.get("Record", {}).get("RecordTitle", "Unknown")
    print_sub(f"Compound name identified: {results['Common Name']}")
    root_sections = data.get("Record", {}).get("Section", [])
    for prop, headings in HEADING_MAP.items():
        val = None
        for h in headings:
            val = find_section_recursive(root_sections, h)
            if val: break
        results[prop] = val if val else "Not Found"
    found = sum(1 for p in ALL_TARGET_PROPS if results[p] != "Not Found")
    print_sub(f"Data collection complete: {found} properties found, {len(ALL_TARGET_PROPS) - found} not found.")
    return results

# ==========================================
# 4. Missing-value analysis and triggered GNN prediction (Step 4)
# ==========================================
GNN_CONFIGS = {
    "Boiling Point": {"dir": "bp_model_artifacts", "pre": "bp_preprocessor.pkl", "w": "bp_model.weights.h5", "type": "complex", "u": "°C", "k_to_c": True},
    "Vapor Pressure": {"dir": "vp_model_artifacts", "pre": "vp_preprocessor.pkl", "w": "vp_model.weights.h5", "type": "simple", "u": "log10(VP / mmHg)", "k_to_c": False},
    "Solubility": {"dir": "sol_model_artifacts", "pre": "sol_preprocessor.pkl", "w": "sol_model.weights.h5", "type": "simple", "u": "log10(S / mol L-1)", "k_to_c": False},
    "Flash Point": {"dir": "fp_model_artifacts", "pre": "fp_preprocessor.pkl", "w": "fp_model.weights.h5", "type": "complex", "u": "°C", "k_to_c": True},
    "Heat of Combustion": {"dir": "hoc_model_artifacts", "pre": "hoc_preprocessor.pkl", "w": "hoc_model.weights.h5", "type": "simple", "u": "MJ/mol (absolute value)", "k_to_c": False},
}

def execute_prediction(results: Dict[str, Any], targets: Optional[List[str]] = None) -> Dict[str, Any]:
    """Trigger GNN prediction only for requested properties that are missing in PubChem and supported by a model."""
    targets = GNN_TARGETS if targets is None else [t for t in targets if t in GNN_TARGETS]
    print_step("Step 4: Missing-Value Analysis and Triggered GNN Prediction")
    smiles = results.get("Canonical SMILES")
    if not smiles or smiles == "Not Found": smiles = results.get("Isomeric SMILES")
    for name in targets:
        if results.get(name) != "Not Found":
            print_sub(f"{name}: PubChem value available ({results.get(name)})"); continue
        if not smiles or smiles == "Not Found":
            print_sub(f"{name}: missing, but no structure (SMILES) is available for prediction."); continue
        cfg = GNN_CONFIGS[name]
        print_sub(f"{name}: missing in PubChem -> running the GNN surrogate...")
        try:
            path = os.path.join(GNN_BASE_PATH, cfg['dir'])
            with open(os.path.join(path, cfg['pre']), 'rb') as f:
                preprocessor = pickle.load(f)
            model = build_gnn_model(preprocessor, model_type=cfg['type'])
            dummy = {k: np.expand_dims(v, axis=0) for k, v in preprocessor.construct_feature_matrices('C', train=False).items()}
            _ = model(dummy, training=False)
            model.load_weights(os.path.join(path, cfg['w']))
            inputs = preprocessor.construct_feature_matrices(smiles, train=False)
            raw_pred = model({k: np.expand_dims(v, axis=0) for k, v in inputs.items()}, training=False).numpy()[0][0]
            final_val = raw_pred - 273.15 if cfg['k_to_c'] else raw_pred
            results[name] = f"{final_val:.4f} {cfg['u']} (AI-predicted value)"
            print_sub(f"   ✅ Prediction complete: {results[name]}")
        except Exception as e:
            print_sub(f"   ❌ Prediction failed for {name}: {e}")
    return results

# ==========================================
# 5. Source-aware answer / report generation (Step 5)
# ==========================================
def generate_answer(query: str, cid: str, data: Dict[str, Any], intent: str) -> str:
    print_step("Step 5: Source-Aware Answer Generation")
    lang_rule = ("Write your answer in the same language as the user's question. Keep chemical names, formulas, "
                 "identifiers, numbers, and units unchanged.")
    if intent == "ALL":
        shown = {k: add_celsius(v) for k, v in data.items()}  # deterministic °F -> °C before the LLM sees the data
        prompt = (f"You are a senior chemical researcher. The user asked: '{query}'. Write a clear report for CID {cid}. "
                  "Use only the supplied data, distinguish PubChem-retrieved values from AI-predicted values, and do not "
                  "invent unavailable information. Temperatures are already given in °C with the original °F in "
                  "parentheses; copy numbers and units exactly as written and never change a unit. "
                  f"{lang_rule} Data: {json.dumps(shown, indent=2, ensure_ascii=False)}")
    else:
        value = add_celsius(data.get(intent, "Not Found"))
        source =("GNN surrogate prediction (not an experimental value)" if "AI-predicted" in str(value)
                  else ("not available" if value == "Not Found" else "PubChem"))
        prompt = ("You answer chemical property questions using only the record below.\n"
                  f"RECORD\nCompound: {data.get('Common Name')} (PubChem CID {cid})\nProperty: {intent}\n"
                  f"Value: {value}\nSource: {source}\n\n"
                  f"QUESTION: {query}\n\n"
                  "Write the answer in one or two sentences: state the value exactly as given in the record and its "
                  "source. Temperatures in the record are already given in °C (original °F in parentheses); copy them "
                  "as written and do not compute conversions yourself. If the "
                  "value is 'Not Found', say that it is "
                  f"not available instead of guessing. {lang_rule} Output only the answer sentences.\nANSWER:")
    answer = query_llm(prompt, timeout=300)
    if VERBOSE:
        print("=" * 70); print(answer); print("=" * 70)
    return answer

# ==========================================
# Main pipeline
# ==========================================
def run_pipeline(user_query: str) -> Dict[str, Any]:
    resolver = ChemicalResolver()
    target_cid, id_type, term = resolver.resolve(user_query)
    if not target_cid:
        print_sub("The compound could not be resolved to a PubChem CID. The process has been terminated.")
        return {"query": user_query, "cid": None, "entity": term, "intent": None, "value": None, "answer": None}
    print_sub(f"Conversion complete: {id_type} '{term}' -> CID {target_cid}")

    intent = extract_intent(user_query)
    print_sub(f"Requested property: {intent}")
    data = execute_full_scan(target_cid)
    data = execute_prediction(data, targets=None if intent == "ALL" else [intent])
    answer = generate_answer(user_query, target_cid, data, intent)
    return {"query": user_query, "cid": target_cid, "entity": term, "intent": intent,
            "value": data.get(intent) if intent != "ALL" else None, "answer": answer}

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" 🧪 [Chemical AI Agent: GNN-Integrated Workflow]")
    print(" 💡 Type 'exit', 'quit', 'escape', or 'q' (or press Ctrl + C) to stop.")
    print("=" * 60)
    while True:
        try:
            u_input = input("\n📝 Query >> ").strip()
            if not u_input:
                continue
            if u_input.lower() in ['exit', 'quit', 'escape', 'q']:
                print("\n👋 The program has been closed safely.")
                break
            run_pipeline(u_input)
        except KeyboardInterrupt:
            print("\n\n🛑 Ctrl+C detected. The program is shutting down safely. 👋")
            break
        except Exception as e:
            print(f"\n⚠️ A critical error occurred: {e}")
