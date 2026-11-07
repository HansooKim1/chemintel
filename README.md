# ChemIntel: Retrieval-Grounded LLM Orchestration with Triggered GNN Surrogates

This repository contains the core implementation accompanying the manuscript

> **Hallucination-Mitigated Chemical Property Acquisition via Retrieval-Grounded LLMs and Triggered GNN Surrogates**
> Hansoo Kim, Jooyeon Lee, Byounghwak Lee\*, Keunhong Jeong\*

The framework answers chemical property queries without letting the language model generate property values from its own memory:

1. **Entity standardization:** a CID, compound name, or SMILES string in the query is resolved to a PubChem CID (PUG-REST).
2. **Intent identification:** the LLM identifies the requested property or a full report.
3. **PubChem retrieval:** the PUG-View record is scanned for a predefined set of target properties.
4. **Missing-property detection and triggered GNN prediction:** if the requested property is missing in PubChem and is one of the five supported targets, a pretrained graph neural network estimates it.
5. **Source-aware synthesis:** a local LLM writes the answer in the language of the query from the source-labeled property set.

---

## Contents

```text
chemintel/
├── agent/
│   ├── chemintel_agent.ipynb      # end-to-end workflow, interactive
│   └── chemintel_agent.py         # same code as an importable module
├── gnn/models/                    # pretrained GNN surrogates (weights + fitted preprocessors)
├── examples/example_outputs.md    # example answers of the framework
├── requirements.txt
└── LICENSE
```

---

## Installation

Tested with Python 3.12 on Windows 10.

```bash
conda create -n chemintel python=3.12
conda activate chemintel
conda install -c conda-forge rdkit=2024.03.5
pip install -r requirements.txt
```

### Local LLM (Ollama)

The agent requires [Ollama](https://ollama.com) to serve Llama 3 8B locally; no cloud LLM service is required. Install Ollama, then download the model:

```bash
ollama pull llama3          # about 4.7 GB
ollama serve                # if the Ollama service is not already running
```

The agent connects to `http://localhost:11434` and uses the model `llama3`, as set by `OLLAMA_URL` and `OLLAMA_MODEL_NAME` in `agent/chemintel_agent.py`. All LLM calls use `temperature = 0.0`. PubChem access requires an internet connection.

---

## Usage

Open and run `agent/chemintel_agent.ipynb` from inside the `agent/` folder, then type a query at the prompt:

```text
Query >> What is the flash point of CID 150000?
Query >> Provide a report on aspirin.
Query >> Tell me about CC(=O)OC1=CC=CC=C1C(=O)O.
```

Type `exit`, `quit`, or `q` to stop. `run_pipeline(query)` in `chemintel_agent.py` can also be called directly; it returns the resolved CID, the identified property, the source-labeled value, and the final answer. Example answers are given in `examples/example_outputs.md`.

| Label | Meaning |
|---|---|
| value with unit | retrieved from PubChem |
| `(AI-predicted value)` | estimated by a pretrained GNN surrogate, because PubChem had no value |
| `Not Found` | unavailable in PubChem and not supported by a GNN surrogate |

---

## GNN surrogate models

Five property-specific message-passing neural networks built with [NFP](https://github.com/NREL/nfp) on TensorFlow/Keras are provided in `gnn/models/`. They are the exact artifacts used for the results in the manuscript.

| Target | Training compounds | Unit reported by the agent |
|---|---:|---|
| Boiling point | 2,181 | °C |
| Vapor pressure | 3,826 | log10(VP / mmHg) |
| Aqueous solubility | 9,914 | log10(S / mol L⁻¹) |
| Flash point | 6,603 | °C |
| Heat of combustion | 5,920 | MJ/mol (absolute value) |

The training data were derived from Lee et al. (*J. Cheminform.* 2026, 18, 28), Jang et al. (*J. Cheminform.* 2026, 18, 18), and Kim et al. (*Green Chem.* 2024, 26, 10247). GNN outputs are surrogate estimates and should not be treated as experimentally verified values. PubChem records are updated over time, so retrieved values may differ from those reported in the manuscript.

---

## Citation

```text
H. Kim, J. Lee, B. Lee, K. Jeong. Hallucination-Mitigated Chemical Property Acquisition via
Retrieval-Grounded LLMs and Triggered GNN Surrogates. [Journal, year, DOI]
```

## License

Released under the MIT License (see `LICENSE`).

## Contact

Corresponding authors: Keunhong Jeong (doas1mind@sogang.ac.kr), Byounghwak Lee (lebaiii@kma.ac.kr)
