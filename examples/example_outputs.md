# Example outputs

Two answers produced by the framework with Llama 3 8B served locally through Ollama, copied unchanged from the evaluation runs reported in the manuscript.

## 1. Property report retrieved from PubChem

**Query:** `What is CID 3283?`

CID 3283: Diethyl Ether

**Common Name:** Diethyl Ether
**Molecular Formula:** C4H10O
**Molecular Weight:** 74.12 g/mol

**Physical Properties:**

* **Boiling Point:** 34.6 °C (94.3 °F) at 760 mmHg
* **Melting Point:** -116.3 °C (-177.3 °F)
* **Flash Point:** -45.0 °C (-49 °F)
* **Density:** 0.714 at 20.0 °C (68 °F) - Less dense than water; will float
* **Vapor Density:** 2.6 - Heavier than air; will sink (Relative to Air)
* **Vapor Pressure:** 442 mmHg at 20.0 °C (68 °F); 760 mmHG at 34.6 °C (94.3 °F)
* **Viscosity:** 0.2448 centipoise at 20 °C

**Chemical Properties:**

* **Stability:** Slowly oxidized by action of air, moisture, and light, with formation of peroxides.
* **Auto-Ignition:** 180.0 °C (356 °F)
* **Heat of Combustion:** -8.807 kcal/g
* **Heat of Vaporization:** 89.80 cal/g at 30 °C

**Solubility and Other Properties:**

* **Solubility:** Greater than or equal to 100 mg/mL at 20.0 °C (68 °F)
* **Record Description:** Diethyl ether appears as a clear colorless liquid with an anesthetic odor. Flash point -45.0 °C (-49 °F). Less dense than water and slightly soluble in water. Hence floats on water. Vapors are heavier than air. Used as a solvent and to make other chemicals.

**Predicted Properties:**

* **XLogP3:** 0.9
* **TPSA:** 9.2 Å²
* **Complexity:** 11.1
* **Charge:** 0
* **LogP:** log Kow= 0.89

## 2. Missing property completed by the GNN surrogate

PubChem has no vapor pressure record for 2-acetylpyrrole (CID 14079), so the vapor pressure surrogate was triggered and the value is labeled as a prediction.

**Query:** `What is the vapor pressure of 2-Acetylpyrrole?`

> The vapor pressure of 2-Acetylpyrrole is -2.4380 log10(VP / mmHg) (AI-predicted value).
