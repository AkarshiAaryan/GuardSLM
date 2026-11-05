# GuardSLM 100-Pair Benchmark: Dataset Structural Validation Report

**Validation Status:** 🟢 **PASS**  
**Checks Passed:** 13 / 13  
**Dataset Name:** `GuardSLM Browser-Action Authorization Benchmark`  
**Dataset Version:** `2.0.0-provisional`  

> [!IMPORTANT]
> **Provisional Gold Notice**: Structural validation confirms schema compliance, relational pair integrity, policy coverage, and screenshot mapping. Ground-truth gold labels remain provisional until independently adjudicated against the live platform fixture.

## 1. Verified Inventory Summary

| Benchmark Dimension | Verified Value | Expected Standard | Status |
| :--- | :---: | :---: | :---: |
| **Policy Clauses** | 36 | 36 | ✅ OK |
| **Source PDFs** | 6 | 6 | ✅ OK |
| **Individual Cases** | 200 | 200 | ✅ OK |
| **Matched Pairs** | 100 | 100 | ✅ OK |
| **Authorization Flip Pairs** | 80 | 80 | ✅ OK |
| **Invariant Control Pairs** | 20 | 20 | ✅ OK |
| **ALLOW Verdicts** | 97 | 97 | ✅ OK |
| **BLOCK Verdicts** | 95 | 95 | ✅ OK |
| **ASK Verdicts** | 8 | 8 | ✅ OK |

## 2. Invariant & Structural Checks

1. **Unique Case & Pair IDs:** All 200 case IDs and 100 pair IDs are globally unique without collisions.
2. **Strict Matched-Pair Completeness:** All 100 pairs contain exactly two cases (`-A` and `-B`).
3. **Complete Policy Book Coverage:** Every single one of the 200 model inputs contains the full 36-clause policy book.
4. **Exact 1-to-1 Model-Gold Alignment:** Model inputs and private gold annotations share identical case IDs.
5. **Policy Checks Cardinality:** All 200 gold annotations contain exactly 36 policy checks matching canonical policy IDs.
6. **Pair Flip & Invariant Verification:**
   - 80 authorization-flip pairs genuinely flip verdicts across the intervention boundary.
   - 20 invariant-control pairs preserve identical verdicts.
7. **Screenshot Provenance:** All 200 cases map to an authentic source PDF and 1-indexed page number in the manifest.

## 3. Discovered Issues

No critical structural or relational errors discovered. Dataset is structurally verified.
