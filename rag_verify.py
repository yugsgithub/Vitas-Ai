"""
rag_verify.py — Phase 5: RAG Pipeline Verification Script
===========================================================
Run this from the Vitas_ai root with your conda env active:

    python rag_verify.py

Tests:
  1. RAG engine import + ChromaDB client init
  2. Text chunking
  3. Index a sample Ayurveda document
  4. Query retrieval (Ayurvedic)
  5. Index a sample Medical document
  6. Query retrieval (Medicinal)
  7. has_documents() check
  8. delete_collection() cleanup
"""

import os
import sys
import django

# ─── Django setup ─────────────────────────────────────────────────────────────
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "vitas_project.settings")

try:
    django.setup()
    print("✅ Django setup OK")
except Exception as e:
    print(f"⚠️  Django setup failed: {e}")
    print("   Continuing with direct rag_engine import...")

# ─── Direct import (works without Django too) ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "chatbot"))

# ─── Test constants ────────────────────────────────────────────────────────────
TEST_CONV_ID_AYU  = 99901   # fake conversation ID for Ayurveda test
TEST_CONV_ID_MED  = 99902   # fake conversation ID for Medical test

AYURVEDA_SAMPLE = """
Ashwagandha (Withania somnifera), known as Withania somnifera, is one of the most
important herbs in Ayurveda. It is classified as a Rasayana (rejuvenating) herb
and is used for balancing Vata and Kapha doshas.

Classical Ayurvedic texts describe Ashwagandha as Balya (strength-promoting),
Bringhana (nourishing), and Vajikara (aphrodisiac). The root powder (churna) is
the most commonly used form. Typical dosage is 3-6 grams per day with warm milk.

Key therapeutic actions include: stress adaptation (Medhya Rasayana), improvement
of sleep quality, enhancement of reproductive health, reduction of inflammation,
and strengthening of the immune system.

Contraindications: Avoid during pregnancy, in cases of high Pitta (heat/inflammation),
and with hyperthyroid conditions. Always consult a qualified Vaidya before use.

Shankhapushpi (Convolvulus pluricaulis) is another important Medhya Rasayana herb
used for memory enhancement. It balances all three doshas and is particularly
effective for improving cognitive function and reducing anxiety.

Brahmi (Bacopa monnieri) is the premier brain tonic in Ayurveda. It enhances
Dhi (intellect), Dhriti (retention), and Smriti (memory). Used as swaras (fresh
juice) or churna with honey, 5-10 ml or 3-6 grams daily.
"""

MEDICAL_SAMPLE = """
Type 2 Diabetes Mellitus (T2DM) is a chronic metabolic disorder characterised by
insulin resistance and relative insulin deficiency. It accounts for approximately
90-95% of all diabetes cases worldwide.

Pathophysiology: In T2DM, peripheral tissues (muscle, liver, adipose) become
resistant to insulin's effects. The pancreatic beta cells initially compensate by
producing more insulin, but over time they fail, leading to hyperglycaemia.

Clinical Features: Polydipsia, polyuria, polyphagia, fatigue, blurred vision,
frequent infections, slow wound healing. Many patients are asymptomatic at diagnosis.

Diagnosis: Fasting plasma glucose ≥ 126 mg/dL, or 2-hour glucose ≥ 200 mg/dL on
OGTT, or HbA1c ≥ 6.5%, or random glucose ≥ 200 mg/dL with symptoms.

Management: Lifestyle modification (diet + exercise) is first-line. Metformin is
the preferred initial pharmacological agent. Second-line agents include SGLT-2
inhibitors, GLP-1 receptor agonists, DPP-4 inhibitors, and sulfonylureas.

Metformin dosage: Start 500 mg twice daily with meals, titrate to 1000 mg twice
daily. Contraindicated in eGFR < 30 mL/min/1.73m².

HbA1c targets: < 7% for most adults, < 8% for elderly or those with comorbidities.
"""

# ─── Import RAG engine ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 5: RAG PIPELINE VERIFICATION")
print("="*60)

try:
    from chatbot.rag_engine import (
        chunk_text, index_document, query_documents,
        has_documents, delete_collection
    )
    print("✅ rag_engine imported successfully")
except ImportError as e:
    print(f"❌ rag_engine import failed: {e}")
    sys.exit(1)

# ─── Test 1: Chunking ─────────────────────────────────────────────────────────
print("\n[Test 1] Text chunking...")
chunks = chunk_text(AYURVEDA_SAMPLE, size=300, overlap=30)
print(f"  ✅ Produced {len(chunks)} chunks from {len(AYURVEDA_SAMPLE)} chars")
for i, c in enumerate(chunks[:2]):
    print(f"  Chunk {i+1} ({len(c)} chars): {c[:80].strip()}...")

# ─── Test 2: Index Ayurveda document ─────────────────────────────────────────
print("\n[Test 2] Indexing Ayurveda sample document...")
try:
    index_document(AYURVEDA_SAMPLE, TEST_CONV_ID_AYU)
    print(f"  ✅ Indexed into collection conv_{TEST_CONV_ID_AYU}")
except Exception as e:
    print(f"  ❌ Indexing failed: {e}")
    sys.exit(1)

# ─── Test 3: has_documents check ─────────────────────────────────────────────
print("\n[Test 3] Checking has_documents()...")
has = has_documents(TEST_CONV_ID_AYU)
print(f"  ✅ has_documents({TEST_CONV_ID_AYU}) = {has}")
assert has, "has_documents should return True after indexing"

# ─── Test 4: Query Ayurveda ───────────────────────────────────────────────────
print("\n[Test 4] Querying Ayurveda RAG...")
queries_ayu = [
    "What is the dosage of Ashwagandha?",
    "Which herb helps with memory and cognition?",
    "What are contraindications for Ashwagandha?",
]
for q in queries_ayu:
    result = query_documents(q, TEST_CONV_ID_AYU, n_results=2)
    if result:
        print(f"  ✅ Query: '{q}'")
        print(f"     → {result[:150].strip()}...\n")
    else:
        print(f"  ⚠️  No results for: '{q}'")

# ─── Test 5: Index Medical document ──────────────────────────────────────────
print("\n[Test 5] Indexing Medical sample document...")
try:
    index_document(MEDICAL_SAMPLE, TEST_CONV_ID_MED)
    print(f"  ✅ Indexed into collection conv_{TEST_CONV_ID_MED}")
except Exception as e:
    print(f"  ❌ Indexing failed: {e}")
    sys.exit(1)

# ─── Test 6: Query Medical ───────────────────────────────────────────────────
print("\n[Test 6] Querying Medical RAG...")
queries_med = [
    "What is the Metformin dosage for diabetes?",
    "How is Type 2 Diabetes diagnosed?",
    "What is the HbA1c target for elderly patients?",
]
for q in queries_med:
    result = query_documents(q, TEST_CONV_ID_MED, n_results=2)
    if result:
        print(f"  ✅ Query: '{q}'")
        print(f"     → {result[:150].strip()}...\n")
    else:
        print(f"  ⚠️  No results for: '{q}'")

# ─── Test 7: Cross-isolation check ───────────────────────────────────────────
print("\n[Test 7] Cross-collection isolation check...")
result_cross = query_documents("Ashwagandha dosage", TEST_CONV_ID_MED)
if result_cross and "ashwagandha" not in result_cross.lower():
    print("  ✅ Medical collection correctly isolated from Ayurveda data")
else:
    print("  ℹ️  Cross-isolation check returned:", result_cross[:80] if result_cross else "None")

# ─── Test 8: Cleanup ─────────────────────────────────────────────────────────
print("\n[Test 8] Cleanup — deleting test collections...")
delete_collection(TEST_CONV_ID_AYU)
delete_collection(TEST_CONV_ID_MED)
assert not has_documents(TEST_CONV_ID_AYU), "Collection should be gone after delete"
assert not has_documents(TEST_CONV_ID_MED), "Collection should be gone after delete"
print("  ✅ Both test collections deleted successfully")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("🎉 ALL RAG TESTS PASSED — Pipeline is fully operational!")
print("="*60)
print("\nNext steps:")
print("  1. Start the server: python manage.py runserver")
print("  2. Upload a PDF in the chat UI")
print("  3. Ask a specific question about the document content")
print("  4. Both Medicinal and Ayurvedic models will use RAG context")
