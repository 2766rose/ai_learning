# -*- coding: ascii -*-
"""cleanup_kb.py: remove junk (interview notes / hello-world / learning-plan) + duplicate chunks."""
import asyncio, sys
from collections import Counter

JUNK_SOURCE_PATTERNS = ["\u9762\u8bd5", "ctest"]
JUNK_SOURCES = {
    "kb-52c7cea1-499f-414e-b20f-182b9bfab40f.xlsx",
    "kb-aef56ec8-ddeb-454b-a76c-f650cefe431f.xlsx",
    "kb-2fab83c4-5189-49c1-a6ba-07cb68be64c3.txt",
    "kb-38b9c4c1-3da8-43dc-bf79-867bd57b7111.txt",
    "kb-bd284c7d-f873-47af-91de-cf24a9c102dd.txt",
    "ctest.txt",
}
JUNK_CONTENT_MARKERS = ["hello world", "hello upload test", "\u5468\u6b21 | \u5b66\u4e60\u4e3b\u9898"]

def is_junk(d):
    meta = d.get("metadata") or {}
    src = meta.get("source", "") or ""
    content = (d.get("content") or "")[:60]
    if src in JUNK_SOURCES:
        return True
    if any(p in src for p in JUNK_SOURCE_PATTERNS):
        return True
    return any(m in content for m in JUNK_CONTENT_MARKERS)

async def main():
    from ai_rag.core.vector_store import get_vector_store
    store = await get_vector_store()
    docs = await store.get_all_documents()
    print("total chunks:", len(docs))
    src_count = Counter((d.get("metadata") or {}).get("source", "?") for d in docs)
    print("\n=== sources ===")
    for s, c in src_count.most_common():
        print("  %4d  %s" % (c, s))
    junk_ids = [d["id"] for d in docs if is_junk(d)]
    junk_src = Counter((d.get("metadata") or {}).get("source", "?") for d in docs if is_junk(d))
    print("\n=== junk breakdown ===")
    for s, c in junk_src.most_common():
        print("  %4d  %s" % (c, s))
    seen, dup_ids = {}, []
    for d in docs:
        c = d.get("content", "")
        if c in seen:
            dup_ids.append(d["id"])
        else:
            seen[c] = d["id"]
    print("\n[junk] to delete:", len(junk_ids))
    print("[duplicate] to delete:", len(dup_ids))
    print("remaining after cleanup:", len(docs) - len(junk_ids) - len(dup_ids))
    if "--apply" not in sys.argv:
        print("\nDRY RUN: pass --apply to actually delete (stop backend first)")
        return
    col = store._collection
    if junk_ids:
        col.delete(ids=junk_ids)
    if dup_ids:
        col.delete(ids=dup_ids)
    print("deleted:", len(junk_ids) + len(dup_ids), "| remaining:", await store.get_count())
    try:
        from ai_rag.core.redis_stability import _get_client
        r = _get_client()
        n = 0
        for k in r.scan_iter("sc:*", count=200):
            r.delete(k); n += 1
        print("semantic cache cleared:", n)
    except Exception as e:
        print("cache clear skipped:", e)

asyncio.run(main())
