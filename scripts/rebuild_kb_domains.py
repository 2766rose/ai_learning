# -*- coding: ascii -*-
"""rebuild_kb_domains.py: rebuild KB with domain metadata (company/personal)."""
import asyncio, os, shutil

from ai_rag.core.vector_store import get_vector_store
from ai_rag.services.etl_service import ingest_document

STAGING = r"D:\ai_learning\data\rebuild_src"

COMPANY = [
    (r"D:\ai_learning\uploads\54c9e463-ceeb-4394-9a67-1e15a2c24eca.txt", "handbook.txt"),
    (r"D:\ai_learning\uploads\8577efbc-89d9-4671-9503-26aa68e282af.txt", "x9000.txt"),
    (r"D:\ai_learning\uploads\8900eb1b-2cc3-4e42-9c47-2e828f9859ef.txt", "order_sop.txt"),
    (r"D:\ai_learning\uploads\5c10f039-5512-4ecf-9e3d-1493b36e4029.md", "rag_handbook.md"),
    (r"D:\ai_learning\uploads\3de72ff7-bce6-4265-b4d4-56bd98d6d573.md", "qwen_plus.md"),
    (r"D:\ai_learning\uploads\11f49667-5323-4da3-b229-01820c77e24d.pdf", "expense.pdf"),
]
PERSONAL_SRC = "D:\\\u9762\u8bd5\u7406\u8bba\u7b14\u8bb0.docx"  # D:\??????.docx
PERSONAL_NAME = "interview.docx"


async def main():
    os.makedirs(STAGING, exist_ok=True)
    store = await get_vector_store()
    col = store._collection
    ids = col.get()["ids"]
    if ids:
        col.delete(ids=ids)
        print("cleared chunks:", len(ids))

    staged = []
    for src_path, name in COMPANY:
        if not os.path.exists(src_path):
            print("SKIP missing:", src_path)
            continue
        dst = os.path.join(STAGING, name)
        shutil.copy2(src_path, dst)
        staged.append((dst, name, "company"))
    if os.path.exists(PERSONAL_SRC):
        dst = os.path.join(STAGING, PERSONAL_NAME)
        shutil.copy2(PERSONAL_SRC, dst)
        staged.append((dst, "interview", "personal"))
    else:
        print("SKIP missing interview docx")

    for dst, label, domain in staged:
        try:
            n = await ingest_document(dst, {"task_id": "rebuild-" + label, "domain": domain, "source": label})
            print("ingested [%s] %s -> %d chunks" % (domain, label, n))
        except Exception as e:
            print("FAIL [%s] %s: %s" % (domain, label, e))

    try:
        shutil.rmtree(STAGING, ignore_errors=True)
    except Exception:
        pass
    print("final count:", await store.get_count())


asyncio.run(main())
