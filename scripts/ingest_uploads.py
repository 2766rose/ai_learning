import os
import logging
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

UPLOAD_DIR = r"D:\ai_learning\data\uploads"
CHROMA_PATH = r"D:\ai_learning\src\chroma_db"
COLLECTION_NAME = "knowledge_base"
MODEL_PATH = r"D:/models/text2vec-base-chinese"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def read_file_content(file_path: Path) -> str:
    """根据文件类型选择正确的读取方式"""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text.strip()
        except Exception as e:
            logger.error(f"PDF解析失败 {file_path.name}: {e}")
            return ""
    elif suffix in [".md", ".txt"]:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".docx":
        try:
            from docx import Document
            doc = Document(str(file_path))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            logger.error(f"DOCX解析失败 {file_path.name}: {e}")
            return ""
    else:
        logger.warning(f"不支持的文件格式: {file_path.name}")
        return ""

def main():
    upload_path = Path(UPLOAD_DIR)
    if not upload_path.exists():
        logger.error(f"上传目录不存在: {upload_path}")
        return

    all_texts, all_metas = [], []
    valid_suffixes = {".md", ".txt", ".pdf", ".docx"}
    valid_files = [f for f in upload_path.glob("**/*.*") if f.suffix.lower() in valid_suffixes]

    if not valid_files:
        logger.warning(f"在 {upload_path} 下未找到支持的文件({valid_suffixes})")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)

    for file_path in valid_files:
        content = read_file_content(file_path)
        if not content:
            logger.warning(f"⚠️ 文件内容为空，跳过: {file_path.name}")
            continue
        chunks = splitter.split_text(content)
        all_texts.extend(chunks)
        all_metas.extend([{"source": file_path.name}] * len(chunks))
        logger.info(f"📄 已处理: {file_path.name} → {len(chunks)} 个分块")

    if not all_texts:
        logger.error("没有生成任何有效分块，终止写入")
        return

    logger.info(f"🔄 加载模型: {MODEL_PATH}")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_PATH, model_kwargs={"device": "cpu"})

    logger.info(f"💾 写入向量库: {CHROMA_PATH}")
    vs = Chroma.from_texts(
        texts=all_texts, metadatas=all_metas,
        embedding=embeddings, persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )

    count = vs._collection.count()
    logger.info(f"✅ 入库完成！当前知识库有效向量数: {count}")
    if count == 0:
        logger.error("⚠️ 警告：写入后向量数仍为0，请检查模型或文本内容是否为空！")

if __name__ == "__main__":
    main()
